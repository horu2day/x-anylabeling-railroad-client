"""
SAM 3.1 텍스트 프롬프트 자동 라벨링
=====================================================
GDINO 없이 SAM 3.1의 자체 텍스트 그라운딩 기능을 직접 사용.

  /v1/predict  params.text_prompt  ->  SAM3 내부 set_text_prompt() 호출
  marks(bbox) 없이 텍스트만으로 검출 + segmentation 한 번에 수행.

사용법:
    python tools/gdino_sam3_autolabel.py --input data/역사이미지/slope/ --output output/autolabel/slope/
    python tools/gdino_sam3_autolabel.py --input data/역사이미지/vertical/ --mode vertical
    python tools/gdino_sam3_autolabel.py --input data/tiles/ --tile_size 0  # 타일 없음

사전 조건:
    - start_server.bat 실행 (SAM 3.1 서버)
"""
import argparse
import base64
from pathlib import Path

import cv2
import numpy as np
import requests

# ── 설정 ──────────────────────────────────────────────────────────────────────
SAM3_SERVER   = "http://localhost:8000"
SAM3_MODEL_ID = "segment_anything_3"

# ── 프롬프트 파일 로드 ─────────────────────────────────────────────────────────
# prompts/*.txt 파일에서 읽음. 각 줄이 하나의 명사구.
# SAM3.1 은 쉼표로 구분된 각 명사구를 독립적으로 검출 후 합산.
_PROMPT_DIR = Path(__file__).parent.parent / "prompts"

_DEFAULT_SLOPE    = "railway catenary pole, overhead line support pole, catenary mast"
_DEFAULT_VERTICAL = "catenary pole top, pole cross arm, horizontal cantilever beam, bracket arm"
_DEFAULT_RAIL     = "railroad, railway"


def _load_prompt(filename: str, default: str) -> str:
    """prompts/<filename> 에서 명사구를 읽어 쉼표 구분 문자열로 반환.
    파일이 없거나 비어 있으면 default 반환."""
    path = _PROMPT_DIR / filename
    if path.exists():
        lines = [l.strip() for l in path.read_text(encoding="utf-8").splitlines()
                 if l.strip() and not l.startswith("#")]
        if lines:
            return ", ".join(lines)
    return default


def _get_prompts() -> tuple:
    """(pole_slope, pole_vertical, rail) 프롬프트 반환."""
    return (
        _load_prompt("pole_slope.txt",    _DEFAULT_SLOPE),
        _load_prompt("pole_vertical.txt", _DEFAULT_VERTICAL),
        _load_prompt("rail.txt",          _DEFAULT_RAIL),
    )


CLASS_MAP: dict = {}   # 기본값 0 (catenary_pole)
CLASS_NAMES = ["catenary_pole"]


def select_prompt(input_path: Path, mode: str) -> str:
    slope, vertical, _ = _get_prompts()
    if mode == "vertical":
        return vertical
    if mode == "slope":
        return slope
    parts = [p.lower() for p in input_path.parts]
    if any("vertical" in p for p in parts):
        print("  [auto] 수직 촬영 감지 -> vertical 프롬프트 사용")
        return vertical
    if any("slope" in p for p in parts):
        print("  [auto] 사선 촬영 감지 -> slope 프롬프트 사용")
        return slope
    print("  [auto] 촬영 방향 불명 -> slope 프롬프트 기본 사용")
    return slope


# ── 이미지 인코딩 ──────────────────────────────────────────────────────────────
def encode_image(image_bgr: np.ndarray, max_size: int = 2048) -> tuple:
    """이미지를 base64 JPEG로 인코딩. max_size 초과 시 축소. (b64, scale) 반환."""
    h, w = image_bgr.shape[:2]
    scale = 1.0
    if max_size > 0 and max(h, w) > max_size:
        scale = max_size / max(h, w)
        image_bgr = cv2.resize(image_bgr, (int(w * scale), int(h * scale)))
    _, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf).decode("utf-8"), scale


# ── SAM 3.1 텍스트 프롬프트 세그멘테이션 ──────────────────────────────────────
def sam3_text_segment(image_bgr: np.ndarray, text_prompt: str,
                      conf_threshold: float = 0.25,
                      sam_max_size: int = 2048) -> list:
    """SAM 3.1 텍스트 프롬프트 -> polygon shapes 반환.

    서버 segment_anything_3.predict()가 text_prompt 를 받으면
    GDINO 없이 SAM3 자체 set_text_prompt()로 처리.
    """
    b64, scale = encode_image(image_bgr, sam_max_size)
    payload = {
        "model": SAM3_MODEL_ID,
        "image": b64,
        "params": {
            "text_prompt": text_prompt,
            "conf_threshold": conf_threshold,
            "show_masks": True,
            "show_boxes": False,
        },
    }
    try:
        r = requests.post(f"{SAM3_SERVER}/v1/predict", json=payload, timeout=180)
        r.raise_for_status()
        resp = r.json()
        if not resp.get("success"):
            print(f"  SAM3 오류: {resp.get('error', {}).get('message', 'unknown')}")
            return []
        shapes = resp.get("data", {}).get("shapes", [])
        shapes = [s if isinstance(s, dict) else s.dict() for s in shapes]
        if scale < 1.0:
            inv = 1.0 / scale
            for s in shapes:
                if s.get("shape_type") == "polygon":
                    s["points"] = [[x * inv, y * inv] for x, y in s["points"]]
        return shapes
    except Exception as e:
        print(f"  SAM3 오류: {e}")
        return []


def shapes_to_pairs(shapes: list) -> list:
    """polygon shapes -> (det, shape) pairs. det = (label, score, x1,y1,x2,y2)."""
    pairs = []
    for s in shapes:
        if s.get("shape_type") != "polygon":
            continue
        pts = s.get("points", [])
        if len(pts) < 3:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        det = (s.get("label", ""), float(s.get("score", 0.0)),
               min(xs), min(ys), max(xs), max(ys))
        pairs.append((det, s))
    return pairs


# ── NMS ───────────────────────────────────────────────────────────────────────
def nms(pairs: list, iou_thresh: float = 0.5) -> list:
    """(det, shape) 쌍에 NMS 적용."""
    if not pairs:
        return []
    boxes  = np.array([[x1, y1, x2, y2] for (_, _, x1, y1, x2, y2), _ in pairs])
    scores = np.array([conf for (_, conf, *_), _ in pairs])
    order  = scores.argsort()[::-1]
    keep   = []
    while len(order):
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        a_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        a_j = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
        iou = inter / (a_i + a_j - inter + 1e-6)
        order = order[1:][iou < iou_thresh]
    return [pairs[i] for i in keep]


# ── 폴리곤 -> YOLO seg 포맷 ──────────────────────────────────────────────────
def save_yolo_label(label_path: Path, pairs: list, W: int, H: int):
    lines = []
    for (label, conf, *_), shape in pairs:
        if shape.get("shape_type") != "polygon":
            continue
        pts = shape.get("points", [])
        if len(pts) < 3:
            continue
        cls_id = CLASS_MAP.get(label, 0)
        coords = " ".join(f"{x/W:.6f} {y/H:.6f}" for x, y in pts)
        lines.append(f"{cls_id} {coords}")
    label_path.write_text("\n".join(lines), encoding="utf-8")


# ── 시각화 저장 ───────────────────────────────────────────────────────────────
def save_vis(vis_path: Path, image_bgr: np.ndarray, pairs: list,
             rail_shapes: list = None):
    vis = image_bgr.copy()
    colors = [(0, 200, 255), (255, 180, 0), (0, 255, 100)]

    for i, ((label, conf, x1, y1, x2, y2), shape) in enumerate(pairs):
        color = colors[i % len(colors)]
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(vis, f"{label} {conf:.2f}", (int(x1), max(0, int(y1) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        if shape.get("shape_type") == "polygon":
            pts = np.array(shape["points"], dtype=np.int32)
            overlay = vis.copy()
            cv2.fillPoly(overlay, [pts], (0, 255, 0))
            cv2.addWeighted(overlay, 0.3, vis, 0.7, 0, vis)
            cv2.polylines(vis, [pts], True, (0, 255, 0), 1)

    # 철로 polygon 빨간색 표시 (디버그용)
    for s in (rail_shapes or []):
        if s.get("shape_type") == "polygon":
            pts = np.array(s["points"], dtype=np.int32)
            cv2.polylines(vis, [pts], True, (0, 0, 255), 2)
            if len(pts):
                cv2.putText(vis, "rail", (int(pts[0][0]), max(0, int(pts[0][1]) - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    max_preview = 2048
    h, w = vis.shape[:2]
    if max(h, w) > max_preview:
        scale = max_preview / max(h, w)
        vis = cv2.resize(vis, (int(w * scale), int(h * scale)))

    cv2.imencode(vis_path.suffix, vis)[1].tofile(str(vis_path))


# ── bbox 크기 필터 ────────────────────────────────────────────────────────────
def _polygon_bbox(shape: dict) -> tuple:
    pts = shape.get("points", [])
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def filter_by_size(pairs: list, min_w: int, max_w: int, min_h: int = 0,
                   img_w: int = 1, img_h: int = 1, debug: bool = False) -> list:
    """SAM 폴리곤 bbox 크기 조건 검사.

    Stage 1 (폴리곤 파생 bbox): 높이 min_h 이상.
    Stage 2 (SAM 마스크 bbox): 높이 min_h*0.7 이상, 너비<=높이*3.
    """
    kept = []
    for (label, conf, x1, y1, x2, y2), shape in pairs:
        det_h = y2 - y1

        s1_h = det_h >= min_h
        if not s1_h:
            print(f"    DROP [{label} {conf:.2f}] Stage1: h={det_h:.0f}(최소:{min_h})")
            continue

        pb = _polygon_bbox(shape)
        if pb:
            mx1, my1, mx2, my2 = pb
            mw = mx2 - mx1
            mh = my2 - my1
            s2_h   = mh >= min_h * 0.7
            s2_max = mw <= mh * 3
            if not (s2_h and s2_max):
                reason = []
                if not s2_h:   reason.append(f"sam_h={mh:.0f}(최소:{min_h*0.7:.0f})")
                if not s2_max: reason.append(f"sam 너무 넓음 w{mw:.0f}>h{mh:.0f}x3")
                print(f"    DROP [{label} {conf:.2f}] Stage2: {', '.join(reason)}")
                continue
            if debug:
                print(f"    KEEP [{label} {conf:.2f}] h={det_h:.0f} sam={mw:.0f}x{mh:.0f}")
        elif debug:
            print(f"    KEEP [{label} {conf:.2f}] h={det_h:.0f} (SAM 마스크 없음)")

        kept.append(((label, conf, x1, y1, x2, y2), shape))
    return kept


# ── 철로 존 필터 ──────────────────────────────────────────────────────────────
def build_zone_mask(shapes: list, H: int, W: int, margin: int) -> np.ndarray:
    """철로 polygon -> binary mask + margin px 원형 팽창."""
    mask = np.zeros((H, W), dtype=np.uint8)
    for s in shapes:
        pts = np.array(s["points"], dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
    if margin > 0 and mask.any():
        r = margin
        yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
        kernel = (xx * xx + yy * yy <= r * r).astype(np.uint8)
        mask = cv2.dilate(mask, kernel)
    return mask


def pole_in_zone(pole_det: tuple, zone_mask: np.ndarray) -> bool:
    """전철주 bbox 중심이 zone_mask 내부이면 True."""
    _, _, x1, y1, x2, y2 = pole_det
    H, W = zone_mask.shape
    cx = int(max(0, min((x1 + x2) / 2, W - 1)))
    cy = int(max(0, min((y1 + y2) / 2, H - 1)))
    return bool(zone_mask[cy, cx])


# ── 이미지 처리 ───────────────────────────────────────────────────────────────
def process_image(image_path: Path,
                  out_label_dir: Path, out_vis_dir: Path,
                  prompt: str, sam3_conf: float,
                  tile_size: int, tile_overlap: float,
                  min_pole_w: int = 15, max_pole_w: int = 130,
                  min_pole_h: int = 220,
                  rail_margin: int = 300) -> int:

    buf = np.fromfile(str(image_path), dtype=np.uint8)
    image_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if image_bgr is None:
        print(f"  이미지 로드 실패: {image_path}")
        return 0
    H, W = image_bgr.shape[:2]

    use_tile = tile_size > 0 and (W > tile_size or H > tile_size)
    if use_tile:
        print(f"  크기 {W}x{H} -> {tile_size}px 타일 처리 중...")
        pairs = _detect_tiled_sam3(image_bgr, prompt, sam3_conf, tile_size, tile_overlap)
    else:
        print(f"  SAM3.1 텍스트 검출: '{prompt}'")
        shapes = sam3_text_segment(image_bgr, prompt, sam3_conf)
        pairs = nms(shapes_to_pairs(shapes))

    if not pairs:
        print("  검출 없음")
        return 0

    # 크기 필터
    before = len(pairs)
    pairs = filter_by_size(pairs, min_pole_w, max_pole_w, min_pole_h, W, H, debug=True)
    print(f"  크기 필터(h>={min_pole_h}): {before}->{len(pairs)}개")
    if not pairs:
        return 0

    # 철로 존 필터 (rail_margin < 0 이면 비활성)
    _, _, rail_prompt = _get_prompts()
    rail_shapes = []
    if rail_margin >= 0:
        print(f"  철로 검출 중 (SAM3 텍스트: '{rail_prompt}')...")
        rail_shapes = sam3_text_segment(image_bgr, rail_prompt, sam3_conf)
        rail_shapes = [s for s in rail_shapes if s.get("shape_type") == "polygon"]
        if rail_shapes:
            zone_mask = build_zone_mask(rail_shapes, H, W, rail_margin)
            before = len(pairs)
            pairs = [(d, s) for d, s in pairs if pole_in_zone(d, zone_mask)]
            print(f"  철로 존 필터(SAM mask+{rail_margin}px): {before}->{len(pairs)}개")
        else:
            print(f"  ! 철로 미검출 -> 철로 존 필터 미적용")
    if not pairs:
        return 0

    print(f"  최종 검출: {len(pairs)}개")
    label_path = out_label_dir / (image_path.stem + ".txt")
    save_yolo_label(label_path, pairs, W, H)

    vis_path = out_vis_dir / (image_path.stem + "_vis.jpg")
    save_vis(vis_path, image_bgr, pairs, rail_shapes)

    return len(pairs)


def _detect_tiled_sam3(image_bgr: np.ndarray, prompt: str, conf: float,
                        tile_size: int, overlap: float) -> list:
    """타일별 SAM3 텍스트 검출 -> 전체 이미지 좌표로 통합, NMS."""
    H, W = image_bgr.shape[:2]
    step = max(1, int(tile_size * (1 - overlap)))

    tile_coords = []
    for y0 in range(0, H, step):
        for x0 in range(0, W, step):
            tile_coords.append((x0, y0, min(x0 + tile_size, W), min(y0 + tile_size, H)))

    print(f"  총 {len(tile_coords)}개 타일")
    all_pairs = []
    for idx, (tx0, ty0, tx1, ty1) in enumerate(tile_coords, 1):
        tile_bgr = image_bgr[ty0:ty1, tx0:tx1]
        shapes = sam3_text_segment(tile_bgr, prompt, conf)
        for s in shapes:
            if s.get("shape_type") == "polygon":
                s["points"] = [[x + tx0, y + ty0] for x, y in s["points"]]
        all_pairs.extend(shapes_to_pairs(shapes))
        if idx % 5 == 0:
            print(f"    타일 {idx}/{len(tile_coords)}, 누적 {len(all_pairs)}개")

    return nms(all_pairs)


# ── classes.txt 저장 ──────────────────────────────────────────────────────────
def save_classes(out_dir: Path):
    (out_dir / "classes.txt").write_text("\n".join(CLASS_NAMES), encoding="utf-8")


# ── 메인 ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="SAM 3.1 텍스트 프롬프트 자동 라벨링")
    parser.add_argument("--input",        required=True, help="이미지 파일 또는 디렉터리")
    parser.add_argument("--output",       default="output/autolabel")
    parser.add_argument("--mode",         default="slope",
                        choices=["auto", "vertical", "slope"])
    parser.add_argument("--prompt",       default=None,
                        help="직접 프롬프트 지정 시 --mode 무시")
    parser.add_argument("--sam3_conf",    type=float, default=0.25)
    parser.add_argument("--ext",          default=".jpg,.jpeg,.png,.tif")
    parser.add_argument("--tile_size",    type=int, default=1280,
                        help="타일 크기 px (0=타일 없음, 기본:1280)")
    parser.add_argument("--tile_overlap", type=float, default=0.2)
    parser.add_argument("--min_pole_w",   type=int, default=15)
    parser.add_argument("--max_pole_w",   type=int, default=130)
    parser.add_argument("--min_pole_h",   type=int, default=220)
    parser.add_argument("--rail_margin",  type=int, default=300,
                        help="철로 mask에서 전철주까지 허용 거리 px (-1=비활성)")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir    = Path(args.output)
    out_label  = out_dir / "labels"
    out_vis    = out_dir / "vis"
    out_label.mkdir(parents=True, exist_ok=True)
    out_vis.mkdir(parents=True, exist_ok=True)

    prompt = args.prompt if args.prompt else select_prompt(input_path, args.mode)

    exts = [e.strip().lower() for e in args.ext.split(",")]
    if input_path.is_dir():
        images = [p for p in input_path.iterdir() if p.suffix.lower() in exts]
    else:
        images = [input_path]

    print(f"처리 대상: {len(images)}장")
    print(f"모드: {args.mode}  프롬프트: {prompt}")
    print(f"타일 크기: {args.tile_size}px  겹침: {args.tile_overlap*100:.0f}%")
    print(f"출력: {out_dir}\n")
    save_classes(out_dir)

    total_det = 0
    for i, img_path in enumerate(sorted(images), 1):
        print(f"[{i}/{len(images)}] {img_path.name}")
        n = process_image(img_path, out_label, out_vis,
                          prompt, args.sam3_conf,
                          args.tile_size, args.tile_overlap,
                          args.min_pole_w, args.max_pole_w, args.min_pole_h,
                          args.rail_margin)
        total_det += n

    print(f"\n완료: 총 {total_det}개 객체 라벨 생성")
    print(f"라벨: {out_label}")
    print(f"시각화: {out_vis}")
    print(f"\nYOLO 학습 명령:")
    print(f"  yolo train model=yolo11n-seg.pt data=output/autolabel/dataset.yaml epochs=100")


if __name__ == "__main__":
    main()
