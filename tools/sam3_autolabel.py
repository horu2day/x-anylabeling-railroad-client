"""
SAM 3.1 텍스트 프롬프트 자동 라벨링
=====================================================
알고리즘:
  A = 전철주 프롬프트(prompts/pole.txt) 검출  →  높이 > min_pole_h 필터
  B = 철로 프롬프트(prompts/rail.txt) 검출   →  zone mask 생성
  결과 = A 중 B zone 내/근처에 있는 것만 유지

사용법:
    python tools/sam3_autolabel.py --input data/역사이미지/slope/
    python tools/sam3_autolabel.py --input data/역사이미지/ --output output/autolabel/

사전 조건:
    - start_server.bat 실행 (SAM 3.1 서버)
"""
import argparse
import base64
from pathlib import Path

import cv2
import numpy as np
import requests

SAM3_SERVER   = "http://localhost:8000"
SAM3_MODEL_ID = "segment_anything_3"

_PROMPT_DIR = Path(__file__).parent.parent / "prompts"

CLASS_NAMES = ["catenary_pole", "bracket"]
_CLS_COLORS = [(0, 200, 255), (255, 130, 0)]


def _load_prompt(filename: str, default: str) -> str:
    path = _PROMPT_DIR / filename
    if path.exists():
        lines = [l.strip() for l in path.read_text(encoding="utf-8").splitlines()
                 if l.strip() and not l.startswith("#")]
        if lines:
            return ", ".join(lines)
    return default


def _get_prompts() -> tuple:
    """(pole_prompt, rail_prompt) 반환."""
    pole = _load_prompt("pole.txt",
                        "railway catenary pole, overhead line support pole, catenary mast")
    rail = _load_prompt("rail.txt", "railroad, railway")
    return pole, rail


# ── 이미지 인코딩 ──────────────────────────────────────────────────────────────
def encode_image(image_bgr: np.ndarray, max_size: int = 2048) -> tuple:
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


# ── 높이 필터 ─────────────────────────────────────────────────────────────────
def filter_by_size(pairs: list, min_h: int, debug: bool = False) -> list:
    """전철주 높이 필터: det bbox 높이 >= min_h, SAM mask도 충분히 높고 가늘어야 함."""
    kept = []
    for (label, conf, x1, y1, x2, y2), shape in pairs:
        det_h = y2 - y1
        if det_h < min_h:
            if debug:
                print(f"    DROP [{label} {conf:.2f}] h={det_h:.0f} (최소:{min_h})")
            continue
        pts = shape.get("points", [])
        if pts:
            ys = [p[1] for p in pts]
            xs = [p[0] for p in pts]
            mh = max(ys) - min(ys)
            mw = max(xs) - min(xs)
            if mh < min_h * 0.7:
                if debug:
                    print(f"    DROP [{label} {conf:.2f}] sam_h={mh:.0f} (최소:{min_h*0.7:.0f})")
                continue
            if mw > mh * 3:
                if debug:
                    print(f"    DROP [{label} {conf:.2f}] 너무 넓음 w{mw:.0f}>h{mh:.0f}x3")
                continue
        kept.append(((label, conf, x1, y1, x2, y2), shape))
    return kept


# ── 철로 존 마스크 ─────────────────────────────────────────────────────────────
def build_zone_mask(shapes: list, H: int, W: int, margin: int,
                    gap_close: int = 500) -> np.ndarray:
    """철로 polygon → binary mask → 수평 갭 보정(beam 차폐) → margin px 팽창."""
    mask = np.zeros((H, W), dtype=np.uint8)
    for s in shapes:
        pts = np.array(s["points"], dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
    if gap_close > 0 and mask.any():
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (gap_close, 30))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, h_kernel)
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


# ── 저장 ──────────────────────────────────────────────────────────────────────
def save_yolo_label(label_path: Path, pairs: list, W: int, H: int):
    lines = []
    for (label, conf, *_), shape in pairs:
        if shape.get("shape_type") != "polygon":
            continue
        pts = shape.get("points", [])
        if len(pts) < 3:
            continue
        cls_id = shape.get("_cls_id", 0)
        coords = " ".join(f"{x/W:.6f} {y/H:.6f}" for x, y in pts)
        lines.append(f"{cls_id} {coords}")
    label_path.write_text("\n".join(lines), encoding="utf-8")


def save_vis(vis_path: Path, image_bgr: np.ndarray, pairs: list,
             rail_shapes: list = None):
    vis = image_bgr.copy()
    for (label, conf, x1, y1, x2, y2), shape in pairs:
        cls_id = shape.get("_cls_id", 0)
        color = _CLS_COLORS[cls_id % len(_CLS_COLORS)]
        if shape.get("shape_type") == "polygon":
            pts = np.array(shape["points"], dtype=np.int32)
            overlay = vis.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.3, vis, 0.7, 0, vis)
            cv2.polylines(vis, [pts], True, color, 2)
            cv2.putText(vis, f"{CLASS_NAMES[cls_id]} {conf:.2f}",
                        (int(x1), max(0, int(y1) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    for s in (rail_shapes or []):
        if s.get("shape_type") == "polygon":
            pts = np.array(s["points"], dtype=np.int32)
            overlay = vis.copy()
            cv2.fillPoly(overlay, [pts], (0, 0, 255))
            cv2.addWeighted(overlay, 0.15, vis, 0.85, 0, vis)
            cv2.polylines(vis, [pts], True, (0, 0, 255), 2)
    h, w = vis.shape[:2]
    if max(h, w) > 2048:
        scale = 2048 / max(h, w)
        vis = cv2.resize(vis, (int(w * scale), int(h * scale)))
    cv2.imencode(vis_path.suffix, vis)[1].tofile(str(vis_path))


# ── 타일 검출 ─────────────────────────────────────────────────────────────────
def _detect_tiled(image_bgr: np.ndarray, prompt: str, conf: float,
                  tile_size: int, overlap: float) -> list:
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


# ── 이미지 처리 ───────────────────────────────────────────────────────────────
def process_image(image_path: Path,
                  out_label_dir: Path, out_vis_dir: Path,
                  sam3_conf: float,
                  tile_size: int, tile_overlap: float,
                  min_pole_h: int = 180,
                  rail_margin: int = 300,
                  rail_gap_close: int = 500) -> int:

    buf = np.fromfile(str(image_path), dtype=np.uint8)
    image_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if image_bgr is None:
        print(f"  이미지 로드 실패: {image_path}")
        return 0
    H, W = image_bgr.shape[:2]

    pole_prompt, rail_prompt = _get_prompts()
    use_tile = tile_size > 0 and (W > tile_size or H > tile_size)

    # A: 전철주 검출
    if use_tile:
        print(f"  [A] 타일 {tile_size}px: '{pole_prompt}'")
        pole_pairs = _detect_tiled(image_bgr, pole_prompt, sam3_conf, tile_size, tile_overlap)
    else:
        print(f"  [A] 전철주 검출: '{pole_prompt}'")
        pole_pairs = nms(shapes_to_pairs(sam3_text_segment(image_bgr, pole_prompt, sam3_conf)))

    # A 높이 필터 (> min_pole_h)
    before = len(pole_pairs)
    pole_pairs = filter_by_size(pole_pairs, min_h=min_pole_h, debug=True)
    print(f"  [A] 높이 필터(h>={min_pole_h}): {before}→{len(pole_pairs)}개")

    for _, shape in pole_pairs:
        shape["_cls_id"] = 0

    # B: 철로 검출
    print(f"  [B] 철로 검출: '{rail_prompt}'")
    rail_shapes = [s for s in sam3_text_segment(image_bgr, rail_prompt, sam3_conf)
                   if s.get("shape_type") == "polygon"]

    # A ∩ B: 철로 zone 필터
    if rail_margin >= 0:
        if rail_shapes:
            zone_mask = build_zone_mask(rail_shapes, H, W, rail_margin, rail_gap_close)
            before = len(pole_pairs)
            pole_pairs = [(d, s) for d, s in pole_pairs if pole_in_zone(d, zone_mask)]
            print(f"  [A∩B] 철로 zone 필터(+{rail_margin}px): {before}→{len(pole_pairs)}개")
        else:
            print(f"  ! [B] 철로 미검출 → 전체 제외")
            pole_pairs = []

    if not pole_pairs:
        return 0

    print(f"  최종: {len(pole_pairs)}개")
    save_yolo_label(out_label_dir / (image_path.stem + ".txt"), pole_pairs, W, H)
    save_vis(out_vis_dir / (image_path.stem + "_vis.jpg"), image_bgr, pole_pairs, rail_shapes)
    return len(pole_pairs)


# ── classes.txt 저장 ──────────────────────────────────────────────────────────
def save_classes(out_dir: Path):
    (out_dir / "classes.txt").write_text("\n".join(CLASS_NAMES), encoding="utf-8")


# ── 메인 ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="SAM 3.1 텍스트 프롬프트 자동 라벨링")
    parser.add_argument("--input",          required=True, help="이미지 파일 또는 디렉터리")
    parser.add_argument("--output",         default="output/autolabel")
    parser.add_argument("--sam3_conf",      type=float, default=0.20)
    parser.add_argument("--ext",            default=".jpg,.jpeg,.png,.tif")
    parser.add_argument("--tile_size",      type=int,   default=1280,
                        help="타일 크기 px (0=타일 없음, 기본:1280)")
    parser.add_argument("--tile_overlap",   type=float, default=0.2)
    parser.add_argument("--min_pole_h",     type=int,   default=180,
                        help="전철주 최소 높이 px (기본:180)")
    parser.add_argument("--rail_margin",    type=int,   default=300,
                        help="철로 mask 팽창 거리 px (-1=비활성, 기본:300)")
    parser.add_argument("--rail_gap_close", type=int,   default=500,
                        help="beam 차폐 갭 보정 수평 closing 폭 px (0=비활성, 기본:500)")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir    = Path(args.output)
    out_label  = out_dir / "labels"
    out_vis    = out_dir / "vis"
    out_label.mkdir(parents=True, exist_ok=True)
    out_vis.mkdir(parents=True, exist_ok=True)

    exts = [e.strip().lower() for e in args.ext.split(",")]
    if input_path.is_dir():
        images = sorted([p for p in input_path.iterdir() if p.suffix.lower() in exts],
                        key=lambda p: p.name)
    else:
        images = [input_path]

    pole_prompt, rail_prompt = _get_prompts()
    print(f"처리 대상: {len(images)}장")
    print(f"전철주 프롬프트: {pole_prompt}")
    print(f"철로 프롬프트:   {rail_prompt}")
    print(f"타일: {args.tile_size}px  겹침: {args.tile_overlap*100:.0f}%")
    print(f"출력: {out_dir}\n")
    save_classes(out_dir)

    total = 0
    for i, img_path in enumerate(images, 1):
        print(f"[{i}/{len(images)}] {img_path.name}")
        total += process_image(img_path, out_label, out_vis,
                               args.sam3_conf,
                               args.tile_size, args.tile_overlap,
                               args.min_pole_h, args.rail_margin, args.rail_gap_close)

    print(f"\n완료: 총 {total}개 라벨 생성")
    print(f"라벨: {out_label}")
    print(f"시각화: {out_vis}")


if __name__ == "__main__":
    main()
