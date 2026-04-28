"""
1단계: Grounding-DINO + SAM 3.1 zero-shot 자동 라벨링
=====================================================
촬영 방향에 따라 프롬프트를 자동 선택:
  vertical (수직/위에서): 팔 구조물(arm/cross-arm)이 위에서 보임
  slope   (사선/측면):   전철주 기둥 전체가 측면에서 보임

사용법:
    # 수직 촬영 디렉터리 (auto 감지)
    python tools/gdino_sam3_autolabel.py --input data/역사이미지/vertical/ --output output/autolabel/vertical/

    # 사선 촬영 디렉터리 (auto 감지)
    python tools/gdino_sam3_autolabel.py --input data/역사이미지/slope/ --output output/autolabel/slope/

    # 명시적 모드 지정
    python tools/gdino_sam3_autolabel.py --input data/역사이미지/vertical/ --mode vertical

    # 타일 크기 조정 (0 = 타일 없음)
    python tools/gdino_sam3_autolabel.py --input data/tiles/ --tile_size 0

사전 조건:
    - start_server.bat 실행 (SAM 3.1 서버)
"""
import argparse
import base64
from pathlib import Path

import cv2
import numpy as np
import requests
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

# ── 설정 ──────────────────────────────────────────────────────────────────────
SAM3_SERVER   = "http://localhost:8000"
SAM3_MODEL_ID = "segment_anything_3"

# ── 촬영 방향별 프롬프트 ───────────────────────────────────────────────────────
# vertical: 수직 하향 촬영
#   - 전철주 상부의 가로팔(cross-arm)과 빔이 위에서 납작하게 보임
#   - 기둥 꼭대기(pole top)가 원형/사각 점으로 보임
#   - 가선(catenary wire)과 빔이 선 형태로 연결되어 보임
PROMPT_VERTICAL = (
    "catenary pole top . pole cross arm . horizontal cantilever beam . bracket arm"
)

# slope: 사선 촬영
#   - 전철주: 레일 옆에 선 얇은 수직 기둥 (두께 8~17px, 길이 250px+)
#   - 기둥에서 수평으로 뻗은 가로팔(cantilever arm)
#   - 트러스/사다리형 빔 연결 구조물 포함
#   - 전봇대(utility pole), 가로등(street light)은 제외
PROMPT_SLOPE = (
    "railway catenary pole . overhead line support pole . catenary mast . "
    "railway pole with cantilever arm . truss beam pole"
)

# 철로 검출용 프롬프트 (railway zone filter에 사용)
PROMPT_RAIL = "railway track . train rail . railroad track . rail line"

# 모든 검출 결과 → class 0 (catenary_pole) 단일 클래스
CLASS_MAP: dict = {}   # 기본값 0 으로 처리 (아래 save_yolo_label 참조)
CLASS_NAMES = ["catenary_pole"]


def select_prompt(input_path: "Path", mode: str) -> str:
    """촬영 모드에 맞는 프롬프트 반환. mode='auto'이면 경로명으로 판단."""
    if mode == "vertical":
        return PROMPT_VERTICAL
    if mode == "slope":
        return PROMPT_SLOPE
    # auto: 경로 어딘가에 'vertical' 또는 'slope' 포함 여부로 결정
    parts = [p.lower() for p in input_path.parts]
    if any("vertical" in p for p in parts):
        print("  [auto] 수직 촬영 감지 → vertical 프롬프트 사용")
        return PROMPT_VERTICAL
    if any("slope" in p for p in parts):
        print("  [auto] 사선 촬영 감지 → slope 프롬프트 사용")
        return PROMPT_SLOPE
    print("  [auto] 촬영 방향 불명 → slope 프롬프트 기본 사용")
    return PROMPT_SLOPE


# ── Grounding-DINO 로드 ────────────────────────────────────────────────────────
def load_gdino(device: str = "cuda"):
    model_id = "IDEA-Research/grounding-dino-base"
    print(f"Grounding-DINO 로드 중: {model_id}")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    print("Grounding-DINO 로드 완료")
    return processor, model


def detect_boxes(processor, model, image_pil: Image.Image,
                 prompt: str, box_thresh: float, device: str) -> list:
    """Grounding-DINO bbox 검출. [(label, conf, x1,y1,x2,y2), ...] 반환."""
    inputs = processor(images=image_pil, text=prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    W, H = image_pil.size
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=box_thresh,
        text_threshold=box_thresh * 0.8,
        target_sizes=[(H, W)],
    )[0]

    # transformers v5: text_labels returns strings, labels returns int ids
    label_key = "text_labels" if "text_labels" in results else "labels"
    dets = []
    for score, label, box in zip(results["scores"], results[label_key], results["boxes"]):
        x1, y1, x2, y2 = box.cpu().numpy()
        dets.append((str(label).lower(), float(score), float(x1), float(y1), float(x2), float(y2)))
    return dets


# ── SAM 3.1 마스크 생성 ───────────────────────────────────────────────────────
def encode_image(image_bgr: np.ndarray, max_size: int = 2048) -> tuple:
    """이미지를 base64 JPEG로 인코딩. max_size 초과 시 축소. (b64, scale) 반환."""
    h, w = image_bgr.shape[:2]
    scale = 1.0
    if max_size > 0 and max(h, w) > max_size:
        scale = max_size / max(h, w)
        image_bgr = cv2.resize(image_bgr, (int(w * scale), int(h * scale)))
    _, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf).decode("utf-8"), scale


def sam3_segment(image_bgr: np.ndarray, boxes: list, conf_threshold: float = 0.25,
                 sam_max_size: int = 2048) -> list:
    """SAM 3.1 Multiplex: 전체 이미지 + 전체 bbox 한 번에 전송.

    이미지를 sam_max_size로 축소해 전송하고, 반환된 폴리곤 좌표는
    원본 이미지 크기로 역스케일링.
    """
    b64, scale = encode_image(image_bgr, sam_max_size)

    # bbox 좌표도 동일 scale 적용
    scaled_boxes = [(lbl, conf, x1*scale, y1*scale, x2*scale, y2*scale)
                    for (lbl, conf, x1, y1, x2, y2) in boxes]

    payload = {
        "model": SAM3_MODEL_ID,
        "image": b64,
        "params": {
            "marks": [{"type": "rectangle", "label": 1,
                       "data": [float(x1), float(y1), float(x2), float(y2)]}
                      for (_, _, x1, y1, x2, y2) in scaled_boxes],
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

        # 폴리곤 좌표를 원본 이미지 크기로 역스케일링
        if scale < 1.0:
            inv = 1.0 / scale
            for s in shapes:
                if s.get("shape_type") == "polygon":
                    s["points"] = [[x * inv, y * inv] for x, y in s["points"]]
        return shapes
    except Exception as e:
        print(f"  SAM3 오류: {e}")
        return []


# ── NMS ───────────────────────────────────────────────────────────────────────
def nms(pairs: list, iou_thresh: float = 0.5) -> list:
    """(det, shape) 쌍에 NMS 적용. det = (label, conf, x1, y1, x2, y2)."""
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


# ── 폴리곤 → YOLO seg 포맷 ───────────────────────────────────────────────────
def save_yolo_label(label_path: Path, pairs: list, W: int, H: int):
    """(det, shape) 쌍 → YOLO segmentation .txt 저장."""
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
def save_vis(vis_path: Path, image_bgr: np.ndarray, pairs: list):
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

    # 큰 이미지는 미리보기 크기로 저장
    max_preview = 2048
    h, w = vis.shape[:2]
    if max(h, w) > max_preview:
        scale = max_preview / max(h, w)
        vis = cv2.resize(vis, (int(w * scale), int(h * scale)))

    cv2.imencode(vis_path.suffix, vis)[1].tofile(str(vis_path))


# ── bbox 크기 필터 ────────────────────────────────────────────────────────────
def _polygon_bbox(shape: dict) -> tuple:
    """SAM 폴리곤의 실제 bbox (x1,y1,x2,y2) 반환. 없으면 None."""
    pts = shape.get("points", [])
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def filter_by_size(pairs: list, min_w: int, max_w: int, min_h: int = 0,
                   img_w: int = 1, img_h: int = 1, debug: bool = False) -> list:
    """GDINO bbox + SAM 마스크 bbox 크기 조건 검사.

    Stage 1 (GDINO bbox):
      - GDINO는 전철주 기둥 + cantilever arm 전체를 감싸는 bbox를 반환하므로
        너비 상한은 적용하지 않고 높이(min_h)와 종횡비(h>=w)만 확인.
      - 너비 하한(min_w)은 너무 작은 점/노이즈 제거용으로만 사용.
    Stage 2 (SAM 마스크 bbox, 픽셀 좌표):
      - SAM이 반환한 폴리곤 좌표는 이미 픽셀 단위 → 그대로 사용.
      - 마스크 높이 >= min_h * 0.7 (도로 등 가로로 긴 위양성 제거).
      - 마스크 종횡비: height >= width (기둥은 세로가 더 길어야 함)."""
    kept = []
    for (label, conf, x1, y1, x2, y2), shape in pairs:
        det_w = x2 - x1
        det_h = y2 - y1

        # Stage 1: GDINO bbox — 높이만 검사
        # (GDINO는 cantilever arm 포함 전체 구조물 bbox를 반환하므로 가로/종횡비 필터 미적용)
        s1_h = det_h >= min_h
        if not s1_h:
            print(f"    DROP [{label} {conf:.2f}] Stage1: gdino_h={det_h:.0f}(최소:{min_h})")
            continue

        # Stage 2: SAM 마스크 bbox — 높이 + 최대 면적 검사
        pb = _polygon_bbox(shape)
        if pb:
            mx1, my1, mx2, my2 = pb
            mw = mx2 - mx1   # 이미 픽셀 좌표
            mh = my2 - my1
            # 마스크 높이는 최소 min_h*0.7 이상
            # 마스크 너비는 높이의 3배 이하 (가로로 과도하게 퍼진 도로/밭 제거)
            s2_h   = mh >= min_h * 0.7
            s2_max = mw <= mh * 3
            if not (s2_h and s2_max):
                reason = []
                if not s2_h:   reason.append(f"sam_h={mh:.0f}(최소:{min_h*0.7:.0f})")
                if not s2_max: reason.append(f"sam 너무 넓음 w{mw:.0f}>h{mh:.0f}×3")
                print(f"    DROP [{label} {conf:.2f}] Stage2: {', '.join(reason)}")
                continue
            if debug:
                print(f"    KEEP [{label} {conf:.2f}] gdino={det_w:.0f}×{det_h:.0f} sam={mw:.0f}×{mh:.0f}")
        elif debug:
            print(f"    KEEP [{label} {conf:.2f}] gdino={det_w:.0f}×{det_h:.0f} (SAM 마스크 없음)")

        kept.append(((label, conf, x1, y1, x2, y2), shape))

    return kept


# ── 철로 존 검출 & 필터 ──────────────────────────────────────────────────────
def detect_railway_zone(processor, model, image_bgr: np.ndarray,
                        device: str, rail_thresh: float = 0.15) -> list:
    """GDINO로 철로 bbox 검출. [(label, conf, x1,y1,x2,y2), ...] 원본 좌표 반환."""
    H, W = image_bgr.shape[:2]
    max_size = 1280
    scale = 1.0
    img = image_bgr
    if max(H, W) > max_size:
        scale = max_size / max(H, W)
        img = cv2.resize(image_bgr, (int(W * scale), int(H * scale)))

    image_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    dets = detect_boxes(processor, model, image_pil, PROMPT_RAIL, rail_thresh, device)
    if not dets:
        return []

    if scale < 1.0:
        inv = 1.0 / scale
        dets = [(lbl, conf, x1*inv, y1*inv, x2*inv, y2*inv)
                for lbl, conf, x1, y1, x2, y2 in dets]
    return dets


def in_railway_zone(pole_det: tuple, rail_dets: list, margin: int) -> bool:
    """전철주 bbox가 철로 bbox 중 하나라도 margin 내에서 겹치면 True."""
    _, _, px1, py1, px2, py2 = pole_det
    for _, _, rx1, ry1, rx2, ry2 in rail_dets:
        if (px2 >= rx1 - margin and px1 <= rx2 + margin and
                py2 >= ry1 - margin and py1 <= ry2 + margin):
            return True
    return False


# ── 이미지 처리 (타일 포함) ───────────────────────────────────────────────────
def process_image(processor, model, image_path: Path,
                  out_label_dir: Path, out_vis_dir: Path,
                  prompt: str, box_thresh: float, sam3_conf: float,
                  device: str, tile_size: int, tile_overlap: float,
                  min_pole_w: int = 15, max_pole_w: int = 130,
                  min_pole_h: int = 220,
                  rail_margin: int = 300, rail_thresh: float = 0.15) -> int:

    buf = np.fromfile(str(image_path), dtype=np.uint8)
    image_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if image_bgr is None:
        print(f"  이미지 로드 실패: {image_path}")
        return 0
    H, W = image_bgr.shape[:2]

    use_tile = tile_size > 0 and (W > tile_size or H > tile_size)
    if use_tile:
        print(f"  크기 {W}×{H} → {tile_size}px 타일 처리 중...")
        pairs = _detect_tiled(processor, model, image_bgr, prompt, box_thresh,
                              sam3_conf, device, tile_size, tile_overlap)
    else:
        image_pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        dets = detect_boxes(processor, model, image_pil, prompt, box_thresh, device)
        if not dets:
            return 0
        shapes = sam3_segment(image_bgr, dets, sam3_conf)
        pairs = list(zip(dets, shapes)) if shapes else [(d, {}) for d in dets]
        pairs = nms(pairs)

    if not pairs:
        return 0

    # 크기 필터
    before = len(pairs)
    pairs = filter_by_size(pairs, min_pole_w, max_pole_w, min_pole_h, W, H, debug=True)
    print(f"  크기 필터(h>={min_pole_h}): {before}→{len(pairs)}개")
    if not pairs:
        return 0

    # 철로 존 필터 (rail_margin < 0 이면 비활성)
    if rail_margin >= 0:
        print(f"  철로 검출 중 (thresh={rail_thresh})...")
        rail_dets = detect_railway_zone(processor, model, image_bgr, device, rail_thresh)
        if rail_dets:
            print(f"  철로 bbox {len(rail_dets)}개 → margin {rail_margin}px 내 전철주만 유지")
            before = len(pairs)
            pairs = [(d, s) for d, s in pairs if in_railway_zone(d, rail_dets, rail_margin)]
            print(f"  철로 존 필터: {before}→{len(pairs)}개")
        else:
            print(f"  ⚠ 철로 미검출 → 철로 존 필터 미적용")
    if not pairs:
        return 0

    print(f"  최종 검출: {len(pairs)}개")
    label_path = out_label_dir / (image_path.stem + ".txt")
    save_yolo_label(label_path, pairs, W, H)

    vis_path = out_vis_dir / (image_path.stem + "_vis.jpg")
    save_vis(vis_path, image_bgr, pairs)

    return len(pairs)


def _detect_tiled(processor, model, image_bgr: np.ndarray,
                  prompt: str, box_thresh: float, sam3_conf: float,
                  device: str, tile_size: int, overlap: float) -> list:
    """타일별 GDINO+SAM → 전체 이미지 좌표로 통합, NMS 적용."""
    H, W = image_bgr.shape[:2]
    step = max(1, int(tile_size * (1 - overlap)))

    all_pairs = []
    tile_coords = []
    for y0 in range(0, H, step):
        for x0 in range(0, W, step):
            x1 = min(x0 + tile_size, W)
            y1 = min(y0 + tile_size, H)
            tile_coords.append((x0, y0, x1, y1))

    print(f"  총 {len(tile_coords)}개 타일")

    # ── Pass 1: GDINO로 모든 타일 검출 ────────────────────────────────────────
    all_dets_full = []   # (label, conf, x1,y1,x2,y2) 전체 이미지 좌표
    for idx, (tx0, ty0, tx1, ty1) in enumerate(tile_coords, 1):
        tile_bgr = image_bgr[ty0:ty1, tx0:tx1]
        tile_pil = Image.fromarray(cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB))
        dets = detect_boxes(processor, model, tile_pil, prompt, box_thresh, device)
        for label, conf, bx1, by1, bx2, by2 in dets:
            all_dets_full.append((label, conf, bx1+tx0, by1+ty0, bx2+tx0, by2+ty0))
        if idx % 10 == 0:
            print(f"    GDINO 타일 {idx}/{len(tile_coords)}, 누적 {len(all_dets_full)}개")

    if not all_dets_full:
        return []

    # NMS로 중복 제거
    dummy_pairs = [(d, {}) for d in all_dets_full]
    deduped = nms(dummy_pairs)
    dets_final = [d for d, _ in deduped]
    print(f"  GDINO 완료: {len(all_dets_full)}개 → NMS 후 {len(dets_final)}개")

    # ── Pass 2: SAM Multiplex — 전체 이미지 + 모든 bbox 한 번에 전송 ──────────
    # 전체 8K 이미지를 직접 넣으면 OOM 위험 → 검출된 bbox를 포함하는
    # 최소 crop 영역 단위로 묶어서 전송 (SAM multiplex의 장점 최대화)
    print(f"  SAM 3.1 Multiplex 세그멘테이션 ({len(dets_final)}개 한 번에)...")
    shapes = sam3_segment(image_bgr, dets_final, sam3_conf, sam_max_size=2048)
    print(f"  SAM 반환 shape 수: {len(shapes)} (bbox 수: {len(dets_final)})")
    if not shapes:
        shapes = [{} for _ in dets_final]
    elif len(shapes) != len(dets_final):
        print(f"  ⚠ SAM shape 수({len(shapes)}) ≠ bbox 수({len(dets_final)}) — zip으로 짧은 쪽 기준 처리")

    for det, shape in zip(dets_final, shapes):
        all_pairs.append((det, shape))

    return nms(all_pairs)


# ── classes.txt 저장 ─────────────────────────────────────────────────────────
def save_classes(out_dir: Path):
    (out_dir / "classes.txt").write_text("\n".join(CLASS_NAMES), encoding="utf-8")


# ── 메인 ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Grounding-DINO + SAM 3.1 자동 라벨링")
    parser.add_argument("--input",        required=True, help="이미지 파일 또는 디렉터리")
    parser.add_argument("--output",       default="output/autolabel")
    parser.add_argument("--mode",         default="slope",
                        choices=["auto", "vertical", "slope"],
                        help="촬영 방향 (기본: slope)")
    parser.add_argument("--prompt",       default=None,
                        help="직접 프롬프트 지정 시 --mode 무시")
    parser.add_argument("--box_thresh",   type=float, default=0.25)
    parser.add_argument("--sam3_conf",    type=float, default=0.25)
    parser.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ext",          default=".jpg,.jpeg,.png,.tif", help="처리할 확장자")
    parser.add_argument("--tile_size",    type=int, default=1280,
                        help="타일 크기 px (0=타일 없음, 기본:1280)")
    parser.add_argument("--tile_overlap", type=float, default=0.2,
                        help="타일 겹침 비율 (기본:0.2 = 20%%)")
    parser.add_argument("--min_pole_w",   type=int, default=15,
                        help="기둥 bbox 최소 가로 px (기본:15)")
    parser.add_argument("--max_pole_w",   type=int, default=130,
                        help="기둥 bbox 최대 가로 px (기본:130)")
    parser.add_argument("--min_pole_h",   type=int, default=220,
                        help="기둥 bbox 최소 세로 px (기본:220)")
    parser.add_argument("--rail_margin",  type=int, default=300,
                        help="철로 bbox에서 전철주까지 허용 거리 px (기본:300, -1=비활성)")
    parser.add_argument("--rail_thresh",  type=float, default=0.15,
                        help="철로 검출 GDINO 임계값 (기본:0.15)")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir    = Path(args.output)
    out_label  = out_dir / "labels"
    out_vis    = out_dir / "vis"
    out_label.mkdir(parents=True, exist_ok=True)
    out_vis.mkdir(parents=True, exist_ok=True)

    # 프롬프트 결정: --prompt 명시 > --mode 기반 자동 선택
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

    processor, model = load_gdino(args.device)
    save_classes(out_dir)

    total_det = 0
    for i, img_path in enumerate(sorted(images), 1):
        print(f"[{i}/{len(images)}] {img_path.name}")
        n = process_image(processor, model, img_path,
                          out_label, out_vis,
                          prompt, args.box_thresh, args.sam3_conf,
                          args.device, args.tile_size, args.tile_overlap,
                          args.min_pole_w, args.max_pole_w, args.min_pole_h,
                          args.rail_margin, args.rail_thresh)
        total_det += n

    print(f"\n완료: 총 {total_det}개 객체 라벨 생성")
    print(f"라벨: {out_label}")
    print(f"시각화: {out_vis}")
    print(f"\nYOLO 학습 명령:")
    print(f"  yolo train model=yolo26n-seg.pt data=output/autolabel/dataset.yaml epochs=100")


if __name__ == "__main__":
    main()
