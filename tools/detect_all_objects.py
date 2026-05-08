"""
이미지에서 모든 객체를 SAM3.1로 검출하여 색상별로 시각화.

전략:
  - 8×6 타일로 분할 (10% 중복)
  - 타일당 SAM3.1 호출 1회 (모든 카테고리 프롬프트 합산) → 총 48회
  - 병렬 처리(ThreadPoolExecutor)로 속도 향상
  - 반환된 label 키워드로 카테고리 자동 분류
  - 전체 이미지 기준 좌표 변환 → NMS → 색상 시각화

사용법:
    python tools/detect_all_objects.py --input <이미지경로>
    python tools/detect_all_objects.py --input <이미지경로> --cols 8 --rows 6 --overlap 0.10 --workers 4
"""
import argparse
import base64
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import requests

SAM3_SERVER   = "http://localhost:8000"
SAM3_MODEL_ID = "segment_anything_3"

# 카테고리 정의: (클래스명, [매핑 키워드 리스트], BGR 색상)
# 키워드는 SAM3가 반환하는 label 과 부분 일치 검사에 사용
CATEGORIES = [
    ("railway",       ["railroad", "railway rail", "steel rail"],
                      (  0, 200, 255)),
    ("catenary_pole", ["catenary pole", "overhead line pole", "catenary mast"],
                      (255, 130,   0)),
    ("highway",       ["highway", "expressway", "paved road"],
                      (160, 160, 160)),
    ("vehicle",       ["car", "truck", "vehicle", "automobile"],
                      (  0, 255,   0)),
    ("building",      ["building", "house", "rooftop", "structure"],
                      ( 50,  50, 255)),
    ("farmland",      ["farmland", "field", "cropland", "vegetable garden", "agricultural"],
                      ( 50, 200,  50)),
    ("vegetation",    ["tree", "forest", "shrub", "vegetation", "bush"],
                      (  0, 120,   0)),
    ("guardrail",     ["guardrail", "highway barrier", "road fence", "crash barrier"],
                      (200,   0, 200)),
    ("bridge",        ["bridge", "overpass", "viaduct"],
                      (  0, 165, 255)),
    ("wire",          ["overhead wire", "catenary wire", "electric cable"],
                      (200, 200, 255)),
]

# 모든 카테고리를 하나의 text_prompt로 합산 (SAM3.1에 한 번에 전달)
_ALL_PROMPTS = ", ".join([
    "railroad track, railway rail, steel rail",
    "railway catenary pole, overhead line pole, catenary mast",
    "highway road, expressway asphalt, paved road lane",
    "car, truck, vehicle, automobile",
    "building, house, rooftop",
    "farmland, agricultural field, vegetable garden",
    "trees, forest, shrubs, vegetation",
    "guardrail, highway barrier, road fence, crash barrier",
    "bridge, overpass, viaduct",
    "overhead wire, catenary wire, electric cable line",
])


def _label_to_category(label: str) -> int:
    """SAM3 반환 label → CATEGORIES 인덱스. 매칭 없으면 -1."""
    label_l = label.lower()
    for i, (_, keywords, _) in enumerate(CATEGORIES):
        for kw in keywords:
            if kw in label_l:
                return i
    return -1


# ── SAM3 호출 ─────────────────────────────────────────────────────────────────
def encode_image(image_bgr: np.ndarray, max_size: int = 1280) -> tuple:
    h, w = image_bgr.shape[:2]
    scale = 1.0
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        image_bgr = cv2.resize(image_bgr, (int(w * scale), int(h * scale)))
    _, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf).decode("utf-8"), scale


def sam3_segment_tile(tile_bgr: np.ndarray, conf: float = 0.20) -> list:
    """타일 하나에 모든 카테고리 프롬프트를 한 번에 요청."""
    b64, scale = encode_image(tile_bgr)
    payload = {
        "model": SAM3_MODEL_ID,
        "image": b64,
        "params": {
            "text_prompt": _ALL_PROMPTS,
            "conf_threshold": conf,
            "show_masks": True,
            "show_boxes": False,
        },
    }
    try:
        r = requests.post(f"{SAM3_SERVER}/v1/predict", json=payload, timeout=120)
        r.raise_for_status()
        resp = r.json()
        if not resp.get("success"):
            return []
        shapes = resp.get("data", {}).get("shapes", [])
        shapes = [s if isinstance(s, dict) else s.dict() for s in shapes]
        if scale < 1.0:
            inv = 1.0 / scale
            for s in shapes:
                if s.get("shape_type") == "polygon":
                    s["points"] = [[x * inv, y * inv] for x, y in s["points"]]
        return [s for s in shapes if s.get("shape_type") == "polygon"]
    except Exception as e:
        return []


# ── NMS ───────────────────────────────────────────────────────────────────────
def _bbox(pts: list) -> tuple:
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def nms_shapes(shapes: list, iou_thresh: float = 0.4) -> list:
    if not shapes:
        return []
    bboxes = np.array([_bbox(s["points"]) for s in shapes], dtype=np.float32)
    scores = np.array([float(s.get("score", 0.5)) for s in shapes])
    order  = scores.argsort()[::-1]
    keep   = []
    while len(order):
        i = order[0]; keep.append(i)
        if len(order) == 1:
            break
        xx1 = np.maximum(bboxes[i, 0], bboxes[order[1:], 0])
        yy1 = np.maximum(bboxes[i, 1], bboxes[order[1:], 1])
        xx2 = np.minimum(bboxes[i, 2], bboxes[order[1:], 2])
        yy2 = np.minimum(bboxes[i, 3], bboxes[order[1:], 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        a_i = (bboxes[i, 2] - bboxes[i, 0]) * (bboxes[i, 3] - bboxes[i, 1])
        a_j = ((bboxes[order[1:], 2] - bboxes[order[1:], 0]) *
               (bboxes[order[1:], 3] - bboxes[order[1:], 1]))
        iou = inter / (a_i + a_j - inter + 1e-6)
        order = order[1:][iou < iou_thresh]
    return [shapes[i] for i in keep]


# ── 타일 그리드 검출 (병렬) ───────────────────────────────────────────────────
def detect_all_tiled(image_bgr: np.ndarray, cols: int, rows: int,
                     overlap: float, conf: float, workers: int) -> list:
    """48타일을 병렬로 검출. 반환: [shape, ...] (전체 이미지 좌표계)"""
    H, W = image_bgr.shape[:2]
    base_w = W / cols
    base_h = H / rows
    pad_x  = int(base_w * overlap)
    pad_y  = int(base_h * overlap)

    # 타일 좌표 목록
    tiles = []
    for r in range(rows):
        for c in range(cols):
            x0 = max(0, int(c * base_w) - pad_x)
            x1 = min(W, int((c + 1) * base_w) + pad_x)
            y0 = max(0, int(r * base_h) - pad_y)
            y1 = min(H, int((r + 1) * base_h) + pad_y)
            tiles.append((c, r, x0, y0, x1, y1))

    total = len(tiles)
    done  = [0]
    all_shapes = []

    def process_tile(args):
        c, r, x0, y0, x1, y1 = args
        tile = image_bgr[y0:y1, x0:x1]
        shapes = sam3_segment_tile(tile, conf)
        for s in shapes:
            s["points"] = [[px + x0, py + y0] for px, py in s["points"]]
        return shapes

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(process_tile, t): t for t in tiles}
        for fut in as_completed(futs):
            shapes = fut.result()
            all_shapes.extend(shapes)
            done[0] += 1
            print(f"  타일 {done[0]:02d}/{total} 완료, 누적 {len(all_shapes)}개", end="\r")

    print()
    return all_shapes


# ── 시각화 ────────────────────────────────────────────────────────────────────
def draw_detections(image_bgr: np.ndarray,
                    per_category: list[list]) -> np.ndarray:
    """per_category[i] = list of shapes for CATEGORIES[i]"""
    vis = image_bgr.copy()

    for i, (cls_name, _, color) in enumerate(CATEGORIES):
        for s in per_category[i]:
            pts = np.array(s["points"], dtype=np.int32)
            overlay = vis.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.35, vis, 0.65, 0, vis)
            cv2.polylines(vis, [pts], True, color, 2)

    # 범례 (우측 상단)
    h, w = vis.shape[:2]
    lx = w - 270
    ly = 20
    panel_h = len(CATEGORIES) * 24 + 10
    vis[0:panel_h, lx - 8:w] = (vis[0:panel_h, lx - 8:w] * 0.35).astype(np.uint8)

    for i, (cls_name, _, color) in enumerate(CATEGORIES):
        cnt = len(per_category[i])
        cv2.rectangle(vis, (lx, ly - 13), (lx + 15, ly + 3), color, -1)
        cv2.putText(vis, f"{cls_name}  ({cnt})",
                    (lx + 20, ly), cv2.FONT_HERSHEY_SIMPLEX,
                    0.54, color, 1, cv2.LINE_AA)
        ly += 24

    return vis


# ── 메인 ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",   required=True)
    ap.add_argument("--output",  default=None)
    ap.add_argument("--conf",    type=float, default=0.20)
    ap.add_argument("--cols",    type=int,   default=8,    help="가로 타일 수 (기본:8)")
    ap.add_argument("--rows",    type=int,   default=6,    help="세로 타일 수 (기본:6)")
    ap.add_argument("--overlap", type=float, default=0.10, help="타일 중복 비율 (기본:0.10)")
    ap.add_argument("--workers", type=int,   default=4,    help="병렬 스레드 수 (기본:4)")
    args = ap.parse_args()

    img_path  = Path(args.input)
    buf       = np.fromfile(str(img_path), dtype=np.uint8)
    image_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if image_bgr is None:
        print(f"이미지 로드 실패: {img_path}"); return

    H, W = image_bgr.shape[:2]
    total_tiles = args.cols * args.rows
    print(f"이미지: {W}×{H}  |  타일: {args.cols}×{args.rows}={total_tiles}개  |  중복: {args.overlap*100:.0f}%")
    print(f"SAM3 호출 횟수: {total_tiles}회 (카테고리 {len(CATEGORIES)}개 합산)  |  병렬: {args.workers}스레드\n")

    t0 = time.time()
    all_shapes = detect_all_tiled(image_bgr, args.cols, args.rows,
                                  args.overlap, args.conf, args.workers)
    print(f"전체 검출 {len(all_shapes)}개 → NMS 처리 중...")

    # 카테고리별 분류 + NMS
    buckets: list[list] = [[] for _ in CATEGORIES]
    unmatched = 0
    for s in all_shapes:
        idx = _label_to_category(s.get("label", ""))
        if idx >= 0:
            buckets[idx].append(s)
        else:
            unmatched += 1

    for i in range(len(CATEGORIES)):
        before = len(buckets[i])
        buckets[i] = nms_shapes(buckets[i])
        print(f"  {CATEGORIES[i][0]:16s}: {before:4d} → NMS → {len(buckets[i]):4d}개")

    if unmatched:
        print(f"  (미분류 {unmatched}개 제외)")

    elapsed = time.time() - t0
    print(f"\n완료: {elapsed:.0f}초")

    vis = draw_detections(image_bgr, buckets)

    h, w = vis.shape[:2]
    if max(h, w) > 4096:
        s = 4096 / max(h, w)
        vis = cv2.resize(vis, (int(w * s), int(h * s)))

    out_path = (Path(args.output) if args.output
                else img_path.parent / (img_path.stem + "_tiled_all.jpg"))
    cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 93])[1].tofile(str(out_path))
    print(f"저장: {out_path}")


if __name__ == "__main__":
    main()
