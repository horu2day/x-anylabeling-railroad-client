"""
이미지에서 객체를 SAM3.1로 검출하여 색상별로 시각화.

전략:
  - cols×rows 타일로 분할 (overlap 중복)
  - --tiles 로 처리할 타일 번호 지정 (예: 9-24, 1,5,9, 전체=all)
  - --categories 로 JSON 설정 파일 로드 (카테고리·프롬프트·색상 정의)
  - 타일당 SAM3.1 1회 호출 (모든 카테고리 프롬프트 합산)
  - 병렬 처리(ThreadPoolExecutor) → NMS → 색상 시각화

사용법:
    python tools/detect_all_objects.py \\
        --input data/역사이미지/slope/DJI_20260306113839_0005.JPG \\
        --categories configs/railway_zone.json \\
        --tiles 9-24 \\
        --cols 8 --rows 6 --overlap 0.10 --workers 4
"""
import argparse
import base64
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import requests

SAM3_SERVER   = "http://localhost:8000"
SAM3_MODEL_ID = "segment_anything_3"

# 기본 카테고리 (--categories 미지정 시 사용)
_DEFAULT_CATEGORIES = [
    {"name": "railway",       "prompt": "railroad track, railway rail, steel rail",
     "keywords": ["railroad", "railway rail", "steel rail"], "color_bgr": [0, 200, 255]},
    {"name": "catenary_pole", "prompt": "railway catenary pole, overhead line pole, catenary mast",
     "keywords": ["catenary pole", "overhead line pole", "catenary mast"], "color_bgr": [255, 130, 0]},
    {"name": "highway",       "prompt": "highway road, expressway asphalt, paved road lane",
     "keywords": ["highway", "expressway", "paved road"], "color_bgr": [160, 160, 160]},
    {"name": "vehicle",       "prompt": "car, truck, vehicle, automobile",
     "keywords": ["car", "truck", "vehicle", "automobile"], "color_bgr": [0, 255, 0]},
    {"name": "building",      "prompt": "building, house, rooftop, structure",
     "keywords": ["building", "house", "rooftop", "structure"], "color_bgr": [50, 50, 255]},
    {"name": "farmland",      "prompt": "farmland, agricultural field, cropland, vegetable garden",
     "keywords": ["farmland", "field", "cropland", "vegetable"], "color_bgr": [50, 200, 50]},
    {"name": "vegetation",    "prompt": "trees, forest, shrubs, vegetation, bushes",
     "keywords": ["tree", "forest", "shrub", "vegetation", "bush"], "color_bgr": [0, 120, 0]},
    {"name": "guardrail",     "prompt": "guardrail, highway barrier, road fence, crash barrier",
     "keywords": ["guardrail", "highway barrier", "road fence", "crash barrier"], "color_bgr": [200, 0, 200]},
    {"name": "bridge",        "prompt": "bridge, overpass, viaduct",
     "keywords": ["bridge", "overpass", "viaduct"], "color_bgr": [0, 165, 255]},
    {"name": "wire",          "prompt": "overhead wire, catenary wire, electric cable line",
     "keywords": ["catenary wire", "overhead wire", "electric cable"], "color_bgr": [200, 200, 255]},
]


# ── 타일 번호 파싱 ────────────────────────────────────────────────────────────
def parse_tiles(tile_str: str, total: int) -> set:
    """'9-24', '1,3,5', 'all' → tile index 집합 (1-based)."""
    if tile_str.lower() == "all":
        return set(range(1, total + 1))
    result = set()
    for part in tile_str.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            result.update(range(int(a), int(b) + 1))
        else:
            result.add(int(part))
    return result


# ── 카테고리 로드 ─────────────────────────────────────────────────────────────
def load_categories(json_path: str | None) -> list:
    if json_path:
        data = json.loads(Path(json_path).read_text(encoding="utf-8"))
        return data["categories"]
    return _DEFAULT_CATEGORIES


def label_to_category(label: str, categories: list) -> int:
    label_l = label.lower()
    for i, cat in enumerate(categories):
        for kw in cat["keywords"]:
            if kw in label_l:
                return i
    return -1


def build_combined_prompt(categories: list) -> str:
    return ", ".join(cat["prompt"] for cat in categories)


# ── SAM3 호출 ─────────────────────────────────────────────────────────────────
def encode_image(image_bgr: np.ndarray, max_size: int = 1280) -> tuple:
    h, w = image_bgr.shape[:2]
    scale = 1.0
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        image_bgr = cv2.resize(image_bgr, (int(w * scale), int(h * scale)))
    _, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf).decode("utf-8"), scale


def sam3_segment_tile(tile_bgr: np.ndarray, prompt: str, conf: float) -> list:
    b64, scale = encode_image(tile_bgr)
    payload = {
        "model": SAM3_MODEL_ID,
        "image": b64,
        "params": {"text_prompt": prompt, "conf_threshold": conf,
                   "show_masks": True, "show_boxes": False},
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
    except Exception:
        return []


# ── NMS ───────────────────────────────────────────────────────────────────────
def _bbox(pts):
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def _nms_core(shapes: list, iou_thresh: float) -> list:
    """IoU 기반 NMS. shapes 각 항목에 score 필드 필요."""
    if not shapes:
        return []
    bboxes = np.array([_bbox(s["points"]) for s in shapes], dtype=np.float32)
    scores = np.array([float(s.get("score", 0.5)) for s in shapes])
    order  = scores.argsort()[::-1]
    keep   = []
    while len(order):
        i = order[0]; keep.append(i)
        if len(order) == 1: break
        xx1 = np.maximum(bboxes[i,0], bboxes[order[1:],0])
        yy1 = np.maximum(bboxes[i,1], bboxes[order[1:],1])
        xx2 = np.minimum(bboxes[i,2], bboxes[order[1:],2])
        yy2 = np.minimum(bboxes[i,3], bboxes[order[1:],3])
        inter = np.maximum(0, xx2-xx1) * np.maximum(0, yy2-yy1)
        a_i = (bboxes[i,2]-bboxes[i,0])*(bboxes[i,3]-bboxes[i,1])
        a_j = (bboxes[order[1:],2]-bboxes[order[1:],0])*(bboxes[order[1:],3]-bboxes[order[1:],1])
        iou = inter / (a_i + a_j - inter + 1e-6)
        order = order[1:][iou < iou_thresh]
    return [shapes[i] for i in keep]


def nms_shapes(shapes: list, iou_thresh: float = 0.4) -> list:
    return _nms_core(shapes, iou_thresh)


def cross_class_nms(buckets: list, categories: list, iou_thresh: float) -> list:
    """클래스 간 NMS: 동일 영역에 다른 클래스가 중복 검출될 때 우선순위 높은 쪽 보존.

    정렬 기준: (priority 오름차순, score 내림차순)
    → priority 낮은 값(=중요 클래스)이 우선 보존됨.
    """
    # 모든 shape에 클래스 인덱스 태깅
    tagged = []
    for i, shapes in enumerate(buckets):
        priority = categories[i].get("priority", 99)
        for s in shapes:
            tagged.append((priority, -float(s.get("score", 0.5)), i, s))

    # priority 오름차순, score 내림차순 정렬
    tagged.sort(key=lambda x: (x[0], x[1]))

    if not tagged:
        return [[] for _ in buckets]

    all_shapes = [t[3] for t in tagged]
    cls_ids    = [t[2] for t in tagged]

    bboxes = np.array([_bbox(s["points"]) for s in all_shapes], dtype=np.float32)
    suppressed = [False] * len(all_shapes)

    for i in range(len(all_shapes)):
        if suppressed[i]:
            continue
        for j in range(i + 1, len(all_shapes)):
            if suppressed[j]:
                continue
            xx1 = max(bboxes[i,0], bboxes[j,0])
            yy1 = max(bboxes[i,1], bboxes[j,1])
            xx2 = min(bboxes[i,2], bboxes[j,2])
            yy2 = min(bboxes[i,3], bboxes[j,3])
            inter = max(0, xx2-xx1) * max(0, yy2-yy1)
            if inter == 0:
                continue
            a_i = (bboxes[i,2]-bboxes[i,0])*(bboxes[i,3]-bboxes[i,1])
            a_j = (bboxes[j,2]-bboxes[j,0])*(bboxes[j,3]-bboxes[j,1])
            iou = inter / (a_i + a_j - inter + 1e-6)
            if iou >= iou_thresh:
                suppressed[j] = True  # i가 우선순위 높으므로 j 제거

    new_buckets = [[] for _ in buckets]
    for i, (keep, cls_i, s) in enumerate(zip(suppressed, cls_ids, all_shapes)):
        if not keep:
            new_buckets[cls_i].append(s)
    return new_buckets


# ── 타일 그리드 검출 (병렬) ───────────────────────────────────────────────────
def detect_tiled(image_bgr: np.ndarray, cols: int, rows: int, overlap: float,
                 conf: float, workers: int, tile_filter: set,
                 prompt: str) -> list:
    H, W   = image_bgr.shape[:2]
    base_w = W / cols
    base_h = H / rows
    pad_x  = int(base_w * overlap)
    pad_y  = int(base_h * overlap)

    tiles = []
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c + 1
            if idx not in tile_filter:
                continue
            x0 = max(0, int(c * base_w) - pad_x)
            x1 = min(W, int((c + 1) * base_w) + pad_x)
            y0 = max(0, int(r * base_h) - pad_y)
            y1 = min(H, int((r + 1) * base_h) + pad_y)
            tiles.append((idx, x0, y0, x1, y1))

    total      = len(tiles)
    done       = [0]
    all_shapes = []

    def process(args):
        idx, x0, y0, x1, y1 = args
        tile   = image_bgr[y0:y1, x0:x1]
        shapes = sam3_segment_tile(tile, prompt, conf)
        for s in shapes:
            s["points"] = [[px + x0, py + y0] for px, py in s["points"]]
        return shapes

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(process, t): t for t in tiles}
        for fut in as_completed(futs):
            all_shapes.extend(fut.result())
            done[0] += 1
            print(f"  타일 {done[0]:02d}/{total} 완료, 누적 {len(all_shapes)}개", end="\r")
    print()
    return all_shapes


# ── 시각화 ────────────────────────────────────────────────────────────────────
def draw_detections(image_bgr: np.ndarray, buckets: list,
                    categories: list, tile_filter: set,
                    cols: int, rows: int, overlap: float) -> np.ndarray:
    vis = image_bgr.copy()
    H, W = vis.shape[:2]

    # 처리된 타일 경계 표시
    base_w = W / cols
    base_h = H / rows
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c + 1
            if idx in tile_filter:
                bx0, by0 = int(c * base_w), int(r * base_h)
                bx1, by1 = min(W, int((c+1)*base_w)), min(H, int((r+1)*base_h))
                cv2.rectangle(vis, (bx0, by0), (bx1, by1), (255, 255, 255), 1)

    # 마스크 + 순번 레이블
    font      = cv2.FONT_HERSHEY_SIMPLEX
    font_sc   = max(0.4, min(W, H) / 8000)
    thickness = max(1, int(font_sc * 2))

    for i, cat in enumerate(categories):
        color  = tuple(cat["color_bgr"])
        prefix = cat["name"][:3].upper()          # 예: RAI, CAT, BRA …
        for seq, s in enumerate(buckets[i], start=1):
            pts = np.array(s["points"], dtype=np.int32)
            overlay = vis.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.35, vis, 0.65, 0, vis)
            cv2.polylines(vis, [pts], True, color, 2)

            # 무게중심에 순번 표시
            cx = int(np.mean(pts[:, 0]))
            cy = int(np.mean(pts[:, 1]))
            score = float(s.get("score", 0.0))
            label = f"{prefix}{seq:03d} {score:.2f}"
            (tw, th), _ = cv2.getTextSize(label, font, font_sc, thickness)
            tx, ty = cx - tw // 2, cy + th // 2
            # 배경 박스
            cv2.rectangle(vis, (tx - 2, ty - th - 2), (tx + tw + 2, ty + 2),
                          (0, 0, 0), -1)
            cv2.putText(vis, label, (tx, ty), font, font_sc,
                        color, thickness, cv2.LINE_AA)

    # 범례
    lx, ly = W - 280, 20
    panel_h = len(categories) * 24 + 10
    vis[0:panel_h, lx-8:W] = (vis[0:panel_h, lx-8:W] * 0.35).astype(np.uint8)
    for i, cat in enumerate(categories):
        color  = tuple(cat["color_bgr"])
        prefix = cat["name"][:3].upper()
        cv2.rectangle(vis, (lx, ly-13), (lx+15, ly+3), color, -1)
        cv2.putText(vis, f"[{prefix}] {cat['name']}  ({len(buckets[i])})",
                    (lx+20, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1, cv2.LINE_AA)
        ly += 24

    return vis


# ── 메인 ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    ap.add_argument("--input",      required=True,              help="입력 이미지 경로")
    ap.add_argument("--output",     default=None,               help="출력 이미지 경로 (기본: 입력파일명_out.jpg)")
    ap.add_argument("--categories", default=None,               help="카테고리 JSON 경로 (기본: 내장 10개)")
    ap.add_argument("--tiles",      default="all",              help="처리할 타일 번호: 9-24 / 1,5,9 / all (기본: all)")
    ap.add_argument("--cols",       type=int,   default=8,      help="가로 타일 수 (기본: 8)")
    ap.add_argument("--rows",       type=int,   default=6,      help="세로 타일 수 (기본: 6)")
    ap.add_argument("--overlap",    type=float, default=0.10,   help="타일 중복 비율 (기본: 0.10)")
    ap.add_argument("--conf",       type=float, default=0.20,   help="SAM3 신뢰도 임계값 (기본: 0.20)")
    ap.add_argument("--workers",    type=int,   default=4,      help="병렬 스레드 수 (기본: 4)")
    ap.add_argument("--save-labels", action="store_true",      help="YOLO 폴리곤 포맷 .txt 라벨 파일 저장")
    args = ap.parse_args()

    img_path  = Path(args.input)
    buf       = np.fromfile(str(img_path), dtype=np.uint8)
    image_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if image_bgr is None:
        print(f"이미지 로드 실패: {img_path}"); return

    H, W          = image_bgr.shape[:2]
    total_tiles   = args.cols * args.rows
    tile_filter   = parse_tiles(args.tiles, total_tiles)
    categories    = load_categories(args.categories)
    combined_prompt = build_combined_prompt(categories)

    # cross-class NMS IoU: JSON > 기본값 0.45
    cc_iou = 0.45
    if args.categories:
        raw = json.loads(Path(args.categories).read_text(encoding="utf-8"))
        cc_iou = raw.get("cross_class_nms_iou", cc_iou)

    # SAM3 호출용 conf = 모든 카테고리 conf 중 최솟값 (낮은 쪽부터 받아 후처리로 필터)
    sam3_conf = min(cat.get("conf", args.conf) for cat in categories)

    print(f"이미지    : {W}×{H}")
    print(f"타일 그리드: {args.cols}×{args.rows}={total_tiles}개  |  처리 대상: {sorted(tile_filter)}")
    print(f"카테고리  : {len(categories)}개  |  중복: {args.overlap*100:.0f}%")
    print(f"SAM3 conf : {sam3_conf} (전체 최솟값)  |  cross-class NMS IoU: {cc_iou}")
    print(f"SAM3 호출 : {len(tile_filter)}회  |  병렬: {args.workers}스레드\n")

    t0 = time.time()
    all_shapes = detect_tiled(image_bgr, args.cols, args.rows, args.overlap,
                              sam3_conf, args.workers, tile_filter, combined_prompt)

    print(f"전체 검출 {len(all_shapes)}개 → 분류 + per-class conf 필터 + NMS...")
    buckets   = [[] for _ in categories]
    unmatched = 0
    for s in all_shapes:
        idx = label_to_category(s.get("label", ""), categories)
        if idx < 0:
            unmatched += 1
            continue
        # per-class conf 필터
        cat_conf = categories[idx].get("conf", args.conf)
        if float(s.get("score", 0.0)) < cat_conf:
            continue
        buckets[idx].append(s)

    # 클래스 내 NMS
    print("  [1] 클래스 내 NMS")
    for i, cat in enumerate(categories):
        before = len(buckets[i])
        buckets[i] = nms_shapes(buckets[i])
        print(f"    {cat['name']:18s}: {before:3d} → {len(buckets[i]):3d}개  (conf≥{cat.get('conf', args.conf)})")

    # 클래스 간 NMS
    total_before = sum(len(b) for b in buckets)
    print(f"  [2] cross-class NMS (IoU≥{cc_iou})")
    buckets = cross_class_nms(buckets, categories, cc_iou)
    total_after = sum(len(b) for b in buckets)
    print(f"    {total_before}개 → {total_after}개")

    if unmatched:
        print(f"  (미분류/conf미달 {unmatched}개 제외)")
    print(f"\n완료: {time.time()-t0:.0f}초")

    vis = draw_detections(image_bgr, buckets, categories,
                          tile_filter, args.cols, args.rows, args.overlap)
    h, w = vis.shape[:2]
    if max(h, w) > 4096:
        s = 4096 / max(h, w)
        vis = cv2.resize(vis, (int(w*s), int(h*s)))

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        tile_tag  = args.tiles.replace(",", "_").replace("-", "to")
        cat_tag   = Path(args.categories).stem if args.categories else "default"
        out_dir   = Path("output") / "detect"
        out_dir.mkdir(parents=True, exist_ok=True)
        base_name = f"{img_path.stem}_tiles{tile_tag}_{cat_tag}"
        n = 1
        while True:
            out_path = out_dir / f"{base_name}_{n:03d}.jpg"
            if not out_path.exists():
                break
            n += 1

    cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 93])[1].tofile(str(out_path))
    print(f"저장: {out_path}")

    if args.save_labels:
        label_path = out_path.with_suffix(".txt")
        with open(label_path, "w", encoding="utf-8") as f:
            for cls_idx, shapes in enumerate(buckets):
                for s in shapes:
                    pts_norm = [[px / W, py / H] for px, py in s["points"]]
                    coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in pts_norm)
                    f.write(f"{cls_idx} {coords}\n")
        print(f"라벨 저장: {label_path}")


if __name__ == "__main__":
    main()
