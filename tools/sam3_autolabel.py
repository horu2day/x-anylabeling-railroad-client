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
    """(pole_prompt, rail_prompt, bracket_prompt) 반환."""
    pole    = _load_prompt("pole.txt",
                           "railway catenary pole, overhead line support pole, catenary mast")
    rail    = _load_prompt("rail.txt", "railroad, railway")
    bracket = _load_prompt("bracket.txt",
                           "catenary pole top, pole cross arm, horizontal cantilever beam, bracket arm")
    return pole, rail, bracket


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
                    gap_close: int = 500,
                    lateral_expand_ratio: float = 0.0) -> np.ndarray:
    """철로 polygon → binary mask → 수평 갭 보정(beam 차폐) → margin px 팽창.
    lateral_expand_ratio > 0 이면 추가로 좌우 수평 팽창 (가장자리 기둥 포착)."""
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
    if lateral_expand_ratio > 0 and mask.any():
        lat = int(W * lateral_expand_ratio)
        if lat > 0:
            lat_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (lat * 2 + 1, 1))
            mask = cv2.dilate(mask, lat_kernel)
    return mask


def pole_in_zone(pole_det: tuple, zone_mask: np.ndarray) -> bool:
    """전철주 bbox 중심 열(column)의 y1~y2 범위 중 zone_mask와 교차하면 True.
    기둥은 철로 근처에 기저부를 두고 위/아래로 뻗으므로
    중심 한 점 대신 세로 전체 범위로 판단."""
    _, _, x1, y1, x2, y2 = pole_det
    H, W = zone_mask.shape
    cx = int(max(0, min((x1 + x2) / 2, W - 1)))
    ys = int(max(0, y1))
    ye = int(min(H, y2 + 1))
    return bool(zone_mask[ys:ye, cx].any())


def _filter_pole_has_beam(pole_pairs: list, beam_pairs: list) -> list:
    """A ∩ C: 전철주 bbox와 겹치는 Beam/Arm이 하나라도 있는 기둥만 유지.
    Beam/Arm bbox와 기둥 bbox의 IoU > 0 (단순 교차)으로 판단."""
    beam_boxes = [(x1, y1, x2, y2)
                  for (_, _, x1, y1, x2, y2), _ in beam_pairs]

    def overlaps_any_beam(pole_det):
        _, _, px1, py1, px2, py2 = pole_det
        for bx1, by1, bx2, by2 in beam_boxes:
            ix1 = max(px1, bx1); iy1 = max(py1, by1)
            ix2 = min(px2, bx2); iy2 = min(py2, by2)
            if ix2 > ix1 and iy2 > iy1:
                return True
        return False

    return [(d, s) for d, s in pole_pairs if overlaps_any_beam(d)]


def filter_long_beams(beam_pairs: list, rail_shapes: list,
                      min_beam_width: int = 400,
                      ratio_thresh: float = 2.0,
                      rail_x_margin: int = 500,
                      debug: bool = False) -> list:
    """역사형 portal frame의 긴 빔만 유지.
    조건:
      1) bbox 폭 >= min_beam_width   (가로등은 좁음)
      2) 폭/높이 >= ratio_thresh     (가로형이어야 함)
      3) 빔 x-범위가 검출된 rail x-범위 ±rail_x_margin 내   (도로 차단)
    """
    if not rail_shapes:
        if debug:
            print(f"    [Beam filter] rail 미검출 → 빔 zone 보강 비활성")
        return []

    rail_xs = []
    for s in rail_shapes:
        if s.get("shape_type") == "polygon":
            xs = [p[0] for p in s["points"]]
            rail_xs.extend([min(xs), max(xs)])
    if not rail_xs:
        return []
    rail_x_min, rail_x_max = min(rail_xs), max(rail_xs)

    kept = []
    for (label, conf, x1, y1, x2, y2), shape in beam_pairs:
        bw = x2 - x1
        bh = max(1, y2 - y1)
        ratio = bw / bh
        if bw < min_beam_width:
            if debug:
                print(f"    DROP beam [{label} {conf:.2f}] w={bw:.0f} (최소:{min_beam_width})")
            continue
        if ratio < ratio_thresh:
            if debug:
                print(f"    DROP beam [{label} {conf:.2f}] w/h={ratio:.2f} (최소:{ratio_thresh})")
            continue
        if x2 < rail_x_min - rail_x_margin or x1 > rail_x_max + rail_x_margin:
            if debug:
                print(f"    DROP beam [{label} {conf:.2f}] rail x-범위 벗어남")
            continue
        kept.append(((label, conf, x1, y1, x2, y2), shape))
    return kept


def add_beams_to_zone(zone_mask: np.ndarray, beam_pairs: list,
                      downward_extend: int = 400, side_margin: int = 100) -> np.ndarray:
    """필터된 긴 빔의 bbox를 zone_mask에 추가.
    빔 위치 + 아래쪽 downward_extend px → 레일이 빔에 차폐된 구간 보강."""
    H, W = zone_mask.shape
    for (_, _, x1, y1, x2, y2), _ in beam_pairs:
        bx1 = max(0, int(x1) - side_margin)
        bx2 = min(W, int(x2) + side_margin)
        by1 = max(0, int(y1))
        by2 = min(H, int(y2) + downward_extend)
        zone_mask[by1:by2, bx1:bx2] = 255
    return zone_mask


def _mask_to_largest_polygon(mask: np.ndarray, min_area: int = 50) -> list:
    """이진 마스크에서 가장 큰 외곽선 → polygon points."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_area:
        return None
    epsilon = 0.002 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)
    return [[float(p[0][0]), float(p[0][1])] for p in approx]


def split_pole_arm(pts: list, H: int, W: int,
                   arm_min_factor: float = 0.5) -> tuple:
    """전철주 polygon을 세로 기둥(column)과 가로 암(arm)으로 분리.

    방법 (row-width 분석):
      1. polygon → mask
      2. 각 가로 row에서 mask width 측정
      3. 좁은 row 절반의 중앙값 = column 폭 추정
      4. column 중심 x = 좁은 row들의 mask 중점 중앙값
      5. column mask = column 중심 ± column 폭/2 범위 내 mask
      6. arm mask = column 범위 외 mask (충분히 가로로 뻗은 row만)

    arm_min_factor: arm으로 인정할 row의 최소 width = column 폭 × factor
    반환: (column_pts, arm_pts | None)
    """
    polygon = np.array(pts, dtype=np.int32)
    if len(polygon) < 3:
        return pts, None

    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)

    x1 = max(0, int(polygon[:, 0].min()))
    x2 = min(W - 1, int(polygon[:, 0].max()))
    y1 = max(0, int(polygon[:, 1].min()))
    y2 = min(H - 1, int(polygon[:, 1].max()))
    sub = mask[y1:y2 + 1, x1:x2 + 1]

    row_widths = (sub > 0).sum(axis=1).astype(np.float32)
    valid = row_widths > 0
    if valid.sum() < 5:
        return pts, None

    # column 폭 = 좁은 row들의 중앙값 (전체의 좁은 절반)
    sorted_widths = np.sort(row_widths[valid])
    column_w = float(np.median(sorted_widths[:max(1, len(sorted_widths) // 2)]))
    if column_w < 5:
        return pts, None

    # column 중심 x 추정 — 좁은 row들의 mask 중점 중앙값
    narrow_threshold = column_w * 1.3
    cx_list = []
    for i, w in enumerate(row_widths):
        if 0 < w <= narrow_threshold:
            row_px = np.where(sub[i] > 0)[0]
            if len(row_px) > 0:
                cx_list.append((row_px[0] + row_px[-1]) / 2)
    if not cx_list:
        return pts, None
    column_cx = int(np.median(cx_list)) + x1
    column_half = max(4, int(column_w * 0.7))

    col_x_min = max(0, column_cx - column_half)
    col_x_max = min(W, column_cx + column_half + 1)

    column_mask = np.zeros_like(mask)
    column_mask[:, col_x_min:col_x_max] = mask[:, col_x_min:col_x_max]

    arm_mask = mask.copy()
    arm_mask[:, col_x_min:col_x_max] = 0

    # arm: 가로로 충분히 뻗은 row만 유지
    arm_min_w = max(20, int(column_w * arm_min_factor))
    arm_row_widths = (arm_mask > 0).sum(axis=1)
    for y in range(H):
        if arm_row_widths[y] < arm_min_w:
            arm_mask[y, :] = 0

    column_pts = _mask_to_largest_polygon(column_mask)
    arm_pts = _mask_to_largest_polygon(arm_mask) if arm_mask.any() else None

    # arm이 너무 작으면 폐기 (column 면적의 5% 미만)
    if arm_pts and column_pts:
        c_area = cv2.contourArea(np.array(column_pts, dtype=np.int32))
        a_area = cv2.contourArea(np.array(arm_pts, dtype=np.int32))
        if a_area < c_area * 0.05:
            arm_pts = None

    return (column_pts or pts), arm_pts


# ── 후처리: 합치기 + Skeleton 분리 (column / beam) ────────────────────────────
def _compute_skeleton(mask: np.ndarray) -> np.ndarray:
    """이진 마스크 → 1px 두께 skeleton (opencv-contrib-python의 ximgproc.thinning)."""
    return cv2.ximgproc.thinning(mask)


def _find_branch_points(skel: np.ndarray) -> np.ndarray:
    """Skeleton의 분기점(8-neighbor degree ≥ 3) 마스크 반환."""
    skel_bin = (skel > 0).astype(np.uint8)
    # 8-neighbor 합 = 3x3 box filter * 9 - center
    box_kernel = np.ones((3, 3), dtype=np.float32)
    nbr = cv2.filter2D(skel_bin.astype(np.float32), cv2.CV_32F, box_kernel) - skel_bin
    branch = (skel_bin == 1) & (nbr >= 3)
    return branch.astype(np.uint8) * 255


def _compute_degree_map(skel: np.ndarray) -> np.ndarray:
    """Skeleton 각 픽셀의 8-neighbor degree (자기 자신 제외)."""
    skel_bin = (skel > 0).astype(np.float32)
    box = np.ones((3, 3), dtype=np.float32)
    nbr = cv2.filter2D(skel_bin, cv2.CV_32F, box) - skel_bin
    deg = (nbr * skel_bin).astype(np.int32)
    return deg


def _split_one_component(mask: np.ndarray) -> tuple:
    """하나의 connected component를 column / beam mask로 분리 (위상 규칙).

    라멘 구조 위상:
      - beam (가로 빔): 양 끝이 branch에 연결 → 원본 free endpoint 0개
      - column (세로 기둥): 한 끝이 branch, 다른 끝이 free → 원본 free endpoint 1개+

    위상 규칙이 적용되지 않는 폴리곤(단일 segment, 가로등 'ㅏ', "ㄱ" 등)은
    라멘이 아니므로 (None, None) 반환 → 호출부에서 폴리곤 삭제.
    """
    skel = _compute_skeleton(mask)
    if not skel.any():
        return None, None

    # 원본 skeleton의 degree map
    deg = _compute_degree_map(skel)
    free_endpoint_mask = ((deg == 1) & (skel > 0))
    branch_mask = ((deg >= 3) & (skel > 0)).astype(np.uint8) * 255

    # branch 제거 → segment 분리
    if branch_mask.any():
        bk = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        seg_skel = cv2.bitwise_and(skel, cv2.bitwise_not(cv2.dilate(branch_mask, bk)))
    else:
        # branch가 전혀 없음 = 단일 선/곡선 → 라멘 아님
        return None, None

    num, labels = cv2.connectedComponents(seg_skel)
    interior_seeds = []  # free endpoint 0개 segment
    edge_seeds = []      # free endpoint 1개+ segment

    for sid in range(1, num):
        seg = (labels == sid)
        if seg.sum() < 10:
            continue
        n_free = int((free_endpoint_mask & seg).sum())
        seg_u8 = seg.astype(np.uint8) * 255
        if n_free == 0:
            interior_seeds.append(seg_u8)
        else:
            edge_seeds.append(seg_u8)

    # 라멘이 되려면 interior(beam)와 edge(column)가 모두 있어야 함
    if not interior_seeds or not edge_seeds:
        return None, None

    col_seed = np.zeros_like(mask)
    beam_seed = np.zeros_like(mask)
    for s in edge_seeds:
        col_seed = cv2.bitwise_or(col_seed, s)
    for s in interior_seeds:
        beam_seed = cv2.bitwise_or(beam_seed, s)

    col_dist = cv2.distanceTransform(255 - col_seed, cv2.DIST_L2, 5)
    beam_dist = cv2.distanceTransform(255 - beam_seed, cv2.DIST_L2, 5)

    mask_bool = mask > 0
    col_region = (mask_bool & (col_dist <= beam_dist)).astype(np.uint8) * 255
    beam_region = (mask_bool & (col_dist > beam_dist)).astype(np.uint8) * 255
    return col_region, beam_region


def _extract_polygons(mask: np.ndarray, min_area: int = 200,
                      epsilon_factor: float = 0.002) -> list:
    """Mask의 모든 connected component → polygon points 리스트."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        eps = epsilon_factor * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, eps, True)
        pts = [[float(p[0][0]), float(p[0][1])] for p in approx]
        if len(pts) >= 3:
            polys.append(pts)
    return polys


def skeleton_split_columns_beams(pole_pairs: list, H: int, W: int,
                                  min_component_area: int = 1000,
                                  min_polygon_area: int = 200) -> tuple:
    """Pole polygon별로 skeleton split → column / beam 분류된 polygon 추출.

    반환: (column_polygons, beam_polygons)
    """
    if not pole_pairs:
        return [], []

    merged = np.zeros((H, W), dtype=np.uint8)
    for _, shape in pole_pairs:
        if shape.get("shape_type") == "polygon":
            pts = np.array(shape["points"], dtype=np.int32)
            cv2.fillPoly(merged, [pts], 255)
    if not merged.any():
        return [], []

    # closing 미적용 — SAM3.1 polygon 형상 그대로 사용
    num, labels = cv2.connectedComponents(merged)
    all_col = np.zeros_like(merged)
    all_beam = np.zeros_like(merged)
    kept = 0
    dropped = 0
    for cid in range(1, num):
        comp = (labels == cid).astype(np.uint8) * 255
        if int(comp.sum() / 255) < min_component_area:
            continue
        col_m, beam_m = _split_one_component(comp)
        if col_m is None or beam_m is None:
            dropped += 1
            continue
        all_col = cv2.bitwise_or(all_col, col_m)
        all_beam = cv2.bitwise_or(all_beam, beam_m)
        kept += 1
    print(f"  [Topo] 라멘 위상 통과: {kept}개, 비라멘 폐기: {dropped}개")

    column_polys = _extract_polygons(all_col, min_area=min_polygon_area)
    beam_polys = _extract_polygons(all_beam, min_area=min_polygon_area)
    return column_polys, beam_polys


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
             rail_shapes: list = None, long_beam_pairs: list = None):
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
    # 레일 zone 시각화 (빨간색, 반투명)
    if rail_shapes:
        for s in rail_shapes:
            if s.get("shape_type") == "polygon":
                pts = np.array(s["points"], dtype=np.int32)
                overlay = vis.copy()
                cv2.fillPoly(overlay, [pts], (0, 0, 200))
                cv2.addWeighted(overlay, 0.25, vis, 0.75, 0, vis)
                cv2.polylines(vis, [pts], True, (0, 0, 255), 2)
    # 긴 빔 시각화 (자홍색 bbox)
    if long_beam_pairs:
        for (label, conf, x1, y1, x2, y2), _ in long_beam_pairs:
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)),
                          (255, 0, 255), 2)
            cv2.putText(vis, f"beam {conf:.2f}", (int(x1), max(0, int(y1) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
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
                  min_pole_h: int = 160,
                  rail_margin: int = 300,
                  rail_gap_close: int = 500,
                  beam_min_width: int = 400,
                  beam_ratio: float = 2.0,
                  beam_downward: int = 400,
                  rail_lateral: float = 0.0,
                  raw_polygons: bool = False) -> int:

    buf = np.fromfile(str(image_path), dtype=np.uint8)
    image_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if image_bgr is None:
        print(f"  이미지 로드 실패: {image_path}")
        return 0
    H, W = image_bgr.shape[:2]

    pole_prompt, rail_prompt, bracket_prompt = _get_prompts()
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

    # [C] Beam/Arm 쿼리는 사용하지 않음 (검출되는 "긴 빔"이 실제로는 레일 일부였음)

    # A ∩ Zone 필터 (Zone = rail + 측면 확장)
    if rail_margin >= 0:
        if rail_shapes:
            zone_mask = build_zone_mask(rail_shapes, H, W, rail_margin, rail_gap_close,
                                        lateral_expand_ratio=rail_lateral)
            if rail_lateral > 0:
                print(f"  [Zone] rail + 측면 {rail_lateral*100:.0f}%")
            before = len(pole_pairs)
            kept = []
            for d, s in pole_pairs:
                if pole_in_zone(d, zone_mask):
                    kept.append((d, s))
                else:
                    label, conf, x1, y1, x2, y2 = d
                    print(f"    DROP zone [{label} {conf:.2f}] bbox=({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f})")
            pole_pairs = kept
            print(f"  [A∩Zone] 필터(+{rail_margin}px): {before}→{len(pole_pairs)}개")
        else:
            print(f"  ! [B] 철로 미검출 → 전체 제외")
            pole_pairs = []

    # 후처리: raw_polygons=True 면 SAM 원본 폴리곤만 cls=0으로 그대로 저장
    if raw_polygons:
        all_pairs = []
        for det, shape in pole_pairs:
            shape["_cls_id"] = 0
            shape["label"] = "catenary_pole"
            all_pairs.append((det, shape))
        print(f"  [Raw] SAM 폴리곤 {len(all_pairs)}개 그대로 저장 (split 미적용)")
    else:
        # skeleton split (column / beam)
        column_polys, beam_polys = skeleton_split_columns_beams(pole_pairs, H, W)
        column_pairs = []
        arm_pairs = []
        for poly in column_polys:
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            det = ("catenary_pole", 1.0, min(xs), min(ys), max(xs), max(ys))
            shape = {"shape_type": "polygon", "points": poly,
                     "_cls_id": 0, "label": "catenary_pole"}
            column_pairs.append((det, shape))
        for poly in beam_polys:
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            det = ("bracket", 1.0, min(xs), min(ys), max(xs), max(ys))
            shape = {"shape_type": "polygon", "points": poly,
                     "_cls_id": 1, "label": "bracket"}
            arm_pairs.append((det, shape))
        print(f"  [Split] column {len(column_pairs)}개, beam {len(arm_pairs)}개")
        all_pairs = column_pairs + arm_pairs
    if not all_pairs:
        return 0

    save_yolo_label(out_label_dir / (image_path.stem + ".txt"), all_pairs, W, H)
    save_vis(out_vis_dir / (image_path.stem + "_vis.jpg"), image_bgr, all_pairs,
             rail_shapes, None)
    return len(all_pairs)


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
    parser.add_argument("--min_pole_h",     type=int,   default=160,
                        help="전철주 최소 높이 px (기본:160, 드론 거리 변화 마진 포함)")
    parser.add_argument("--rail_margin",    type=int,   default=300,
                        help="철로 mask 팽창 거리 px (-1=비활성, 기본:300)")
    parser.add_argument("--rail_gap_close", type=int,   default=500,
                        help="beam 차폐 갭 보정 수평 closing 폭 px (0=비활성, 기본:500)")
    parser.add_argument("--beam_min_width", type=int,   default=400,
                        help="긴 빔 최소 폭 px - 가로등/짧은 arm 차단 (기본:400, 역사형만)")
    parser.add_argument("--beam_ratio",     type=float, default=2.0,
                        help="긴 빔 폭/높이 최소 비율 (기본:2.0)")
    parser.add_argument("--beam_downward",  type=int,   default=400,
                        help="빔 zone 아래 확장 px (기본:400)")
    parser.add_argument("--rail_lateral",   type=float, default=0.0,
                        help="rail zone 측면(좌우) 확장 비율 (이미지 폭 기준, 예:0.10=10%%)")
    parser.add_argument("--raw_polygons",   action="store_true",
                        help="후처리 skeleton split 없이 SAM 원본 폴리곤만 저장")
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

    pole_prompt, rail_prompt, bracket_prompt = _get_prompts()
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
                               args.min_pole_h, args.rail_margin, args.rail_gap_close,
                               args.beam_min_width, args.beam_ratio, args.beam_downward,
                               args.rail_lateral, args.raw_polygons)

    print(f"\n완료: 총 {total}개 라벨 생성")
    print(f"라벨: {out_label}")
    print(f"시각화: {out_vis}")


if __name__ == "__main__":
    main()
