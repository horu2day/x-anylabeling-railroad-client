"""SAM 원본 폴리곤의 skeleton을 까만색으로 이미지에 겹쳐 표시.
union mask 기준 skeleton을 그리므로, 겹쳐있는 fragment들의 합쳐진 골격이 보임.

사용:
    python tools/render_skeleton_overlay.py \
        --image data/역사이미지/slope/DJI_20260306113842_0006.JPG \
        --label output/autolabel/raw_0006/labels/DJI_20260306113842_0006.txt \
        --output output/autolabel/raw_0006/vis_skeleton.jpg
"""
import argparse
import base64
import requests
import cv2
import numpy as np
from pathlib import Path

SAM3_SERVER   = "http://localhost:8000"
SAM3_MODEL_ID = "segment_anything_3"

SUB_CATEGORIES = [
    {"name": "pole_shaft",  "name_kr": "전주기둥",
     "prompt": "vertical steel pole shaft, cylindrical mast, tubular pole body",
     "keywords": ["pole", "shaft", "mast", "tubular"],
     "color_bgr": [0, 200, 255]},
    {"name": "bracket_arm", "name_kr": "브라켓/암",
     "prompt": "horizontal cantilever arm, bracket arm, cross arm, pole cross arm",
     "keywords": ["bracket", "cantilever", "cross arm", "arm"],
     "color_bgr": [255, 100, 0]},
    {"name": "insulator",   "name_kr": "애자",
     "prompt": "ceramic insulator, glass insulator, suspension insulator, string of insulators",
     "keywords": ["insulator"],
     "color_bgr": [0, 255, 255]},
    {"name": "wire",        "name_kr": "가선/조가선",
     "prompt": "overhead contact wire, catenary wire, messenger wire, electric cable",
     "keywords": ["wire", "cable", "catenary"],
     "color_bgr": [200, 200, 255]},
]


def _sam3_call(crop_bgr: np.ndarray, prompt: str, conf: float = 0.15) -> list:
    h, w = crop_bgr.shape[:2]
    scale = 1.0
    if max(h, w) > 1280:
        scale = 1280 / max(h, w)
        crop_bgr = cv2.resize(crop_bgr, (int(w * scale), int(h * scale)))
    _, buf = cv2.imencode(".jpg", crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    b64 = base64.b64encode(buf).decode()
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
    except Exception as e:
        print(f"    SAM3 오류: {e}")
        return []


def _label_to_subcat(label_str: str):
    ll = label_str.lower()
    for cat in SUB_CATEGORIES:
        if any(kw in ll for kw in cat["keywords"]):
            return cat
    return None


def _subdetect_group(img: np.ndarray, group_polys: list, pad: int = 80) -> int:
    """그룹 폴리곤들의 union bbox crop → SAM3 sub-detect → img 오버레이."""
    all_pts = np.vstack(group_polys)
    H, W = img.shape[:2]
    x0 = max(0, int(all_pts[:, 0].min()) - pad)
    y0 = max(0, int(all_pts[:, 1].min()) - pad)
    x1 = min(W, int(all_pts[:, 0].max()) + pad)
    y1 = min(H, int(all_pts[:, 1].max()) + pad)

    crop = img[y0:y1, x0:x1].copy()
    combined_prompt = ", ".join(c["prompt"] for c in SUB_CATEGORIES)
    shapes = _sam3_call(crop, combined_prompt)

    font = cv2.FONT_HERSHEY_SIMPLEX
    for s in shapes:
        cat = _label_to_subcat(s.get("label", ""))
        color = tuple(cat["color_bgr"]) if cat else (128, 128, 128)
        pts = np.array([[int(px + x0), int(py + y0)] for px, py in s["points"]], dtype=np.int32)
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, 0.30, img, 0.70, 0, img)
        cv2.polylines(img, [pts], True, color, 2)
        cx, cy = int(pts[:, 0].mean()), int(pts[:, 1].mean())
        name = cat["name"] if cat else "?"
        cv2.putText(img, name, (cx, cy), font, 0.7, color, 2, cv2.LINE_AA)

    cv2.rectangle(img, (x0, y0), (x1, y1), (255, 200, 0), 2)
    return len(shapes)


def render(image_path: Path, label_path: Path, output_path: Path,
           selected_indices=None, branch_radius: int = 10, class_ids=None,
           subdetect: bool = False, subdetect_polys: set = None):
    buf = np.fromfile(str(image_path), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    H, W = img.shape[:2]

    text = label_path.read_text(encoding="utf-8").strip() if label_path.exists() else ""
    if class_ids is not None:
        text = "\n".join(
            ln for ln in text.splitlines()
            if ln.split() and int(ln.split()[0]) in class_ids
        )
    all_polygons = []
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        coords = list(map(float, parts[1:]))
        pts = np.array(
            [[coords[i] * W, coords[i + 1] * H] for i in range(0, len(coords), 2)],
            dtype=np.int32,
        )
        if len(pts) >= 3:
            all_polygons.append(pts)

    if selected_indices is not None:
        keep_idx = [i for i in sorted(selected_indices) if 0 <= i < len(all_polygons)]
    else:
        keep_idx = list(range(len(all_polygons)))
    polygons = [all_polygons[i] for i in keep_idx]
    print(f"  total polygons in label: {len(all_polygons)}, "
          f"selected: {len(polygons)} (indices: {keep_idx})")

    # 폴리곤 외곽선 (연한 색) — 원본 인덱스 표시
    for orig_idx, pts in zip(keep_idx, polygons):
        cv2.polylines(img, [pts], True, (180, 180, 100), 2)
        cx = int(pts[:, 0].mean())
        cy = int(pts[:, 1].mean())
        cv2.putText(img, str(orig_idx), (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 5)
        cv2.putText(img, str(orig_idx), (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)

    # 모든 폴리곤 합친 마스크 → skeleton
    merged = np.zeros((H, W), dtype=np.uint8)
    for pts in polygons:
        cv2.fillPoly(merged, [pts], 255)
    # 미세한 픽셀 overlap으로 다른 구조가 합쳐지는 것 방지: 작은 erosion
    # 5px dilation으로 인접 폴리곤 gap 브리지 후 3px erosion으로 복원
    merged_dilated = cv2.dilate(merged, np.ones((11, 11), dtype=np.uint8))
    merged_eroded  = cv2.erode(merged_dilated, np.ones((7, 7), dtype=np.uint8))
    skel = cv2.ximgproc.thinning(merged_eroded)

    # skeleton 두껍게 (시인성)
    skel_thick = cv2.dilate(skel, np.ones((5, 5), dtype=np.uint8))
    img[skel_thick > 0] = (0, 0, 0)

    # spur pruning: 짧은 dead-end arm 제거 → 진짜 branch만 남음
    _box3 = np.ones((3, 3), dtype=np.float32)

    def _prune_spurs(sk, min_arm_px):
        """branch pixel 제거 후 arm component 분석, 짧은 arm 제거."""
        s_i = (sk > 0).astype(np.int32)
        sp2 = np.pad(s_i, 1, mode='constant')
        h2, w2 = s_i.shape
        n2 = [sp2[0:h2, 0:w2], sp2[0:h2, 1:w2+1], sp2[0:h2, 2:w2+2],
              sp2[1:h2+1, 2:w2+2], sp2[2:h2+2, 2:w2+2], sp2[2:h2+2, 1:w2+1],
              sp2[2:h2+2, 0:w2], sp2[1:h2+1, 0:w2]]
        cn2 = np.zeros((h2, w2), dtype=np.int32)
        for i2 in range(8):
            cn2 += (1 - n2[i2]) * n2[(i2 + 1) % 8]
        cn2 *= s_i
        arm_sk = sk.copy()
        arm_sk[cn2 >= 3] = 0  # branch pixel 제거
        n_arm, arm_lbl, arm_stats, arm_cents = cv2.connectedComponentsWithStats(arm_sk)
        sk_out = sk.copy()
        spur_cs = []
        for a_id in range(1, n_arm):
            if arm_stats[a_id, cv2.CC_STAT_AREA] < min_arm_px:
                sk_out[arm_lbl == a_id] = 0
                spur_cs.append((int(arm_cents[a_id, 0]), int(arm_cents[a_id, 1])))
        return sk_out, spur_cs

    skel_pruned, spur_centroids = _prune_spurs(skel, min_arm_px=30)
    print(f"  spur pruning: {len(spur_centroids)}개 arm 제거 (min_arm_px=30)")

    skel_pf = (skel_pruned > 0).astype(np.float32)
    nbr_p = cv2.filter2D(skel_pf, cv2.CV_32F, _box3) - skel_pf
    deg_p = (nbr_p * skel_pf).astype(np.int32)

    branch_mask_global   = ((deg_p >= 3) & (skel_pruned > 0)).astype(np.uint8) * 255
    endpoint_mask_global = ((deg_p == 1) & (skel_pruned > 0)).astype(np.uint8) * 255

    def cluster_centroids(mask, cluster_radius):
        if not mask.any():
            return []
        dilated = cv2.dilate(mask, np.ones((cluster_radius, cluster_radius), np.uint8))
        num, _, _, cents = cv2.connectedComponentsWithStats(dilated)
        return [(int(cents[i, 0]), int(cents[i, 1])) for i in range(1, num)]

    def extreme_endpoint_pair(eps):
        if len(eps) < 2:
            return None
        best, best_d2 = None, -1
        for i in range(len(eps)):
            for j in range(i + 1, len(eps)):
                dx = eps[i][0] - eps[j][0]
                dy = eps[i][1] - eps[j][1]
                d2 = dx * dx + dy * dy
                if d2 > best_d2:
                    best_d2 = d2
                    best = (eps[i], eps[j])
        return best

    # 그룹핑 = pruned skeleton의 connected component
    num_comp, comp_labels = cv2.connectedComponents(skel_pruned)

    # 진단: 각 polygon이 어느 component에 속하는지 + poly_idx→cids 역매핑 빌드
    print(f"\n  [Diagnostic] skeleton components: {num_comp - 1}")
    poly_to_cids: dict = {}
    for i, pts in enumerate(polygons):
        poly_mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(poly_mask, [pts], 255)
        in_skel = (skel_pruned > 0) & (poly_mask > 0)
        if in_skel.any():
            cids = sorted(set(int(c) for c in np.unique(comp_labels[in_skel]) if c > 0))
            poly_to_cids[i] = cids
            print(f"    poly {i:>2d}: component(s) {cids}")
        else:
            print(f"    poly {i:>2d}: (no skeleton inside — too small or eroded)")
    print()

    # subdetect_polys가 지정된 경우, 해당 poly들이 속한 component만 sub-detect
    do_subdetect = subdetect or bool(subdetect_polys)
    target_cids: set = set()
    if subdetect_polys:
        for pi in subdetect_polys:
            for cid_ in poly_to_cids.get(pi, []):
                target_cids.add(cid_)
        print(f"  sub-detect 대상 그룹: {sorted(target_cids)} (poly {sorted(subdetect_polys)} 기준)\n")

    # cid → 해당 component에 속하는 polygon pts 목록 (subdetect용)
    comp_to_polys: dict = {}
    if do_subdetect:
        for pts in polygons:
            pmask = np.zeros((H, W), dtype=np.uint8)
            cv2.fillPoly(pmask, [pts], 255)
            in_s = (skel_pruned > 0) & (pmask > 0)
            if in_s.any():
                for cid_ in set(int(c) for c in comp_labels[in_s] if c > 0):
                    comp_to_polys.setdefault(cid_, []).append(pts)

    total_branches = 0
    total_endpoints = 0
    total_groups = 0
    total_lines = 0

    for cid in range(1, num_comp):
        comp_mask = (comp_labels == cid)
        if int(comp_mask.sum()) < 50:
            continue
        total_groups += 1

        comp_skel = (skel_pruned * comp_mask.astype(np.uint8)).astype(np.uint8)
        comp_branch   = cv2.bitwise_and(branch_mask_global, comp_skel)
        comp_endpoint = cv2.bitwise_and(endpoint_mask_global, comp_skel)

        branches  = cluster_centroids(comp_branch, 7)
        endpoints = cluster_centroids(comp_endpoint, 5)

        # Virtual endpoints: endpoint 3개인 component에만 적용 (라멘에서 1개 누락된 경우).
        # 알고리즘:
        #   1) skeleton bbox 계산 → 4 변 (top/bottom/left/right)
        #   2) 각 endpoint를 가장 가까운 변에 배정
        #   3) 4 변 중 endpoint 없는 "빠진 방향" 찾음
        #   4) 빠진 방향에 따라 branch 중 극값(가장 X/Y가 작거나 큰)을 virtual endpoint로
        virtual_endpoints = []
        if len(endpoints) == 3 and branches:
            ys_arr, xs_arr = np.where(comp_skel > 0)
            if len(ys_arr) > 0:
                xmin, xmax = int(xs_arr.min()), int(xs_arr.max())
                ymin, ymax = int(ys_arr.min()), int(ys_arr.max())
                # 각 endpoint를 가장 가까운 변에 배정
                occupied = set()
                for ex, ey in endpoints:
                    d_top = ey - ymin
                    d_bot = ymax - ey
                    d_left = ex - xmin
                    d_right = xmax - ex
                    side, _ = min(
                        (('top', d_top), ('bottom', d_bot),
                         ('left', d_left), ('right', d_right)),
                        key=lambda x: x[1])
                    occupied.add(side)
                # 빠진 방향들
                all_sides = {'top', 'bottom', 'left', 'right'}
                missing_sides = all_sides - occupied
                # 각 빠진 방향마다 극값 branch 찾기
                for ms in missing_sides:
                    if ms == 'right':
                        target = max(branches, key=lambda b: b[0])
                    elif ms == 'left':
                        target = min(branches, key=lambda b: b[0])
                    elif ms == 'top':
                        target = min(branches, key=lambda b: b[1])
                    elif ms == 'bottom':
                        target = max(branches, key=lambda b: b[1])
                    too_close = any((target[0] - px) ** 2 + (target[1] - py) ** 2 < 80 ** 2
                                    for px, py in endpoints + virtual_endpoints)
                    if not too_close:
                        virtual_endpoints.append(target)
                print(f"    virtual endpoint: missing={list(missing_sides)}, "
                      f"added={virtual_endpoints}")

        # 통합 endpoint 리스트 (real + virtual)
        endpoints_all = endpoints + virtual_endpoints
        total_branches += len(branches)
        total_endpoints += len(endpoints_all)

        # ── 새 알고리즘: 최상단 노드 기반 topology ──────────────────────
        if len(endpoints_all) < 2:
            pass
        else:
            def node_dist(a, b):
                return ((a[0]-b[0])**2 + (a[1]-b[1])**2) ** 0.5

            # 1. 최상단 노드 T (y 최솟값)
            top_i = min(range(len(endpoints_all)),
                        key=lambda i: (endpoints_all[i][1], endpoints_all[i][0]))
            T = endpoints_all[top_i]
            other_idx = [i for i in range(len(endpoints_all)) if i != top_i]

            connected_idx = {top_i}
            lines_to_draw = []

            if len(other_idx) == 1:
                j = other_idx[0]
                lines_to_draw.append((T, endpoints_all[j]))
                connected_idx.add(j)
                longer_i = j
            else:
                # 2. Left(min x), Right(max x) → T에서 이 둘에만 연결
                left_i  = min(other_idx, key=lambda i: endpoints_all[i][0])
                right_i = max(other_idx, key=lambda i: endpoints_all[i][0])
                lines_to_draw.append((T, endpoints_all[left_i]))
                lines_to_draw.append((T, endpoints_all[right_i]))
                connected_idx.update([left_i, right_i])

                # 3. 긴 쪽 결정
                d_l = node_dist(T, endpoints_all[left_i])
                d_r = node_dist(T, endpoints_all[right_i])
                longer_i = left_i if d_l >= d_r else right_i

            # 4. 체이닝: 긴 쪽 끝점에서 y+ 방향 미연결 노드로 계속 연결
            current_i = longer_i
            while True:
                cur = endpoints_all[current_i]
                below = [(i, node_dist(cur, endpoints_all[i]))
                         for i in range(len(endpoints_all))
                         if i not in connected_idx and endpoints_all[i][1] > cur[1]]
                if not below:
                    break
                next_i = min(below, key=lambda x: x[1])[0]
                lines_to_draw.append((cur, endpoints_all[next_i]))
                connected_idx.add(next_i)
                current_i = next_i

            for p1, p2 in lines_to_draw:
                cv2.line(img, p1, p2, (0, 255, 255), 5)
                total_lines += 1
                print(f"    line {p1}→{p2}: {node_dist(p1,p2):.0f}px")

        # 노드 표시 (line 위에 올라오게)
        for cx, cy in branches:
            cv2.circle(img, (cx, cy), 14, (0, 220, 0), 3)       # 초록 중간 원 (진짜 T/X junction)
        for cx, cy in endpoints:
            cv2.circle(img, (cx, cy), 18, (0, 255, 0), 4)
        # virtual endpoint = 오렌지 원
        for cx, cy in virtual_endpoints:
            cv2.circle(img, (cx, cy), 18, (0, 140, 255), 4)

        should_sub = do_subdetect and cid in comp_to_polys
        if should_sub and target_cids:
            should_sub = cid in target_cids
        if should_sub:
            n = _subdetect_group(img, comp_to_polys[cid])
            print(f"    그룹 {cid}: sub-detect {n}개")

    # 제거된 spur arm centroids = 분홍 작은 원
    for cx, cy in spur_centroids:
        cv2.circle(img, (cx, cy), 6, (180, 120, 180), 2)

    print(f"  {len(polygons)} polygons, skeleton 픽셀 {int((skel > 0).sum())}")
    print(f"  그룹(skeleton component) {total_groups}개")
    print(f"  분기점 {total_branches}개, 끝점 {total_endpoints}개, 연결선 {total_lines}개")

    h, w = img.shape[:2]
    scale = min(1.0, 4096 / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(output_path.suffix, img)[1].tofile(str(output_path))
    print(f"  → {output_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image",  required=True)
    ap.add_argument("--label",  required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--polygons", type=str, default="",
                    help="comma-separated polygon indices (예: '10,11,12,26,27'). 비우면 전체")
    ap.add_argument("--class-ids", type=str, default="",
                    help="포함할 클래스 ID (예: '1' or '1,3'). 비우면 전체")
    ap.add_argument("--branch-radius", type=int, default=10,
                    help="branch 판정 원 반지름 px (기본 10)")
    ap.add_argument("--subdetect", action="store_true",
                    help="모든 그룹 sub-detect (SAM3 서버 필요)")
    ap.add_argument("--subdetect-polys", type=str, default="",
                    help="지정 polygon이 속한 그룹만 sub-detect (예: '2,27')")
    args = ap.parse_args()
    selected = None
    if args.polygons.strip():
        selected = set(int(x.strip()) for x in args.polygons.split(',') if x.strip())
    class_ids = None
    if args.class_ids.strip():
        class_ids = set(int(x.strip()) for x in args.class_ids.split(',') if x.strip())
    subdetect_polys = None
    if args.subdetect_polys.strip():
        subdetect_polys = set(int(x.strip()) for x in args.subdetect_polys.split(',') if x.strip())
    render(Path(args.image), Path(args.label), Path(args.output), selected,
           branch_radius=args.branch_radius, class_ids=class_ids,
           subdetect=args.subdetect, subdetect_polys=subdetect_polys)


if __name__ == "__main__":
    main()
