"""
드론 사선 촬영 이미지에서 라멘형(門자형) 전철주 검출.
기존 픽셀 위상수학(Skeleton) 방식을 공간 기하학 방식으로 교체.

파이프라인:
  Phase 1: 폴리곤 단순화(approxPolyDP) + 소실점(Vanishing Point) 계산
  Phase 2: 동적 V/H 분류 (소실점 기반 기대 각도)
  Phase 3: 근접성 기반 그룹핑 (H 앵커 → 아래 V 탐색)
  Phase 4: 라멘 구조 판정 + 예외(가림) 처리

사용:
    python tools/detect_raamen.py \
        --image <path> --label <path> --output <path> \
        [--class-ids 1] [--epsilon 4.0] [--v-thresh 20.0]
"""
import argparse
import numpy as np
import cv2
from pathlib import Path


# ── Phase 1: 파싱 + 단순화 + 소실점 ─────────────────────────────────────

def load_polygons(label_path, W, H, class_ids=None):
    """YOLO .txt 라벨 → 픽셀 좌표 float32 폴리곤 리스트."""
    if not label_path.exists():
        raise FileNotFoundError(f"라벨 파일을 찾을 수 없습니다: {label_path}")
    polys = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        if class_ids is not None and int(parts[0]) not in class_ids:
            continue
        coords = list(map(float, parts[1:]))
        pts = np.array(
            [[coords[i] * W, coords[i + 1] * H] for i in range(0, len(coords), 2)],
            dtype=np.float32,
        )
        if len(pts) >= 3:
            polys.append(pts)
    return polys


def smooth_polygon(pts, epsilon):
    """approxPolyDP로 노이즈 경계 → 직선 위주 단순화."""
    approx = cv2.approxPolyDP(pts.astype(np.int32), epsilon, closed=True)
    result = approx.reshape(-1, 2).astype(np.float32)
    return result if len(result) >= 3 else pts.astype(np.float32)


def _minrect(pts):
    """cv2.minAreaRect 래퍼 (int32 변환 포함)."""
    return cv2.minAreaRect(pts.astype(np.int32))


def _long_axis_angle(pts):
    """minAreaRect 장축 각도 (degrees) 반환."""
    (cx, cy), (rw, rh), angle = _minrect(pts)
    return angle if rw >= rh else angle + 90, cx, cy, max(rw, rh), min(rw, rh)


def _radial_cos_sim(pts, img_cx, img_cy):
    """VP 후보 bootstrap용 기존 radial cos_sim."""
    long_angle_deg, cx, cy, long_side, short_side = _long_axis_angle(pts)
    if short_side < 1 or long_side / short_side < 1.3:
        return 0.0
    lx = np.cos(np.radians(long_angle_deg))
    ly = np.sin(np.radians(long_angle_deg))
    rdx, rdy = cx - img_cx, cy - img_cy
    n = (rdx ** 2 + rdy ** 2) ** 0.5
    return abs(lx * rdx / n + ly * rdy / n) if n > 1 else 0.0


def compute_vanishing_point(polys):
    """
    후보 폴리곤 장축 선분들의 최소자승 교점 (소실점) 계산.
    각 장축 선분: 법선 n=(-dy, dx), 방정식 -dy*x + dx*y = -dy*cx + dx*cy
    """
    A_rows, b_vals = [], []
    for pts in polys:
        long_angle_deg, cx, cy, _, _ = _long_axis_angle(pts)
        dx = np.cos(np.radians(long_angle_deg))
        dy = np.sin(np.radians(long_angle_deg))
        A_rows.append([-dy, dx])
        b_vals.append(-dy * cx + dx * cy)
    vp, *_ = np.linalg.lstsq(np.array(A_rows), np.array(b_vals), rcond=None)
    return float(vp[0]), float(vp[1])


# ── Phase 2: 동적 V/H 분류 ──────────────────────────────────────────────

def classify_vh(pts, vp_x, vp_y, v_thresh):
    """
    소실점 기준 V/H 분류.
    Returns: (orient, angle_diff_deg)
      orient = 'V' | 'H' | '?'
    """
    long_angle_deg, cx, cy, long_side, short_side = _long_axis_angle(pts)
    if short_side < 1 or long_side / short_side < 1.3:
        return '?', 90.0
    # 기대 수직 각도: 폴리곤 중심 → 소실점 방향
    exp_angle = np.degrees(np.arctan2(vp_y - cy, vp_x - cx))
    # 장축은 방향 무관 → 0~90° 범위로 정규화
    diff = abs(long_angle_deg - exp_angle) % 180.0
    if diff > 90.0:
        diff = 180.0 - diff
    return ('V' if diff < v_thresh else 'H'), diff


# ── Phase 3: 폴리곤 접촉/교차 기반 그룹핑 ───────────────────────────────

def connectivity_groups(polys, orients, margin=30):
    """
    폴리곤 bbox가 margin px 이내로 닿거나 교차하면 같은 그룹 (H/V 구분 없음).
    Union-Find로 연결된 폴리곤들을 묶은 뒤, 각 그룹 내에서 H/V 목록 분리.
    Returns: list of {'id': int, 'H': [idx,...], 'V': [idx,...]}
    """
    n = len(polys)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # 폴리곤 bbox 미리 계산
    bboxes = []
    for pts in polys:
        bboxes.append((pts[:, 0].min(), pts[:, 1].min(),
                       pts[:, 0].max(), pts[:, 1].max()))

    # bbox가 margin 이내로 닿거나 교차하면 union
    for i in range(n):
        ax0, ay0, ax1, ay1 = bboxes[i]
        for j in range(i + 1, n):
            bx0, by0, bx1, by1 = bboxes[j]
            if (ax1 + margin >= bx0 and bx1 + margin >= ax0 and
                    ay1 + margin >= by0 and by1 + margin >= ay0):
                union(i, j)

    # 연결 컴포넌트 수집
    from collections import defaultdict
    comp = defaultdict(list)
    for i in range(n):
        comp[find(i)].append(i)

    groups = []
    for gid, members in enumerate(comp.values(), 1):
        h_list = sorted(i for i in members if orients[i] == 'H')
        v_list = sorted(i for i in members if orients[i] == 'V')
        groups.append({'id': gid, 'H': h_list, 'V': v_list})

    # 면적 내림차순으로 ID 재부여 (큰 그룹이 G1)
    for g in groups:
        g['area'] = sum(cv2.contourArea(polys[i].astype(np.int32))
                        for i in g['H'] + g['V'])
    groups.sort(key=lambda x: x['area'], reverse=True)
    for gid, g in enumerate(groups, 1):
        g['id'] = gid

    return groups


# ── Phase 4: 라멘 구조 판정 ─────────────────────────────────────────────

def _cluster_polys(indices, polys, margin=60):
    """
    인접 폴리곤을 클러스터링 → 물리적 객체(기둥) 단위 반환.
    Returns: list of [idx, ...] clusters
    """
    if not indices:
        return []
    n = len(indices)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    bboxes = [(polys[i][:, 0].min(), polys[i][:, 1].min(),
               polys[i][:, 0].max(), polys[i][:, 1].max()) for i in indices]
    for a in range(n):
        ax0, ay0, ax1, ay1 = bboxes[a]
        for b in range(a + 1, n):
            bx0, by0, bx1, by1 = bboxes[b]
            if (ax1 + margin >= bx0 and bx1 + margin >= ax0 and
                    ay1 + margin >= by0 and by1 + margin >= ay0):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb

    from collections import defaultdict
    clusters = defaultdict(list)
    for i in range(n):
        clusters[find(i)].append(indices[i])
    return list(clusters.values())


def judge_raamen(group, polys, W, center_ratio=0.2, v_cluster_margin=60):
    """
    라멘 구조 판정.
    - 빔 기준: 그룹 내 가장 큰 H 폴리곤
    - 기둥 수: V 폴리곤을 근접 클러스터링한 클러스터 수
    - 빔 x 범위(±50%) 밖의 V 클러스터는 잡폴리곤으로 무시
    Returns: ('RAAMEN' | 'RAAMEN_OCCLUDED' | 'PARTIAL' | '', n_poles)
    """
    hs, vs = group['H'], group['V']

    # H 없음: 중앙 영역이면 V/H 분류 자체가 신뢰 불가 (기둥·빔 각도 수렴)
    # → 2개 이상의 V 폴리곤이 중앙에 모여 있으면 RAAMEN_CENTER 처리
    if not hs:
        if len(vs) >= 2:
            all_v_pts = np.vstack([polys[i] for i in vs])
            gcx = float(all_v_pts[:, 0].mean())
            if abs(gcx - W / 2) < W * center_ratio:
                return 'RAAMEN_CENTER', len(vs)
        return '', 0

    # 가장 큰 H 폴리곤을 빔 기준으로 사용 (잡 H 폴리곤 무시)
    main_h = max(hs, key=lambda i: cv2.contourArea(polys[i].astype(np.int32)))
    pts_h = polys[main_h]
    hx0 = int(pts_h[:, 0].min()); hx1 = int(pts_h[:, 0].max())
    hcx = (hx0 + hx1) / 2.0
    is_center = abs(hcx - W / 2) < W * center_ratio
    span = hx1 - hx0

    # V 폴리곤 클러스터링 → 기둥 단위
    pole_clusters = _cluster_polys(vs, polys, margin=v_cluster_margin)

    # 기둥은 빔 양 끝단(좌 35% / 우 35%)에만 존재. 중앙부 클러스터는 부속물로 제외.
    x_tol = max(span * 0.5, 50)
    left_zone  = hx0 + span * 0.35   # 좌끝단 경계
    right_zone = hx1 - span * 0.35   # 우끝단 경계
    valid_cxs = []
    for cluster in pole_clusters:
        ccx = float(np.mean([polys[i][:, 0].mean() for i in cluster]))
        in_range = hx0 - x_tol <= ccx <= hx1 + x_tol
        in_end_zone = ccx <= left_zone or ccx >= right_zone
        if in_range and in_end_zone:
            valid_cxs.append(ccx)

    n_poles = len(valid_cxs)

    if n_poles >= 2:
        lcx, rcx = min(valid_cxs), max(valid_cxs)
        if lcx <= hx0 + span * 0.4 and rcx >= hx1 - span * 0.4:
            return 'RAAMEN', n_poles
        return 'PARTIAL', n_poles

    if n_poles == 1:
        return ('RAAMEN_OCCLUDED', 1) if not is_center else ('', 1)

    return '', 0


# ── 시각화 상수 ─────────────────────────────────────────────────────────

_VH_COLOR = {
    'V': (255,  80,   0),   # 주황 (수직 기둥)
    'H': (  0,  80, 255),   # 파란 (수평 빔)
    '?': (140, 140, 140),   # 회색 (미분류)
}
_RAAMEN_COLOR = {
    'RAAMEN':          (  0, 255,   0),   # 초록
    'RAAMEN_CENTER':   (  0, 255, 255),   # 노랑 (중앙 영역, H/V 분류 불신뢰)
    'RAAMEN_OCCLUDED': (  0, 165, 255),   # 주황 (가림/부분 검출)
    'PARTIAL':         (128, 128, 128),   # 회색
}


# ── 메인 렌더링 ─────────────────────────────────────────────────────────

def render(image_path, label_path, output_path, args,
           class_ids=None, epsilon=4.0, v_thresh=20.0,
           vp_min_ar=2.5, vp_min_len=80.0, vp_outer_ratio=0.2):

    buf = np.fromfile(str(image_path), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    H, W = img.shape[:2]

    # ── Phase 1 ──────────────────────────────────────────────────────────
    raw_polys = load_polygons(label_path, W, H, class_ids)
    polys = [smooth_polygon(p, epsilon) for p in raw_polys]
    print(f"  {len(polys)}개 폴리곤 파싱 (epsilon={epsilon})")

    img_cx, img_cy = W / 2.0, H / 2.0

    # VP 후보: 외곽 + 고세장비 + 충분한 길이 + radial cos_sim > 0.5
    vp_cands = []
    for i, pts in enumerate(polys):
        _, cx, cy, long_side, short_side = _long_axis_angle(pts)
        ar = long_side / short_side if short_side > 0 else 0
        dist = ((cx - img_cx) ** 2 + (cy - img_cy) ** 2) ** 0.5
        cos_sim = _radial_cos_sim(pts, img_cx, img_cy)
        if (ar > vp_min_ar
                and long_side > vp_min_len
                and dist > min(W, H) * vp_outer_ratio
                and cos_sim > 0.5):
            vp_cands.append(i)
    print(f"  VP 후보: {len(vp_cands)}개 (indices: {vp_cands})")

    if len(vp_cands) >= 2:
        vp_x, vp_y = compute_vanishing_point([polys[i] for i in vp_cands])
        print(f"  소실점: ({vp_x:.1f}, {vp_y:.1f})")
    else:
        # 후보 부족 → 이미지 중앙 위쪽 먼 지점으로 fallback
        vp_x, vp_y = img_cx, -H * 3.0
        print(f"  VP 후보 부족 → fallback VP ({vp_x:.1f}, {vp_y:.1f})")

    # ── Phase 2 ──────────────────────────────────────────────────────────
    print(f"\n  [V/H 분류] threshold=±{v_thresh}°")
    orients, adiffs = [], []
    for i, pts in enumerate(polys):
        orient, adiff = classify_vh(pts, vp_x, vp_y, v_thresh)
        orients.append(orient)
        adiffs.append(adiff)
        print(f"    poly {i:>2d}: {orient}  diff={adiff:.1f}°")
    print(f"  V:{orients.count('V')}  H:{orients.count('H')}  ?:{orients.count('?')}")

    # ── Phase 3 ──────────────────────────────────────────────────────────
    groups = connectivity_groups(polys, orients, margin=args.margin)
    print(f"\n  [그룹핑] 연결 컴포넌트 {len(groups)}개 (margin={args.margin}px)")
    for g in groups:
        print(f"    G{g['id']}: H={g['H']}  V={g['V']}")

    # ── Phase 4 ──────────────────────────────────────────────────────────
    print(f"\n  [라멘 판정]")
    for g in groups:
        verdict, n_poles = judge_raamen(g, polys, W)
        g['verdict'] = verdict
        g['n_poles'] = n_poles
        pole_str = f"{n_poles}poles" if n_poles else "-"
        print(f"    G{g['id']}: H={g['H']} V={g['V']} → {verdict or '-':18s} ({pole_str})")

    valid_items = [g for g in groups if g['verdict']]
    valid_items.sort(key=lambda x: x['area'], reverse=True)

    print(f"\n  [최종 라멘 객체] {len(valid_items)}개 (면적순)")
    for g in valid_items:
        print(f"    G{g['id']}: H={g['H']} V={g['V']} → {g['verdict']:18s} ({g['n_poles']}poles)  Area={g['area']:,.0f}")

    # 최소 면적 필터링
    if args.min_group_area > 0:
        before = len(valid_items)
        valid_items = [g for g in valid_items if g['area'] >= args.min_group_area]
        print(f"  [필터] 최소 면적 {args.min_group_area} 미만 제거: {before} → {len(valid_items)}개")

    # ── 시각화 ───────────────────────────────────────────────────────────

    # 1. 폴리곤 V/H 색상 반투명 오버레이
    for i, (pts, orient) in enumerate(zip(polys, orients)):
        color = _VH_COLOR[orient]
        pts_i = pts.astype(np.int32)
        ov = img.copy()
        cv2.fillPoly(ov, [pts_i], color)
        cv2.addWeighted(ov, 0.25, img, 0.75, 0, img)
        cv2.polylines(img, [pts_i], True, color, 2)
        cx, cy = int(pts[:, 0].mean()), int(pts[:, 1].mean())
        lbl = f"{i}{orient}"
        cv2.putText(img, lbl, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 4)
        cv2.putText(img, lbl, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    # 2. VP 후보 폴리곤에 청록 테두리
    for i in vp_cands:
        cv2.polylines(img, [polys[i].astype(np.int32)], True, (0, 220, 220), 3)

    # 3. 소실점 표시 (이미지 내부: 원, 외부: 방향 화살표)
    vp_ix, vp_iy = int(vp_x), int(vp_y)
    if 0 <= vp_ix < W and 0 <= vp_iy < H:
        cv2.circle(img, (vp_ix, vp_iy), 20, (0, 255, 255), 3)
        cv2.putText(img, "VP", (vp_ix + 25, vp_iy),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    else:
        ax_s, ay_s = W // 2, H // 5
        dx, dy = vp_x - img_cx, vp_y - img_cy
        n = (dx ** 2 + dy ** 2) ** 0.5
        if n > 0:
            ax_e = max(0, min(W - 1, int(ax_s + dx / n * 80)))
            ay_e = max(0, min(H - 1, int(ay_s + dy / n * 80)))
            cv2.arrowedLine(img, (ax_s, ay_s), (ax_e, ay_e), (0, 255, 255), 3)
        cv2.putText(img, f"VP({vp_ix},{vp_iy})", (ax_s + 5, ay_s - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # 4. 라멘 그룹 강조: 바운딩 박스 + 그룹 ID + 판정 라벨
    for g in valid_items:
        verdict = g['verdict']
        color = _RAAMEN_COLOR[verdict]
        all_idx = g['H'] + g['V']
        all_pts = np.vstack([polys[i] for i in all_idx]).astype(np.int32)
        x0 = all_pts[:, 0].min() - 15; y0 = all_pts[:, 1].min() - 15
        x1 = all_pts[:, 0].max() + 15; y1 = all_pts[:, 1].max() + 15
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 4)
        lbl = f"G{g['id']} {verdict} ({g['n_poles']}poles)"
        cv2.putText(img, lbl, (x0, y0 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 6)
        cv2.putText(img, lbl, (x0, y0 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)

    # 이미지 저장
    scale = min(1.0, 4096 / max(H, W))
    if scale < 1.0:
        img = cv2.resize(img, (int(W * scale), int(H * scale)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(output_path.suffix, img)[1].tofile(str(output_path))
    print(f"\n  → {output_path}")

    # JSON 결과 저장 (output_path와 같은 위치, .json 확장자)
    import json
    result = {
        "image": str(image_path),
        "label": str(label_path),
        "vanishing_point": {"x": round(vp_x, 1), "y": round(vp_y, 1)},
        "raamen_groups": [
            {
                "group_id": g['id'],
                "verdict": g['verdict'],
                "n_poles": g['n_poles'],
                "area": round(g['area'], 1),
                "H_polygons": g['H'],
                "V_polygons": g['V'],
            }
            for g in valid_items
        ],
    }
    json_path = output_path.with_suffix(".json")
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  → {json_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image",      required=True)
    ap.add_argument("--label",      required=True)
    ap.add_argument("--output",     required=True)
    ap.add_argument("--class-ids",  default="1",
                    help="포함할 클래스 ID, 콤마 구분 (기본: '1')")
    ap.add_argument("--epsilon",    type=float, default=4.0,
                    help="approxPolyDP epsilon (기본 4.0px)")
    ap.add_argument("--v-thresh",   type=float, default=20.0,
                    help="V/H 분류 각도 임계값 degrees (기본 20°)")
    ap.add_argument("--margin",         type=int,   default=30,
                    help="폴리곤 접촉 판정 margin px (기본 30)")
    ap.add_argument("--min-group-area", type=float, default=0,
                    help="라멘 그룹의 최소 면적 합계 (기본 0)")
    args = ap.parse_args()

    class_ids = ({int(x) for x in args.class_ids.split(',') if x.strip()}
                 if args.class_ids else None)
    render(Path(args.image), Path(args.label), Path(args.output), args,
           class_ids=class_ids, epsilon=args.epsilon, v_thresh=args.v_thresh)


if __name__ == "__main__":
    main()
