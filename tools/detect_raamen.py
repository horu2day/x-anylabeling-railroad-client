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
        return []
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


# ── Phase 3: 근접성 기반 그룹핑 ─────────────────────────────────────────

def proximity_groups(polys, orients, v_search_scale=3.0, x_margin_scale=0.4):
    """
    H 앵커 기반 그룹핑.
    H bbox 아래 탐색 영역(v_search_scale * H_height)에서 V 폴리곤 수집.
    Returns: list of {'H': idx, 'V': [idx, ...]}
    """
    h_idx = [i for i, o in enumerate(orients) if o == 'H']
    v_idx = [i for i, o in enumerate(orients) if o == 'V']

    groups = []
    for hi in h_idx:
        pts_h = polys[hi]
        hx0 = int(pts_h[:, 0].min()); hx1 = int(pts_h[:, 0].max())
        hy0 = int(pts_h[:, 1].min()); hy1 = int(pts_h[:, 1].max())
        h_w, h_h = hx1 - hx0, hy1 - hy0
        x_margin = max(h_w * x_margin_scale, 20)
        search_y_end = hy1 + max(h_h * v_search_scale, 60)

        found_v = []
        for vi in v_idx:
            vcx = float(polys[vi][:, 0].mean())
            vcy = float(polys[vi][:, 1].mean())
            if (hx0 - x_margin <= vcx <= hx1 + x_margin
                    and hy0 <= vcy <= search_y_end):
                found_v.append(vi)
        groups.append({'H': hi, 'V': found_v})
    return groups


# ── Phase 4: 라멘 구조 판정 ─────────────────────────────────────────────

def judge_raamen(group, polys, W, center_ratio=0.2):
    """
    라멘 구조 판정.
    Returns: 'RAAMEN' | 'RAAMEN_CENTER' | 'RAAMEN_OCCLUDED' | 'PARTIAL' | ''
    """
    hi, vs = group['H'], group['V']
    pts_h = polys[hi]
    hx0 = int(pts_h[:, 0].min()); hx1 = int(pts_h[:, 0].max())
    hcx = (hx0 + hx1) / 2.0
    is_center = abs(hcx - W / 2) < W * center_ratio

    if len(vs) >= 2:
        # V 폴리곤을 x 중심 기준으로 정렬
        vs_by_x = sorted(vs, key=lambda i: float(polys[i][:, 0].mean()))
        lcx = float(polys[vs_by_x[0]][:, 0].mean())
        rcx = float(polys[vs_by_x[-1]][:, 0].mean())
        span = hx1 - hx0
        # 좌측 V는 H 좌측 40% 내, 우측 V는 H 우측 40% 내에 위치
        if lcx <= hx0 + span * 0.4 and rcx >= hx1 - span * 0.4:
            return 'RAAMEN'
        return 'PARTIAL'

    if len(vs) == 1:
        # 중앙 영역: 기둥/빔 중첩 → 정상 라멘 간주
        # 외곽 영역: 가림 현상
        return 'RAAMEN_CENTER' if is_center else 'RAAMEN_OCCLUDED'

    return ''


# ── 시각화 상수 ─────────────────────────────────────────────────────────

_VH_COLOR = {
    'V': (255,  80,   0),   # 주황 (수직 기둥)
    'H': (  0,  80, 255),   # 파란 (수평 빔)
    '?': (140, 140, 140),   # 회색 (미분류)
}
_RAAMEN_COLOR = {
    'RAAMEN':          (  0, 255,   0),   # 초록
    'RAAMEN_CENTER':   (  0, 255, 255),   # 노랑
    'RAAMEN_OCCLUDED': (  0, 165, 255),   # 주황
    'PARTIAL':         (128, 128, 128),   # 회색
}


# ── 메인 렌더링 ─────────────────────────────────────────────────────────

def render(image_path, label_path, output_path,
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
    groups = proximity_groups(polys, orients)
    print(f"\n  [그룹핑] H-앵커 그룹 {len(groups)}개")
    for g in groups:
        print(f"    H={g['H']}  V={g['V']}")

    # ── Phase 4 ──────────────────────────────────────────────────────────
    print(f"\n  [라멘 판정]")
    verdicts = []
    for g in groups:
        v = judge_raamen(g, polys, W)
        verdicts.append(v)
        print(f"    H={g['H']} V={g['V']} → {v or '-'}")

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

    # 4. 라멘 그룹 강조: 바운딩 박스 + 판정 라벨
    for g, verdict in zip(groups, verdicts):
        if not verdict:
            continue
        color = _RAAMEN_COLOR[verdict]
        all_idx = [g['H']] + g['V']
        all_pts = np.vstack([polys[i] for i in all_idx]).astype(np.int32)
        x0 = all_pts[:, 0].min() - 15; y0 = all_pts[:, 1].min() - 15
        x1 = all_pts[:, 0].max() + 15; y1 = all_pts[:, 1].max() + 15
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 4)
        cv2.putText(img, verdict, (x0, y0 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 7)
        cv2.putText(img, verdict, (x0, y0 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

    # 저장
    scale = min(1.0, 4096 / max(H, W))
    if scale < 1.0:
        img = cv2.resize(img, (int(W * scale), int(H * scale)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(output_path.suffix, img)[1].tofile(str(output_path))
    print(f"\n  → {output_path}")


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
    args = ap.parse_args()

    class_ids = ({int(x) for x in args.class_ids.split(',') if x.strip()}
                 if args.class_ids else None)
    render(Path(args.image), Path(args.label), Path(args.output),
           class_ids=class_ids, epsilon=args.epsilon, v_thresh=args.v_thresh)


if __name__ == "__main__":
    main()
