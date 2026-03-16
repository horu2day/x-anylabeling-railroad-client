"""
철도 평면 선형 피팅 모듈
스켈레톤 좌표점 → 직선/원곡선/완화곡선 분류 및 피팅 → 매끄러운 폴리라인

선형 구성요소:
  1. 직선 (Straight)        : y = ax + b
  2. 원곡선 (Circular)      : R = 상수, Taubin 원 피팅
  3. 완화곡선 (Transition)  : y = x³/(6RL), 3차포물선 (KR 설계기준)
"""

import numpy as np
from scipy.signal import savgol_filter
from scipy.linalg import eig as scipy_eig


# ── 1. 곡률 계산 ──────────────────────────────────────────────────────────

def compute_curvature(pts, smooth_window=15):
    """
    길이 가중 방향벡터를 이용한 곡률 계산

    Parameters
    ----------
    pts : (N, 2) array  세계 좌표 (미터)
    smooth_window : int  Savitzky-Golay 스무딩 윈도우 (홀수)

    Returns
    -------
    curvatures : (N,)  곡률 κ [1/m]
    arc_lengths : (N,)  각 점의 누적 호 길이 [m]
    angles : (N,)  방향각 [rad], unwrap 처리됨
    """
    pts = np.asarray(pts, dtype=float)
    n = len(pts)
    if n < 3:
        return np.zeros(n), np.zeros(n), np.zeros(n)

    # 세그먼트 벡터 및 길이
    segs = np.diff(pts, axis=0)          # (n-1, 2)
    lens = np.linalg.norm(segs, axis=1)  # (n-1,)
    lens = np.maximum(lens, 1e-12)

    # 단위 방향벡터
    dirs = segs / lens[:, np.newaxis]

    # 가중 방향벡터: 인접 세그먼트 길이를 가중치로 평균
    directions = np.empty((n, 2))
    directions[0] = dirs[0]
    directions[-1] = dirs[-1]
    for i in range(1, n - 1):
        w1, w2 = lens[i - 1], lens[i]
        directions[i] = (w1 * dirs[i - 1] + w2 * dirs[i]) / (w1 + w2)

    # 정규화
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions /= np.maximum(norms, 1e-12)

    # 방향각 (연속성 보장)
    angles = np.arctan2(directions[:, 1], directions[:, 0])
    angles = np.unwrap(angles)

    # 누적 호 길이
    arc_lengths = np.zeros(n)
    arc_lengths[1:] = np.cumsum(lens)

    # 곡률 κ = dθ/ds (중앙차분)
    curvatures = np.zeros(n)
    for i in range(1, n - 1):
        dtheta = angles[i + 1] - angles[i - 1]
        ds = arc_lengths[i + 1] - arc_lengths[i - 1]
        if ds > 1e-10:
            curvatures[i] = dtheta / ds
    curvatures[0] = curvatures[1]
    curvatures[-1] = curvatures[-2]

    # Savitzky-Golay 스무딩 (노이즈 억제)
    w = smooth_window
    if n >= w and w >= 5:
        w = w if w % 2 == 1 else w - 1
        curvatures = savgol_filter(curvatures, w, 3)

    return curvatures, arc_lengths, angles


# ── 2. 구간 분류 ──────────────────────────────────────────────────────────

def segment_by_curvature(curvatures, arc_lengths,
                          kappa_straight=0.001,   # 1/1000m → R > 1000m 직선 처리
                          cv_thresh=0.35,          # 변동계수(표준편차/평균) 임계값
                          min_seg_len=3.0):        # 최소 구간 길이 [m]
    """
    곡률 패턴으로 구간 분류

    Returns
    -------
    segments : list of (start_idx, end_idx, type_str)
        type_str ∈ {'straight', 'circular', 'transition'}
    """
    n = len(curvatures)
    kappa_abs = np.abs(curvatures)
    win = max(5, n // 15)

    def classify_window(k_arr):
        km = k_arr.mean()
        ks = k_arr.std()
        if km < kappa_straight:
            return 'straight'
        cv = ks / (km + 1e-12)
        if cv < cv_thresh:
            return 'circular'
        return 'transition'

    # 각 점의 로컬 타입
    local_type = []
    for i in range(n):
        lo = max(0, i - win // 2)
        hi = min(n, i + win // 2 + 1)
        local_type.append(classify_window(kappa_abs[lo:hi]))

    # 연속된 같은 타입 구간 합치기
    segments = []
    cur_type = local_type[0]
    cur_start = 0
    for i in range(1, n):
        if local_type[i] != cur_type:
            segments.append((cur_start, i - 1, cur_type))
            cur_type = local_type[i]
            cur_start = i
    segments.append((cur_start, n - 1, cur_type))

    # 너무 짧은 구간은 인접 구간에 병합
    merged = True
    while merged and len(segments) > 1:
        merged = False
        new_segs = []
        i = 0
        while i < len(segments):
            s, e, t = segments[i]
            seg_len = arc_lengths[e] - arc_lengths[s]
            if seg_len < min_seg_len and len(segments) > 1:
                if i == 0 and i + 1 < len(segments):
                    ns, ne, nt = segments[i + 1]
                    segments[i + 1] = (s, ne, nt)
                    i += 1
                    merged = True
                elif new_segs:
                    new_segs[-1] = (new_segs[-1][0], e, new_segs[-1][2])
                    merged = True
                else:
                    new_segs.append((s, e, t))
            else:
                new_segs.append((s, e, t))
            i += 1
        if merged:
            segments = new_segs

    return segments


# ── 3. 개별 구간 피팅 ────────────────────────────────────────────────────

def fit_straight(pts, spacing=0.5):
    """PCA 직선 피팅 → 균등 샘플 점 목록"""
    pts = np.asarray(pts, dtype=float)
    center = pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(pts - center)
    direction = Vt[0]
    t = (pts - center) @ direction
    t_min, t_max = t.min(), t.max()
    n_pts = max(2, int((t_max - t_min) / spacing) + 1)
    ts = np.linspace(t_min, t_max, n_pts)
    return [(center + ti * direction).tolist() for ti in ts]


def taubin_circle_fit(pts):
    """Taubin 방법으로 안정적 원 피팅 → (cx, cy, R) 또는 None"""
    pts = np.asarray(pts, dtype=float)
    mean = pts.mean(axis=0)
    x = pts[:, 0] - mean[0]
    y = pts[:, 1] - mean[1]
    z = x**2 + y**2

    Mxx = (x * x).mean()
    Myy = (y * y).mean()
    Mxy = (x * y).mean()
    Mxz = (x * z).mean()
    Myz = (y * z).mean()
    Mzz = (z * z).mean()

    Mz = Mxx + Myy
    Cov_xy = Mxx * Myy - Mxy**2
    A3 = 4 * Mz
    A2 = -3 * Mz**2 - Mzz
    A1 = Mzz * Mz + 4 * Cov_xy * Mz - Mxz**2 - Myz**2 - Mz**3
    A0 = Mxz**2 * Myy + Myz**2 * Mxx - Mzz * Cov_xy - 2 * Mxz * Myz * Mxy + Mz**2 * Cov_xy

    try:
        roots = np.roots([A3, A2, A1, A0])
        real_roots = roots[np.isreal(roots)].real
        if len(real_roots) == 0:
            return None
        xn = max(real_roots)
        yn = (xn**2 - Mz) * xn + A0 / A3
        det = xn**2 - xn * Mz + Cov_xy
        if abs(det) < 1e-12:
            return None
        xcen = (Mxz * (Myy - xn) - Myz * Mxy) / (2 * det) + mean[0]
        ycen = (Myz * (Mxx - xn) - Mxz * Mxy) / (2 * det) + mean[1]
        R = np.sqrt(np.mean((pts[:, 0] - xcen)**2 + (pts[:, 1] - ycen)**2))
        return xcen, ycen, R
    except Exception:
        return None


def fit_circular(pts, spacing=0.5):
    """원곡선 피팅 → 호 위의 균등 샘플 점 목록"""
    result = taubin_circle_fit(pts)
    if result is None:
        return fit_straight(pts, spacing)

    cx, cy, R = result
    pts = np.asarray(pts, dtype=float)
    angs = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    angs = np.unwrap(angs)
    a_start, a_end = angs[0], angs[-1]

    arc_len = abs(a_end - a_start) * R
    n_pts = max(2, int(arc_len / spacing) + 1)
    thetas = np.linspace(a_start, a_end, n_pts)
    return [(cx + R * np.cos(t), cy + R * np.sin(t)) for t in thetas]


def fit_transition_cubic(pts, R_adj, spacing=0.5):
    """
    3차포물선 완화곡선 피팅  y = x³ / (6·R·L)
    로컬 좌표계(시작점, 시작방향 기준)에서 피팅 후 세계 좌표로 역변환

    Parameters
    ----------
    pts   : (N, 2) 세계 좌표
    R_adj : 인접 원곡선 반경 [m] (없으면 곡률 역수로 추정)
    """
    pts = np.asarray(pts, dtype=float)
    origin = pts[0].copy()

    # 로컬 좌표계 축 (PCA 주방향)
    _, _, Vt = np.linalg.svd(pts - origin)
    axis_x = Vt[0]
    axis_y = np.array([-axis_x[1], axis_x[0]])

    local = pts - origin
    lx = local @ axis_x
    ly = local @ axis_y

    L = lx.max()
    if L < 1e-5 or R_adj < 1e-5:
        return pts.tolist()

    # 균등 샘플
    n_pts = max(2, int(L / spacing) + 1)
    xs = np.linspace(0, L, n_pts)
    ys = xs**3 / (6.0 * R_adj * L)

    return [(origin + xi * axis_x + yi * axis_y).tolist()
            for xi, yi in zip(xs, ys)]


# ── 4. 전체 피팅 파이프라인 ───────────────────────────────────────────────

def fit_alignment(pts, spacing=0.5):
    """
    메인 함수: 세계 좌표 점 목록 → 선형 피팅 → 매끄러운 폴리라인

    Parameters
    ----------
    pts     : list of (x, y)  EPSG:5186 세계 좌표 [m]
    spacing : float           출력 점 간격 [m]

    Returns
    -------
    smooth_pts : list of (x, y)  피팅된 매끄러운 폴리라인 점 목록
    seg_info   : list of dict    각 구간 정보 (디버깅용)
    """
    pts = np.asarray(pts, dtype=float)
    n = len(pts)

    if n < 5:
        return pts.tolist(), []

    # --- 곡률 계산 ---
    sw = min(21, n if n % 2 == 1 else n - 1)
    sw = max(5, sw)
    curvatures, arc_lengths, angles = compute_curvature(pts, sw)

    # --- 구간 분류 ---
    segments = segment_by_curvature(curvatures, arc_lengths)

    # --- 인접 원곡선 R 사전 계산 (완화곡선에서 사용) ---
    seg_R = {}
    for idx, (s, e, t) in enumerate(segments):
        if t == 'circular':
            km = np.abs(curvatures[s:e+1]).mean()
            seg_R[idx] = 1.0 / km if km > 1e-6 else 1000.0

    smooth_pts = []
    seg_info = []

    for idx, (s, e, seg_type) in enumerate(segments):
        seg_pts = pts[s:e+1]

        if len(seg_pts) < 2:
            smooth_pts.extend(seg_pts.tolist())
            continue

        km = np.abs(curvatures[s:e+1]).mean()
        R_est = 1.0 / km if km > 1e-6 else 9999.0

        if seg_type == 'straight':
            fitted = fit_straight(seg_pts, spacing)
        elif seg_type == 'circular':
            fitted = fit_circular(seg_pts, spacing)
        else:  # transition
            # 인접 원곡선 R 탐색
            R_use = R_est
            for di in [1, -1]:
                ni = idx + di
                if ni in seg_R:
                    R_use = seg_R[ni]
                    break
            fitted = fit_transition_cubic(seg_pts, R_use, spacing)

        seg_len = arc_lengths[e] - arc_lengths[s]
        seg_info.append({
            'type': seg_type,
            'start_idx': s,
            'end_idx': e,
            'length_m': round(seg_len, 2),
            'R_m': round(R_est, 1),
            'n_pts_in': len(seg_pts),
            'n_pts_out': len(fitted),
        })

        # 이음새 중복 제거
        if smooth_pts and fitted:
            smooth_pts.extend(fitted[1:])
        else:
            smooth_pts.extend(fitted)

    return smooth_pts, seg_info


# ── 5. 간단 테스트 ────────────────────────────────────────────────────────

if __name__ == '__main__':
    # 직선 + 원곡선 + 완화곡선 합성 테스트
    t = np.linspace(0, 2*np.pi, 300)
    # 직선 구간
    straight = [(float(i)*0.5, 0.0) for i in range(40)]
    # 원곡선 구간 R=300m
    R = 300.0
    theta = np.linspace(0, np.pi/4, 60)
    cx, cy = straight[-1][0], straight[-1][1] + R
    circ = [(cx + R*np.sin(th), cy - R*np.cos(th)) for th in theta]
    pts_test = straight + circ[1:]

    smooth, info = fit_alignment(pts_test, spacing=1.0)
    print(f"입력: {len(pts_test)}점 → 출력: {len(smooth)}점")
    for s in info:
        print(f"  [{s['type']:10s}] L={s['length_m']:.1f}m  R={s['R_m']:.0f}m  "
              f"점: {s['n_pts_in']}→{s['n_pts_out']}")
