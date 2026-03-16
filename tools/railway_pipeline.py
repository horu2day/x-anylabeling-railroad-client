"""
railway_pipeline.py
===================
정사영상(GeoTIFF/PNG)에서 철도 시설물을 자동 검출하여
실좌표(UTM/WGS84) 기반 DXF + GeoJSON 출력.

이미지에 실제로 보이는 것을 그대로 검출 (표준 규격 기반 아님).

사용법:
    python tools/railway_pipeline.py <image.tif|png> [output_dir]

예시:
    python tools/railway_pipeline.py 경부선.tif output/
    python tools/railway_pipeline.py 경부선.png output/   # PNG는 GSD=0.05m 가정

출력:
    output/railway_rails.dxf        - 레일 중심선
    output/railway_objects.dxf      - 전체 시설물 (레이어별)
    output/railway_objects.geojson  - GIS 활용용
    output/railway_debug.jpg        - 검출 결과 시각화
"""

import sys
import json
import math
import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict


# ─── 검출 파라미터 ─────────────────────────────────────────────────────────────
RAIL = dict(
    canny_low=30, canny_high=80,
    hough_threshold=400, min_len=500, max_gap=30,
    cluster_dist=14, min_total_len=3000,
)
SLEEPER = dict(
    search_width=60,        # 레일 양옆 검색 폭 (px)
    min_area=30,            # 최소 면적
    max_area=2000,          # 최대 면적
    aspect_min=2.0,         # 최소 가로세로비 (침목은 길쭉)
)
POLE = dict(
    search_margin=200,      # 레일 옆 검색 범위 (px)
    min_radius=3,           # 최소 반지름 (px)
    max_radius=25,          # 최대 반지름 (px)
    min_dist=40,            # 검출 최소 간격 (px)
    param1=50, param2=25,   # Hough Circle 파라미터
)
CBOX = dict(
    search_margin=300,      # 레일 옆 검색 범위 (px)
    min_area=100,
    max_area=8000,
    aspect_min=1.2,         # 직사각형 비율
    aspect_max=6.0,
    min_solidity=0.75,      # 컨투어 충실도 (직사각형에 가까울수록 1)
)
GABOR = dict(
    lambdas=[5, 7, 10],    # 침목 간격 범위 (px, 축소 이미지 기준)
                            # 5cm GSD, 8000px 폭(scale≈0.54): 600mm/50mm*0.54 ≈ 6.5px
    sigma=3.0,              # Gabor 커널 폭
    gamma=0.5,              # 종횡비 (1=원형, 0.5=타원)
    n_angles=12,            # 방향 수 (0°~165°, 15° 간격)
    thresh_pct=60,          # 응답 임계값 백분위 (높을수록 엄격)
    min_area=4000,          # 최소 연결성분 면적 (작은 노이즈 제거)
)
SCALE_FACTOR = 8000         # 침목 패턴 감지를 위해 높은 해상도 유지
# ──────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# 좌표 변환
# ══════════════════════════════════════════════════════════════════════════════

class GeoTransform:
    """픽셀좌표 ↔ 실좌표 변환."""

    def __init__(self, image_path: str):
        self.crs = None
        self.affine = None
        self.origin = (0.0, 0.0)
        self.pixel_size = (0.05, 0.05)  # 기본 5cm GSD
        self._load(image_path)

    def _load(self, path: str):
        if path.lower().endswith(('.tif', '.tiff')):
            try:
                import rasterio
                with rasterio.open(path) as r:
                    self.crs = str(r.crs)
                    t = r.transform
                    self.affine = t
                    self.pixel_size = (abs(t.a), abs(t.e))
                    self.origin = (t.c, t.f)
                    print(f"  CRS: {self.crs}")
                    print(f"  픽셀크기: {self.pixel_size[0]:.4f}m x {self.pixel_size[1]:.4f}m")
                    print(f"  원점: ({self.origin[0]:.2f}, {self.origin[1]:.2f})")
            except Exception as e:
                print(f"  [경고] rasterio 실패: {e} → 픽셀좌표 사용")
        else:
            # PNG: world file 탐색
            wld = Path(path).with_suffix('.pgw')
            if not wld.exists():
                wld = Path(path).with_suffix('.wld')
            if wld.exists():
                vals = [float(l.strip()) for l in wld.read_text().splitlines() if l.strip()]
                if len(vals) >= 6:
                    self.pixel_size = (abs(vals[0]), abs(vals[3]))
                    self.origin = (vals[4], vals[5])
                    print(f"  World file 적용: origin={self.origin}")
            else:
                print(f"  World file 없음 → 픽셀좌표 사용 (GSD=0.05m)")

    def px_to_world(self, px: float, py: float):
        """픽셀 (col, row) → 실좌표 (x, y)."""
        if self.affine:
            from rasterio.transform import xy as rio_xy
            try:
                x, y = self.affine * (px, py)
                return float(x), float(y)
            except Exception:
                pass
        x = self.origin[0] + px * self.pixel_size[0]
        y = self.origin[1] - py * self.pixel_size[1]
        return x, y

    def scale_coords(self, pts, scale: float):
        """축소된 이미지의 픽셀좌표를 원본 픽셀좌표로 변환 후 실좌표 출력."""
        return [self.px_to_world(p[0] / scale, p[1] / scale) for p in pts]

    def scale_point(self, px: float, py: float, scale: float):
        return self.px_to_world(px / scale, py / scale)


# ══════════════════════════════════════════════════════════════════════════════
# 이미지 로드 (한글 경로 대응)
# ══════════════════════════════════════════════════════════════════════════════

def load_image(path: str):
    """TIF/PNG 모두 BGR numpy array로 반환."""
    if path.lower().endswith(('.tif', '.tiff')):
        try:
            import rasterio
            with rasterio.open(path) as r:
                if r.count >= 3:
                    arr = np.dstack([r.read(i) for i in [1, 2, 3]])
                    # uint16 → uint8 변환
                    if arr.dtype != np.uint8:
                        arr = (arr / arr.max() * 255).astype(np.uint8)
                    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                else:
                    band = r.read(1)
                    if band.dtype != np.uint8:
                        band = (band / band.max() * 255).astype(np.uint8)
                    return cv2.cvtColor(band, cv2.COLOR_GRAY2BGR)
        except Exception as e:
            print(f"  rasterio 읽기 실패: {e}, PIL 시도...")
    # PNG 또는 fallback
    with open(path, 'rb') as f:
        data = f.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


# ══════════════════════════════════════════════════════════════════════════════
# 자갈도상 마스크
# ══════════════════════════════════════════════════════════════════════════════

def extract_ballast_mask(img):
    """자갈도상(ballast) 구역 마스크 추출.
    항공 정사영상에서 자갈도상은 밝고(V>70) 채도 낮음(S<60) — HSV 임계값 후
    큰 연결성분만 유지하여 선로 띠 영역만 남김.
    """
    H, W = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 낮은 채도 + 중간-높은 밝기 = 자갈/회색  (채도 45 이하로 더 엄격하게)
    mask = cv2.inRange(hsv,
                       np.array([0,   0,  80], np.uint8),
                       np.array([180, 45, 210], np.uint8))

    # 작은 노이즈 제거
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open, iterations=2)

    # 팽창: 자갈 구역 연결 + 레일 엣지까지 포함 (2회로 줄여 과팽창 방지)
    k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.dilate(mask, k_dil, iterations=2)

    # 연결성분 중 이미지 면적 0.5% 이상만 유지 (도로·건물 등 소규모 제거)
    min_area = max(H * W * 0.005, 5000)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    result = np.zeros_like(mask)
    for lbl in range(1, n):
        if stats[lbl, cv2.CC_STAT_AREA] >= min_area:
            result[labels == lbl] = 255

    kept = int((result > 0).sum())
    total = H * W
    print(f"  자갈도상 마스크: {kept:,}px ({kept/total*100:.1f}%)")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Gabor 침목 패턴 마스크 + 레일 검출 (통합)
# ══════════════════════════════════════════════════════════════════════════════

def detect_rails_gabor(img):
    """
    [1단계] Gabor 필터 뱅크로 침목 줄무늬 패턴 영역 추출
            → 자갈도상 색상이 아닌 침목 주기 텍스처로 선로 구역 판별
            → 도로는 이 패턴 없으므로 자동 제거

    [2단계] 침목 마스크 안에서 Hough로 개별 레일 중심선 검출

    Returns: (rail_polylines, gabor_mask)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape

    # ── Gabor 필터 뱅크 적용 ──────────────────────────────────
    response_max = np.zeros((H, W), dtype=np.float32)
    n_ang = GABOR['n_angles']
    for i in range(n_ang):
        theta = i * np.pi / n_ang          # 0 ~ π (모든 방향)
        for lam in GABOR['lambdas']:
            ksize = max(int(GABOR['sigma'] * 6) | 1, 7)   # 홀수 보장
            kernel = cv2.getGaborKernel(
                (ksize, ksize),
                GABOR['sigma'], theta, lam,
                GABOR['gamma'], 0, cv2.CV_32F
            )
            resp = np.abs(cv2.filter2D(gray.astype(np.float32), cv2.CV_32F, kernel))
            response_max = np.maximum(response_max, resp)

    # ── 응답 정규화 + 임계값 ──────────────────────────────────
    resp_u8 = cv2.normalize(response_max, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    nz = resp_u8[resp_u8 > 0]
    thresh_val = int(np.percentile(nz, GABOR['thresh_pct'])) if len(nz) else 128
    _, binary = cv2.threshold(resp_u8, thresh_val, 255, cv2.THRESH_BINARY)

    # ── 모폴로지 정리 ─────────────────────────────────────────
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k5, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  k3, iterations=1)

    # ── 큰 연결성분만 유지 (소규모 노이즈 제거) ───────────────
    n_cc, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    gabor_mask = np.zeros_like(binary)
    for lbl in range(1, n_cc):
        if stats[lbl, cv2.CC_STAT_AREA] >= GABOR['min_area']:
            gabor_mask[labels == lbl] = 255

    kept = int((gabor_mask > 0).sum())
    print(f"  Gabor 마스크: {kept:,}px ({kept / (H * W) * 100:.1f}%)")

    # ── Gabor 마스크 내에서 Hough 레일 중심선 검출 ───────────
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    edges = cv2.Canny(blurred, RAIL['canny_low'], RAIL['canny_high'])
    edges = cv2.bitwise_and(edges, edges, mask=gabor_mask)
    print(f"  마스크 내 엣지: {int(edges.sum() // 255):,}px")

    lines_raw = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=RAIL['hough_threshold'],
        minLineLength=RAIL['min_len'],
        maxLineGap=RAIL['max_gap'],
    )
    if lines_raw is None:
        print("  Hough 검출 없음")
        return [], gabor_mask

    lines = [tuple(l[0]) for l in lines_raw]
    print(f"  Hough 검출: {len(lines)}개")

    # ── 클러스터링 → 폴리라인 ────────────────────────────────
    # 곡선 구간 지원: 각도 기준 아닌 미드포인트 거리 기준으로 클러스터링
    # 같은 레일의 곡선 세그먼트들은 인접 미드포인트를 가짐 → 체인으로 연결
    ATOL = RAIL['cluster_dist']

    # 각도별 그룹화 후 수직거리 기반 클러스터링
    # 분기기 구간: 강제 연결 없이 세그먼트 단위로 유지 (분기기는 수동 연결)
    def angle(x1, y1, x2, y2):
        return np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180

    def perp_dist(x1, y1, x2, y2, px, py):
        dx, dy = x2 - x1, y2 - y1
        L = math.hypot(dx, dy)
        if L == 0:
            return math.hypot(px - x1, py - y1)
        return abs(dy * px - dx * py + x2 * y1 - y2 * x1) / L

    angle_groups = defaultdict(list)
    for l in lines:
        key = round(angle(*l) / 5) * 5
        angle_groups[key].append(l)

    clusters = []
    for grp in angle_groups.values():
        used = [False] * len(grp)
        for i, li in enumerate(grp):
            if used[i]:
                continue
            cl = [li]
            used[i] = True
            for j, lj in enumerate(grp):
                if used[j]:
                    continue
                mxj = ((lj[0]+lj[2])/2, (lj[1]+lj[3])/2)
                if perp_dist(*li, *mxj) < ATOL:
                    cl.append(lj)
                    used[j] = True
            clusters.append(cl)

    def merge(cluster):
        pts = []
        for x1, y1, x2, y2 in cluster:
            pts += [(x1, y1), (x2, y2)]
        arr = np.array(pts, float)
        mean = arr.mean(0)
        cov = np.cov((arr - mean).T)
        if cov.ndim < 2:
            direction = np.array([1.0, 0.0])
        else:
            ev, evec = np.linalg.eig(cov)
            direction = evec[:, np.argmax(ev)]
        proj = (arr - mean).dot(direction)
        sorted_pts = arr[np.argsort(proj)]
        merged = [sorted_pts[0]]
        for p in sorted_pts[1:]:
            if math.hypot(p[0] - merged[-1][0], p[1] - merged[-1][1]) > 5:
                merged.append(p)
        return merged

    def poly_len(poly):
        return sum(
            math.hypot(poly[i + 1][0] - poly[i][0], poly[i + 1][1] - poly[i][1])
            for i in range(len(poly) - 1)
        )

    result = []
    for cl in clusters:
        poly = merge(cl)
        if poly_len(poly) >= RAIL['min_total_len']:
            result.append(poly)
    result.sort(key=lambda p: -poly_len(p))
    return result, gabor_mask


# ══════════════════════════════════════════════════════════════════════════════
# 레일 검출 (Hough — fallback용, 주 검출은 detect_rails_gabor 사용)
# ══════════════════════════════════════════════════════════════════════════════

def detect_rails(img, ballast_mask=None):
    """Hough Line으로 레일 중심선 폴리라인 반환."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5,5), 0)
    edges = cv2.Canny(blurred, RAIL['canny_low'], RAIL['canny_high'])

    # 자갈도상 마스크 적용 — 선로 구역 외 엣지 제거
    if ballast_mask is not None:
        edges = cv2.bitwise_and(edges, edges, mask=ballast_mask)
        print(f"  마스크 적용 후 엣지: {int(edges.sum()//255):,}px")

    lines_raw = cv2.HoughLinesP(edges, 1, np.pi/180,
                                 threshold=RAIL['hough_threshold'],
                                 minLineLength=RAIL['min_len'],
                                 maxLineGap=RAIL['max_gap'])
    if lines_raw is None:
        return []

    lines = [tuple(l[0]) for l in lines_raw]

    def angle(x1,y1,x2,y2):
        return np.degrees(np.arctan2(y2-y1, x2-x1)) % 180

    def line_len(x1,y1,x2,y2):
        return math.hypot(x2-x1, y2-y1)

    def midpoint(x1,y1,x2,y2):
        return ((x1+x2)/2, (y1+y2)/2)

    def perp_dist(x1,y1,x2,y2,px,py):
        dx,dy = x2-x1, y2-y1
        L = math.hypot(dx,dy)
        if L == 0: return math.hypot(px-x1, py-y1)
        return abs(dy*px - dx*py + x2*y1 - y2*x1) / L

    # 방향 필터 불필요 — 자갈도상 마스크가 공간 제약 역할을 하므로
    # 커브 구간도 포함하려면 모든 방향 허용
    print(f"  Hough 검출: {len(lines)}개")

    # 각도별 그룹화 후 거리 클러스터링
    ATOL = RAIL['cluster_dist']
    angle_groups = defaultdict(list)
    for l in lines:
        key = round(angle(*l) / 5) * 5
        angle_groups[key].append(l)

    clusters = []
    for grp in angle_groups.values():
        used = [False]*len(grp)
        for i, li in enumerate(grp):
            if used[i]: continue
            cl = [li]; used[i] = True
            mx, my = midpoint(*li)
            for j, lj in enumerate(grp):
                if used[j]: continue
                if perp_dist(*li, *midpoint(*lj)) < ATOL:
                    cl.append(lj); used[j] = True
            clusters.append(cl)

    def merge(cluster):
        pts = []
        for x1,y1,x2,y2 in cluster:
            pts += [(x1,y1),(x2,y2)]
        arr = np.array(pts, float)
        mean = arr.mean(0)
        cov  = np.cov((arr - mean).T)
        if cov.ndim < 2: direction = np.array([1.0,0.0])
        else:
            ev, evec = np.linalg.eig(cov)
            direction = evec[:, np.argmax(ev)]
        proj = (arr - mean).dot(direction)
        idx  = np.argsort(proj)
        sorted_pts = arr[idx]
        merged = [sorted_pts[0]]
        for p in sorted_pts[1:]:
            if math.hypot(p[0]-merged[-1][0], p[1]-merged[-1][1]) > 5:
                merged.append(p)
        return merged

    def poly_len(poly):
        return sum(math.hypot(poly[i+1][0]-poly[i][0], poly[i+1][1]-poly[i][1])
                   for i in range(len(poly)-1))

    result = []
    for cl in clusters:
        poly = merge(cl)
        if poly_len(poly) >= RAIL['min_total_len']:
            result.append(poly)
    result.sort(key=lambda p: -poly_len(p))
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 침목 검출
# ══════════════════════════════════════════════════════════════════════════════

def detect_sleepers(img, rail_polys, gabor_mask=None):
    """레일 영역 주변에서 침목(가로 직사각형) 검출."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 레일 위치로 마스크 생성
    mask = np.zeros(gray.shape, np.uint8)
    sw = SLEEPER['search_width']
    for poly in rail_polys:
        pts = np.array([[int(p[0]), int(p[1])] for p in poly])
        cv2.polylines(mask, [pts], False, 255, sw * 2)
    masked = cv2.bitwise_and(thresh, thresh, mask=mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(masked, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    sleepers = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (SLEEPER['min_area'] <= area <= SLEEPER['max_area']):
            continue
        rect = cv2.minAreaRect(cnt)
        (cx, cy), (w, h), angle = rect
        if w == 0 or h == 0: continue
        aspect = max(w, h) / min(w, h)
        if aspect < SLEEPER['aspect_min']:
            continue
        sleepers.append({
            'center': (float(cx), float(cy)),
            'size': (float(w), float(h)),
            'angle': float(angle),
            'area': float(area),
        })
    return sleepers


# ══════════════════════════════════════════════════════════════════════════════
# 전철주 검출
# ══════════════════════════════════════════════════════════════════════════════

def detect_poles(img, rail_polys):
    """레일 주변에서 원형/점형 전철주 검출 (Hough Circle + Blob)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9,9), 2)

    # 레일 근처 마스크
    mask = np.zeros(gray.shape, np.uint8)
    for poly in rail_polys:
        pts = np.array([[int(p[0]), int(p[1])] for p in poly])
        cv2.polylines(mask, [pts], False, 255, POLE['search_margin'] * 2)

    masked = cv2.bitwise_and(blurred, blurred, mask=mask)

    circles = cv2.HoughCircles(
        masked, cv2.HOUGH_GRADIENT, dp=1,
        minDist=POLE['min_dist'],
        param1=POLE['param1'],
        param2=POLE['param2'],
        minRadius=POLE['min_radius'],
        maxRadius=POLE['max_radius'],
    )

    poles = []
    if circles is not None:
        for (cx, cy, r) in circles[0]:
            poles.append({
                'center': (float(cx), float(cy)),
                'radius': float(r),
            })
    return poles


# ══════════════════════════════════════════════════════════════════════════════
# 컨트롤박스 검출
# ══════════════════════════════════════════════════════════════════════════════

def detect_control_boxes(img, rail_polys):
    """레일 근처 직사각형 객체 검출."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blurred, 30, 90)

    # 레일 근처 마스크
    mask = np.zeros(gray.shape, np.uint8)
    for poly in rail_polys:
        pts = np.array([[int(p[0]), int(p[1])] for p in poly])
        cv2.polylines(mask, [pts], False, 255, CBOX['search_margin'] * 2)
    masked_edges = cv2.bitwise_and(edges, edges, mask=mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    closed = cv2.morphologyEx(masked_edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (CBOX['min_area'] <= area <= CBOX['max_area']):
            continue
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0: continue
        solidity = area / hull_area
        if solidity < CBOX['min_solidity']:
            continue
        rect = cv2.minAreaRect(cnt)
        (cx, cy), (w, h), angle = rect
        if w == 0 or h == 0: continue
        aspect = max(w, h) / min(w, h)
        if not (CBOX['aspect_min'] <= aspect <= CBOX['aspect_max']):
            continue
        boxes.append({
            'center': (float(cx), float(cy)),
            'size': (float(w), float(h)),
            'angle': float(angle),
            'area': float(area),
        })
    return boxes


# ══════════════════════════════════════════════════════════════════════════════
# DXF 출력
# ══════════════════════════════════════════════════════════════════════════════

def save_dxf(output_path, rails_world, sleepers_world, poles_world, boxes_world):
    import ezdxf
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()

    # 레이어 정의
    doc.layers.add("RAIL",         color=1)   # 빨강
    doc.layers.add("SLEEPER",      color=3)   # 초록
    doc.layers.add("POLE",         color=5)   # 파랑
    doc.layers.add("CONTROL_BOX",  color=6)   # 마젠타

    # 레일 중심선
    for poly in rails_world:
        if len(poly) >= 2:
            msp.add_lwpolyline(poly, dxfattribs={"layer": "RAIL"})

    # 침목
    for s in sleepers_world:
        cx, cy = s['world_center']
        w, h = s['size']
        ang = s['angle']
        scale = s.get('scale', 1.0)
        # 회전된 직사각형
        box_pts = cv2.boxPoints(((cx, -cy), (w/scale, h/scale), ang))
        msp.add_lwpolyline(
            [(float(p[0]), float(p[1])) for p in box_pts] + [(float(box_pts[0][0]), float(box_pts[0][1]))],
            dxfattribs={"layer": "SLEEPER"}
        )

    # 전철주 (원)
    for p in poles_world:
        cx, cy = p['world_center']
        r = p.get('world_radius', 1.0)
        msp.add_circle((cx, -cy), r, dxfattribs={"layer": "POLE"})

    # 컨트롤박스
    for b in boxes_world:
        cx, cy = b['world_center']
        w, h = b['size']
        ang = b['angle']
        scale = b.get('scale', 1.0)
        box_pts = cv2.boxPoints(((cx, -cy), (w/scale, h/scale), ang))
        msp.add_lwpolyline(
            [(float(p[0]), float(p[1])) for p in box_pts] + [(float(box_pts[0][0]), float(box_pts[0][1]))],
            dxfattribs={"layer": "CONTROL_BOX"}
        )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    doc.saveas(output_path)
    print(f"  DXF: {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
# GeoJSON 출력
# ══════════════════════════════════════════════════════════════════════════════

def save_geojson(output_path, rails_world, sleepers_world, poles_world, boxes_world):
    features = []

    for i, poly in enumerate(rails_world):
        features.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": [[x, y] for x, y in poly]},
            "properties": {"type": "rail", "id": i}
        })
    for i, s in enumerate(sleepers_world):
        cx, cy = s['world_center']
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [cx, cy]},
            "properties": {"type": "sleeper", "id": i, "angle": s['angle']}
        })
    for i, p in enumerate(poles_world):
        cx, cy = p['world_center']
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [cx, cy]},
            "properties": {"type": "pole", "id": i, "radius_m": p.get('world_radius', 0)}
        })
    for i, b in enumerate(boxes_world):
        cx, cy = b['world_center']
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [cx, cy]},
            "properties": {"type": "control_box", "id": i,
                           "width_m": b.get('world_w', 0), "height_m": b.get('world_h', 0)}
        })

    geojson = {"type": "FeatureCollection", "features": features}
    if not output_path.endswith('.geojson'):
        output_path += '.geojson'
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(geojson, f, ensure_ascii=False, indent=2)
    print(f"  GeoJSON: {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 시각화
# ══════════════════════════════════════════════════════════════════════════════

def save_debug(img, rails, sleepers, poles, boxes, output_path, ballast_mask=None):
    vis = img.copy()
    # 자갈도상 마스크 반투명 오버레이 (노란색)
    if ballast_mask is not None:
        overlay = vis.copy()
        overlay[ballast_mask > 0] = (0, 200, 200)
        cv2.addWeighted(overlay, 0.2, vis, 0.8, 0, vis)
    # 레일
    for poly in rails:
        pts = np.array([[int(p[0]), int(p[1])] for p in poly])
        cv2.polylines(vis, [pts], False, (0,0,255), 2)
    # 침목
    for s in sleepers:
        cx, cy = int(s['center'][0]), int(s['center'][1])
        rect = ((cx,cy), (s['size'][0], s['size'][1]), s['angle'])
        box = cv2.boxPoints(rect).astype(int)
        cv2.drawContours(vis, [box], 0, (0,255,0), 1)
    # 전철주
    for p in poles:
        cx, cy, r = int(p['center'][0]), int(p['center'][1]), int(p['radius'])
        cv2.circle(vis, (cx, cy), r, (255,0,0), 2)
    # 컨트롤박스
    for b in boxes:
        cx, cy = int(b['center'][0]), int(b['center'][1])
        rect = ((cx,cy), (b['size'][0], b['size'][1]), b['angle'])
        box = cv2.boxPoints(rect).astype(int)
        cv2.drawContours(vis, [box], 0, (255,0,255), 2)

    # 범례
    legend = [("RAIL",        (0,0,255)),
              ("SLEEPER",     (0,255,0)),
              ("POLE",        (255,0,0)),
              ("CONTROL_BOX", (255,0,255))]
    for i, (name, color) in enumerate(legend):
        cv2.rectangle(vis, (10, 10+i*28), (24, 24+i*28), color, -1)
        cv2.putText(vis, name, (30, 22+i*28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    with open(str(output_path), 'wb') as f:
        _, buf = cv2.imencode('.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
        f.write(buf.tobytes())
    print(f"  시각화: {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 메인 파이프라인
# ══════════════════════════════════════════════════════════════════════════════

def process(image_path: str, output_dir: str):
    print(f"\n{'='*60}")
    print(f"입력: {image_path}")
    print(f"{'='*60}")

    # 좌표 변환기 초기화
    print("\n[좌표계]")
    geo = GeoTransform(image_path)

    # 이미지 로드
    print("\n[이미지 로드]")
    img = load_image(image_path)
    if img is None:
        print("오류: 이미지를 열 수 없습니다.")
        sys.exit(1)
    H, W = img.shape[:2]
    print(f"  크기: {W} x {H} px")

    # 처리용 축소
    scale = 1.0
    if W > SCALE_FACTOR:
        scale = SCALE_FACTOR / W
        small = cv2.resize(img, (int(W*scale), int(H*scale)))
        print(f"  축소: {int(W*scale)} x {int(H*scale)} (x{scale:.2f})")
    else:
        small = img.copy()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem

    # ── Gabor 침목 패턴 검출 + 레일 중심선 ───────────────────
    print("\n[1] Gabor 침목 패턴 → 레일 검출...")
    rails_px, gabor_mask = detect_rails_gabor(small)
    print(f"  검출: {len(rails_px)}개")

    # Gabor 마스크 저장 (확인용)
    mask_path = out / f"{stem}_gabor_mask.jpg"
    with open(str(mask_path), 'wb') as _f:
        _, _buf = cv2.imencode('.jpg', gabor_mask)
        _f.write(_buf.tobytes())
    print(f"  Gabor 마스크: {mask_path}")

    # 레일 실좌표 변환
    rails_world = []
    for poly in rails_px:
        world_pts = geo.scale_coords(poly, scale)
        rails_world.append(world_pts)

    # ── 침목 검출 ──────────────────────────────────────────
    print("\n[2] 침목 검출...")
    sleepers_px = detect_sleepers(small, rails_px, gabor_mask)
    print(f"  검출: {len(sleepers_px)}개")

    sleepers_world = []
    for s in sleepers_px:
        cx, cy = s['center']
        wx, wy = geo.scale_point(cx, cy, scale)
        ps = geo.pixel_size[0] / scale
        sleepers_world.append({
            **s,
            'world_center': (wx, wy),
            'scale': scale,
        })

    # ── 전철주 검출 ────────────────────────────────────────
    print("\n[3] 전철주 검출...")
    poles_px = detect_poles(small, rails_px)
    print(f"  검출: {len(poles_px)}개")

    poles_world = []
    for p in poles_px:
        cx, cy = p['center']
        wx, wy = geo.scale_point(cx, cy, scale)
        r_m = p['radius'] * geo.pixel_size[0] / scale
        poles_world.append({
            **p,
            'world_center': (wx, wy),
            'world_radius': r_m,
        })

    # ── 컨트롤박스 검출 ────────────────────────────────────
    print("\n[4] 컨트롤박스 검출...")
    boxes_px = detect_control_boxes(small, rails_px)
    print(f"  검출: {len(boxes_px)}개")

    boxes_world = []
    ps = geo.pixel_size[0]
    for b in boxes_px:
        cx, cy = b['center']
        wx, wy = geo.scale_point(cx, cy, scale)
        w_m = b['size'][0] * ps / scale
        h_m = b['size'][1] * ps / scale
        boxes_world.append({
            **b,
            'world_center': (wx, wy),
            'world_w': w_m,
            'world_h': h_m,
            'scale': scale,
        })

    # ── 결과 요약 ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"검출 결과 요약")
    print(f"  레일:       {len(rails_world):>5}개")
    print(f"  침목:       {len(sleepers_world):>5}개")
    print(f"  전철주:     {len(poles_world):>5}개")
    print(f"  컨트롤박스: {len(boxes_world):>5}개")
    print(f"{'='*60}")

    # ── 출력 ──────────────────────────────────────────────
    print("\n[출력]")
    save_dxf(
        str(out / f"{stem}_railway.dxf"),
        rails_world, sleepers_world, poles_world, boxes_world
    )
    save_geojson(
        str(out / f"{stem}_railway.geojson"),
        rails_world, sleepers_world, poles_world, boxes_world
    )
    save_debug(
        small, rails_px, sleepers_px, poles_px, boxes_px,
        out / f"{stem}_railway_debug.jpg",
        ballast_mask=gabor_mask
    )

    print(f"\n완료.")
    print(f"  DXF 레이어: RAIL(빨강) / SLEEPER(초록) / POLE(파랑) / CONTROL_BOX(마젠타)")
    print(f"  Rhino: Import DXF → 레이어별 3D 모델 교체")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python tools/railway_pipeline.py <image.tif|png> [output_dir]")
        sys.exit(1)

    img_path   = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) >= 3 else "output"
    process(img_path, output_dir)
