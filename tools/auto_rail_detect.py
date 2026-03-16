"""
auto_rail_detect.py
===================
항공 이미지에서 레일 라인을 자동 검출하여 Rhino용 DXF로 저장.
수동 라벨링 없이 이미지 1장에서 바로 실행.

사용법:
    python tools/auto_rail_detect.py <image.png> [output.dxf]

예시:
    python tools/auto_rail_detect.py "경부선_2-2구간_미션33-34(5cm).png"
    python tools/auto_rail_detect.py "경부선.png" output/rail.dxf

원리:
    이미지 → 엣지 검출 → Hough 직선 검출 → 방향별 클러스터링 → DXF 폴리라인
"""

import sys
import cv2
import numpy as np
from pathlib import Path


# ─── 설정 (조정 가능) ─────────────────────────────────────────────────────────
CANNY_LOW       = 30     # Canny 엣지 낮은 임계값 (낮을수록 엣지 많이 검출)
CANNY_HIGH      = 80     # Canny 엣지 높은 임계값
HOUGH_THRESHOLD = 100    # Hough 투표 임계값 (높을수록 긴 선만 검출)
HOUGH_MIN_LEN   = 200    # 최소 선 길이 (픽셀)
HOUGH_MAX_GAP   = 30     # 선 간격 허용 (픽셀, 클수록 끊긴 선도 연결)
ANGLE_TOLERANCE = 5      # 같은 방향으로 볼 각도 범위 (도)
CLUSTER_DIST    = 20     # 같은 레일로 볼 선 간격 (픽셀)
MIN_TOTAL_LEN   = 500    # 최종 레일 최소 총 길이 (픽셀, 짧은 잡선 제거)
# ──────────────────────────────────────────────────────────────────────────────


def detect_edges(img_gray):
    """CLAHE 대비 강화 → Gaussian 블러 → Canny 엣지"""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img_gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    edges = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH, apertureSize=3)
    return edges


def detect_lines(edges):
    """확률적 Hough 변환으로 직선 검출"""
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LEN,
        maxLineGap=HOUGH_MAX_GAP,
    )
    if lines is None:
        return []
    return [tuple(l[0]) for l in lines]  # [(x1,y1,x2,y2), ...]


def line_angle(x1, y1, x2, y2):
    """선의 각도 (0~180도)"""
    angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
    return angle


def line_length(x1, y1, x2, y2):
    return np.hypot(x2 - x1, y2 - y1)


def line_midpoint(x1, y1, x2, y2):
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def perpendicular_dist(x1, y1, x2, y2, px, py):
    """점 (px,py)에서 선 (x1,y1)-(x2,y2)까지 수직거리"""
    dx, dy = x2 - x1, y2 - y1
    length = np.hypot(dx, dy)
    if length == 0:
        return np.hypot(px - x1, py - y1)
    return abs(dy * px - dx * py + x2 * y1 - y2 * x1) / length


def cluster_lines(lines):
    """비슷한 방향 + 가까운 위치의 선들을 같은 레일로 묶기"""
    if not lines:
        return []

    # 각도별 그룹 먼저 분리
    angle_groups = {}
    for line in lines:
        x1, y1, x2, y2 = line
        ang = round(line_angle(x1, y1, x2, y2) / ANGLE_TOLERANCE) * ANGLE_TOLERANCE
        angle_groups.setdefault(ang, []).append(line)

    clusters = []
    for ang, grp in angle_groups.items():
        # 같은 방향 그룹 내에서 거리 기반 클러스터링
        used = [False] * len(grp)
        for i, line_i in enumerate(grp):
            if used[i]:
                continue
            cluster = [line_i]
            used[i] = True
            mx_i, my_i = line_midpoint(*line_i)
            for j, line_j in enumerate(grp):
                if used[j]:
                    continue
                mx_j, my_j = line_midpoint(*line_j)
                dist = perpendicular_dist(*line_i, mx_j, my_j)
                if dist < CLUSTER_DIST:
                    cluster.append(line_j)
                    used[j] = True
            clusters.append(cluster)

    return clusters


def merge_cluster_to_polyline(cluster):
    """클러스터의 선들을 하나의 정렬된 폴리라인으로 합치기"""
    # 모든 끝점 수집
    all_pts = []
    for x1, y1, x2, y2 in cluster:
        all_pts.append((x1, y1))
        all_pts.append((x2, y2))

    if not all_pts:
        return []

    # 주성분 방향으로 정렬
    pts_arr = np.array(all_pts, dtype=float)
    mean = pts_arr.mean(axis=0)
    centered = pts_arr - mean
    cov = np.cov(centered.T)
    if cov.ndim < 2:
        direction = np.array([1.0, 0.0])
    else:
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        direction = eigenvectors[:, np.argmax(eigenvalues)]

    # 방향으로 투영하여 정렬
    projections = centered.dot(direction)
    sorted_idx = np.argsort(projections)
    sorted_pts = pts_arr[sorted_idx]

    # 중복 제거 (가까운 점 합치기)
    merged = [sorted_pts[0]]
    for pt in sorted_pts[1:]:
        if np.hypot(pt[0] - merged[-1][0], pt[1] - merged[-1][1]) > 5:
            merged.append(pt)

    return merged


def total_polyline_length(polyline):
    if len(polyline) < 2:
        return 0
    total = 0
    for i in range(len(polyline) - 1):
        total += np.hypot(
            polyline[i+1][0] - polyline[i][0],
            polyline[i+1][1] - polyline[i][1]
        )
    return total


def save_debug_image(img, all_polylines, output_path):
    """검출 결과를 이미지에 시각화 (확인용)"""
    vis = img.copy() if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255),
              (255, 0, 255), (255, 255, 0)]
    for i, poly in enumerate(all_polylines):
        color = colors[i % len(colors)]
        pts = np.array([[int(p[0]), int(p[1])] for p in poly])
        cv2.polylines(vis, [pts], False, color, 2)
        # 번호 표시
        cx, cy = int(poly[len(poly)//2][0]), int(poly[len(poly)//2][1])
        cv2.putText(vis, str(i+1), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX,
                    1.5, color, 3)
    cv2.imwrite(str(output_path), vis)
    print(f"  시각화 저장: {output_path}")


def process(image_path: str, dxf_path: str):
    import ezdxf

    print(f"[입력] {image_path}")
    # 한글 경로 대응: numpy로 읽어서 cv2 디코딩
    import numpy as np_io
    with open(image_path, "rb") as f:
        data = f.read()
    img_arr = np_io.frombuffer(data, dtype=np_io.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    if img is None:
        print(f"오류: 이미지를 열 수 없습니다: {image_path}")
        sys.exit(1)

    H, W = img.shape[:2]
    print(f"[이미지] {W} x {H} px")

    # 전처리: 작은 이미지로 축소 (속도)
    scale = 1.0
    if W > 4000:
        scale = 4000 / W
        small = cv2.resize(img, (int(W * scale), int(H * scale)))
        print(f"  축소: {int(W*scale)} x {int(H*scale)} (scale={scale:.2f})")
    else:
        small = img.copy()

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    # 엣지 검출
    print("[1단계] 엣지 검출...")
    edges = detect_edges(gray)
    edge_count = int(edges.sum() / 255)
    print(f"  엣지 픽셀: {edge_count:,}")

    # Hough 직선 검출
    print("[2단계] Hough 직선 검출...")
    lines = detect_lines(edges)
    print(f"  검출된 선: {len(lines)}개")

    if not lines:
        print("선을 검출하지 못했습니다. HOUGH_THRESHOLD를 낮춰보세요.")
        sys.exit(1)

    # 클러스터링
    print("[3단계] 레일 클러스터링...")
    clusters = cluster_lines(lines)
    print(f"  클러스터: {len(clusters)}개")

    # 폴리라인 변환 + 길이 필터
    polylines = []
    for cluster in clusters:
        poly = merge_cluster_to_polyline(cluster)
        length = total_polyline_length(poly)
        if length >= MIN_TOTAL_LEN / scale:
            # 원본 해상도로 역변환
            poly_orig = [(p[0] / scale, p[1] / scale) for p in poly]
            polylines.append((poly_orig, length / scale))

    # 길이순 정렬
    polylines.sort(key=lambda x: -x[1])
    print(f"  유효 레일 라인: {len(polylines)}개")

    if not polylines:
        print("유효한 레일을 찾지 못했습니다. MIN_TOTAL_LEN을 낮춰보세요.")
        sys.exit(1)

    for i, (poly, length) in enumerate(polylines):
        print(f"    레일 {i+1}: {length:.0f}px ({length*0.05:.1f}m)")

    # DXF 저장
    print("[4단계] DXF 저장...")
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()
    doc.layers.add("RAIL_AUTO", color=1)      # 빨강 — 자동검출 레일
    doc.layers.add("RAIL_AUTO_LABEL", color=2) # 노랑 — 번호

    for i, (poly, length) in enumerate(polylines):
        # Y축 반전 (이미지→DXF)
        dxf_pts = [(float(p[0]), float(-p[1])) for p in poly]
        msp.add_lwpolyline(dxf_pts, dxfattribs={"layer": "RAIL_AUTO"})

        # 중간점에 번호 텍스트
        mid = poly[len(poly) // 2]
        msp.add_text(
            f"Rail_{i+1}",
            dxfattribs={
                "layer": "RAIL_AUTO_LABEL",
                "height": 20,
                "insert": (float(mid[0]), float(-mid[1])),
            }
        )

    Path(dxf_path).parent.mkdir(parents=True, exist_ok=True)
    doc.saveas(dxf_path)
    print(f"[완료] DXF 저장: {dxf_path}")
    print(f"  레이어: RAIL_AUTO(빨강) = {len(polylines)}개 레일 중심선")

    # 디버그 이미지 저장
    debug_path = Path(dxf_path).parent / (Path(dxf_path).stem + "_debug.jpg")
    all_polys = [p for p, _ in polylines]
    save_debug_image(small, [([(x*scale, y*scale) for x, y in p]) for p in all_polys],
                     debug_path)

    print(f"\nRhino 사용법:")
    print(f"  1. File -> Import -> {Path(dxf_path).name}")
    print(f"  2. RAIL_AUTO 레이어 선택")
    print(f"  3. Sweep1 명령 -> 레일 단면 선택 -> 3D 레일 완성")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python tools/auto_rail_detect.py <image.png> [output.dxf]")
        sys.exit(1)

    img_path = sys.argv[1]
    dxf_path = sys.argv[2] if len(sys.argv) >= 3 else str(
        Path(img_path).with_suffix(".dxf")
    )
    process(img_path, dxf_path)
