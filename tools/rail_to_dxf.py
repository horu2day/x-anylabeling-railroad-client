"""
rail_to_dxf.py
==============
X-AnyLabeling JSON 어노테이션에서 레일 중심선을 추출하여 Rhino용 DXF로 저장.

사용법:
    python tools/rail_to_dxf.py <annotation.json> [output.dxf]

예시:
    python tools/rail_to_dxf.py images/rail.json
    python tools/rail_to_dxf.py images/rail.json output/rail_centerline.dxf

라벨 이름 (X-AnyLabeling에서 Finish 후 입력한 이름):
    rail, railway_track, track, AUTOLABEL_OBJECT 자동 인식
    다른 이름이면 스크립트 하단 TARGET_LABELS 수정

Rhino에서 사용:
    1. DXF Import
    2. RAIL_CENTERLINE 레이어 선택
    3. Sweep2 또는 Rail Sweep으로 레일 단면 적용
"""

import json
import sys
import numpy as np
import cv2
from pathlib import Path
from skimage.morphology import skeletonize


# ─── 설정 ─────────────────────────────────────────────────────────────────────
TARGET_LABELS = [
    "rail",
    "railline",
    "railway_track",
    "track",
    "레일",
    "철로",
    "AUTOLABEL_OBJECT",
]
# line/linestrip 타입은 스켈레톤 불필요 — 직접 DXF 출력
LINE_SHAPE_TYPES  = {"line", "linestrip", "lines"}
SMOOTH_WINDOW   = 15   # 중심선 스무딩 강도 (클수록 부드러움, 0=비활성)
DOWNSAMPLE_STEP = 8    # Rhino 폴리라인 포인트 간격 (클수록 포인트 수 감소)
DILATION_ITER   = 3    # 마스크 팽창 반복 (얇은 레일 마스크 연결 보완)
# ──────────────────────────────────────────────────────────────────────────────


def polygon_to_mask(points, h, w):
    mask = np.zeros((h, w), dtype=np.uint8)
    pts  = np.array([[int(p[0]), int(p[1])] for p in points], dtype=np.int32)
    cv2.fillPoly(mask, [pts], 1)
    return mask


def extract_skeleton(mask):
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=DILATION_ITER)
    return skeletonize(dilated > 0).astype(np.uint8)


def order_skeleton(skeleton):
    """스켈레톤 픽셀을 끝점에서 시작해 순서대로 연결."""
    ys, xs = np.where(skeleton > 0)
    if len(xs) == 0:
        return []

    pt_set = set(zip(xs.tolist(), ys.tolist()))

    def neighbors(pt):
        x, y = pt
        return [(x+dx, y+dy)
                for dx in (-1,0,1) for dy in (-1,0,1)
                if not (dx==0 and dy==0) and (x+dx, y+dy) in pt_set]

    # 끝점(이웃 1개) 찾기 → 없으면 임의 시작
    endpoints = [p for p in pt_set if len(neighbors(p)) == 1]
    start = endpoints[0] if endpoints else next(iter(pt_set))

    ordered, visited = [start], {start}
    current = start

    while True:
        nbs = [n for n in neighbors(current) if n not in visited]
        if not nbs:
            break
        # 가장 직선에 가까운 방향 우선 선택
        if len(ordered) >= 2:
            dx = current[0] - ordered[-2][0]
            dy = current[1] - ordered[-2][1]
            nbs.sort(key=lambda n: -((n[0]-current[0])*dx + (n[1]-current[1])*dy))
        current = nbs[0]
        visited.add(current)
        ordered.append(current)

    return ordered


def smooth_polyline(points, window):
    if window < 3 or len(points) < window:
        return points
    pts = np.array(points, dtype=float)
    half = window // 2
    out  = pts.copy()
    for i in range(half, len(pts) - half):
        out[i] = pts[i-half:i+half+1].mean(axis=0)
    return out.tolist()


def downsample(points, step):
    if step <= 1 or len(points) <= step:
        return points
    sampled = points[::step]
    if list(sampled[-1]) != list(points[-1]):
        sampled = list(sampled) + [points[-1]]
    return sampled


def to_dxf_coords(points, flip_y=True):
    """이미지 좌표(Y↓) → DXF 좌표(Y↑)"""
    if flip_y:
        return [(float(p[0]), float(-p[1])) for p in points]
    return [(float(p[0]), float(p[1]))  for p in points]


def process(json_path: str, dxf_path: str):
    import ezdxf

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    H = data.get("imageHeight", 1000)
    W = data.get("imageWidth",  1000)
    print(f"[이미지] {W} × {H} px")

    shapes = data.get("shapes", [])
    targets = [s for s in shapes if s.get("label") in TARGET_LABELS]

    if not targets:
        print("⚠  TARGET_LABELS 일치 없음 → 모든 shape 사용")
        targets = shapes

    type_counts = {}
    for s in targets:
        t = s.get("shape_type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1
    print(f"[처리] {len(targets)}개: {type_counts}")

    doc = ezdxf.new("R2010")
    msp = doc.modelspace()
    doc.layers.add("RAIL_CENTERLINE", color=1)   # 빨강 — Sweep 경로
    doc.layers.add("RAIL_POLYGON",    color=3)   # 초록 — 원본 마스크 윤곽

    for i, shape in enumerate(targets):
        label      = shape.get("label", f"shape_{i}")
        pts        = shape.get("points", [])
        shape_type = shape.get("shape_type", "polygon")
        print(f"\n  [{i+1}] label={label!r}  type={shape_type!r}  pts={len(pts)}")

        if len(pts) < 2:
            print("      → 포인트 부족, 건너뜀")
            continue

        # 중복 끝점 제거 (Polygon 도구로 그린 선: 마지막 점이 앞 점과 거의 동일)
        if len(pts) >= 3:
            dedup = [pts[0]]
            for p in pts[1:]:
                if abs(p[0]-dedup[-1][0]) > 2 or abs(p[1]-dedup[-1][1]) > 2:
                    dedup.append(p)
            if len(dedup) < len(pts):
                print(f"      중복점 제거: {len(pts)}pt → {len(dedup)}pt")
                pts = dedup

        # ── LINE 타입: 스켈레톤 없이 직접 DXF 출력 ──────
        if shape_type in LINE_SHAPE_TYPES or len(pts) == 2:
            cl_dxf = to_dxf_coords(pts)
            msp.add_lwpolyline(cl_dxf, dxfattribs={"layer": "RAIL_CENTERLINE"})
            print(f"      → 라인 직접 출력 ({len(pts)}pt)")
            continue

        # ── POLYGON 타입: 마스크 → 스켈레톤 → 중심선 ────
        if len(pts) < 3:
            print("      → 폴리곤 포인트 부족, 건너뜀")
            continue

        poly_dxf = to_dxf_coords(pts)
        poly_dxf.append(poly_dxf[0])
        msp.add_lwpolyline(poly_dxf, dxfattribs={"layer": "RAIL_POLYGON"})

        mask  = polygon_to_mask(pts, H, W)
        area  = int(mask.sum())
        print(f"      마스크 면적: {area} px²")
        if area < 20:
            print("      → 면적 너무 작음, 건너뜀")
            continue

        skel   = extract_skeleton(mask)
        skel_n = int(skel.sum())
        print(f"      스켈레톤 픽셀: {skel_n}")
        if skel_n < 2:
            print("      → 스켈레톤 생성 실패, 건너뜀")
            continue

        ordered   = order_skeleton(skel)
        smoothed  = smooth_polyline(ordered, SMOOTH_WINDOW)
        final_pts = downsample(smoothed, DOWNSAMPLE_STEP)
        print(f"      중심선 포인트: {len(final_pts)}")
        if len(final_pts) < 2:
            continue

        cl_dxf = to_dxf_coords(final_pts)
        msp.add_lwpolyline(cl_dxf, dxfattribs={"layer": "RAIL_CENTERLINE"})

    Path(dxf_path).parent.mkdir(parents=True, exist_ok=True)
    doc.saveas(dxf_path)
    print(f"\n[완료] DXF 저장: {dxf_path}")
    print("   레이어: RAIL_CENTERLINE(빨강) = Sweep 경로")
    print("           RAIL_POLYGON(초록)    = 원본 마스크")
    print("\nRhino 사용법:")
    print("  1. File → Import → DXF 선택")
    print("  2. RAIL_CENTERLINE 레이어 선택")
    print("  3. Rail 단면 커브 그리기 (레일 단면 약 60x60mm)")
    print("  4. Sweep1 또는 Sweep2 명령으로 단면 × 중심선 → 3D 레일 생성")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python tools/rail_to_dxf.py <annotation.json> [output.dxf]")
        sys.exit(1)

    json_file = sys.argv[1]
    dxf_file  = sys.argv[2] if len(sys.argv) >= 3 else str(Path(json_file).with_suffix(".dxf"))

    process(json_file, dxf_file)
