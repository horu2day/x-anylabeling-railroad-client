"""
2cm GSD 드론 TIF 전체에서 레일 중심선 추출 → DXF 저장
"""
import numpy as np
import rasterio
from rasterio.windows import Window
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
from ultralytics import YOLO
from PIL import Image
import ezdxf
import os, sys
from collections import defaultdict

# ── 설정 ──────────────────────────────────────────
SRC_TIF  = "drone_2cm/22)조치원(STA.127+570~131+300).tif"
MODEL_PT = "runs/segment/output/yolo_train_2cm/rail_seg_v2/weights/best.pt"
OUT_DXF  = "output/rail_centerline_2cm.dxf"
TILE     = 1024
OVERLAP  = 128
STEP     = TILE - OVERLAP
CONF     = 0.3
BLACK_THR = 0.4   # 검은 픽셀 비율 임계값

# ── 모델 로드 ──────────────────────────────────────
print("모델 로드 중...")
model = YOLO(MODEL_PT)

# ── DXF 초기화 ────────────────────────────────────
doc = ezdxf.new()
msp = doc.modelspace()
doc.layers.add("RAIL_CENTERLINE", color=3)

# ── TIF 열기 ──────────────────────────────────────
src = rasterio.open(SRC_TIF)
W, H = src.width, src.height
transform = src.transform
print(f"이미지 크기: {W} x {H}")

def pixel_to_world(px, py):
    """픽셀 좌표 → 세계 좌표 (EPSG:5186)"""
    wx = transform.c + px * transform.a
    wy = transform.f + py * transform.e
    return wx, wy

def trace_skeleton(skel):
    """골격 픽셀을 연결성을 따라 추적 → 순서 있는 점 목록 반환"""
    ys, xs = np.where(skel)
    if len(xs) < 5:
        return []

    # 픽셀 집합 및 이웃 탐색용
    pixel_set = set(zip(xs.tolist(), ys.tolist()))

    def neighbors(x, y):
        return [(x+dx, y+dy) for dx in (-1,0,1) for dy in (-1,0,1)
                if (dx,dy) != (0,0) and (x+dx, y+dy) in pixel_set]

    # 끝점 찾기 (이웃이 1개)
    endpoints = [(x, y) for x, y in pixel_set if len(neighbors(x, y)) == 1]
    if not endpoints:
        # 끝점 없으면 임의 시작
        endpoints = [(xs[0], ys[0])]

    # 가장 먼 두 끝점 쌍에서 시작
    start = endpoints[0]

    # DFS로 경로 추적
    path = []
    visited = set()
    stack = [start]
    while stack:
        cur = stack.pop()
        if cur in visited:
            continue
        visited.add(cur)
        path.append(cur)
        nbrs = [n for n in neighbors(*cur) if n not in visited]
        # 이웃 1개씩 따라가기 (분기 시 가장 직선에 가까운 방향 선택)
        if nbrs:
            if len(path) >= 2:
                dx = path[-1][0] - path[-2][0]
                dy = path[-1][1] - path[-2][1]
                def continuity(n):
                    return (n[0]-path[-1][0])*dx + (n[1]-path[-1][1])*dy
                nbrs.sort(key=continuity, reverse=True)
            stack.append(nbrs[0])

    return path


def extract_centerline_points(mask):
    """세그멘테이션 마스크 → 중심선 점 목록"""
    if mask.sum() < 50:
        return []
    skel = skeletonize(mask > 0)
    return trace_skeleton(skel)

# ── 타일 순회 ─────────────────────────────────────
total_polylines = 0
xs_range = range(0, W - TILE//2, STEP)
ys_range = range(0, H - TILE//2, STEP)
total_tiles = len(xs_range) * len(ys_range)
processed = 0

print(f"총 타일 수(예상): {total_tiles}, 처리 시작...")

for ty in ys_range:
    for tx in xs_range:
        tw = min(TILE, W - tx)
        th = min(TILE, H - ty)
        if tw < 64 or th < 64:
            continue

        # 타일 읽기
        win = Window(tx, ty, tw, th)
        data = src.read([1,2,3], window=win)
        img_arr = np.transpose(data, (1,2,0)).astype(np.uint8)

        # 검은 타일 스킵
        black_ratio = np.mean(np.all(img_arr < 10, axis=2))
        if black_ratio > BLACK_THR:
            processed += 1
            continue

        # YOLO 추론
        pil_img = Image.fromarray(img_arr)
        results = model.predict(pil_img, imgsz=TILE, conf=CONF,
                                device='cuda:0', verbose=False)

        if not results or results[0].masks is None:
            processed += 1
            continue

        masks = results[0].masks.data.cpu().numpy()
        h_r, w_r = masks.shape[1], masks.shape[2]
        sx = tw / w_r
        sy = th / h_r

        for mask in masks:
            pts_local = extract_centerline_points(mask)
            if len(pts_local) < 2:
                continue

            # 겹침 영역(overlap) 가장자리 점 제거 (타일 이음새 중복 방지)
            pts_filtered = []
            for lx, ly in pts_local:
                px_global = tx + lx * sx
                py_global = ty + ly * sy
                # 오른쪽/아래 overlap 영역은 다음 타일에서 처리
                if tx + tw < W and lx * sx > (tw - OVERLAP // 2):
                    continue
                if ty + th < H and ly * sy > (th - OVERLAP // 2):
                    continue
                pts_filtered.append(pixel_to_world(px_global, py_global))

            if len(pts_filtered) >= 2:
                msp.add_lwpolyline(pts_filtered, dxfattribs={"layer": "RAIL_CENTERLINE"})
                total_polylines += 1

        processed += 1
        if processed % 500 == 0:
            pct = processed / total_tiles * 100
            print(f"  진행: {processed}/{total_tiles} ({pct:.1f}%) | 폴리라인: {total_polylines}")

src.close()

doc.saveas(OUT_DXF)
print(f"\n완료: {total_polylines}개 폴리라인 → {OUT_DXF}")
