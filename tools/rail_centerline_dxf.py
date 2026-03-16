"""
2cm GSD 드론 TIF 전체에서 레일 중심선 추출 → DXF 저장

중심선 추출 방법:
  masks.xy 폴리곤 → PCA 주축(길이 방향) 찾기
  → 주축 방향으로 슬라이싱 → 폭 방향 중앙값
  → 계단 현상 없는 매끄러운 중심선
"""
import numpy as np
import rasterio
from rasterio.windows import Window
from ultralytics import YOLO
from PIL import Image
import ezdxf


def polygon_to_centerline(xy_tile, spacing_px=10):
    """
    타일 픽셀 좌표 폴리곤 → 중심선 점 목록

    Parameters
    ----------
    xy_tile    : list of (x, y)  타일 픽셀 좌표 (masks.xy × sx/sy)
    spacing_px : int  슬라이싱 간격 [픽셀]  (10px = 0.2m at 2cm GSD)

    Returns
    -------
    list of (x, y)  중심선 타일 픽셀 좌표
    """
    pts = np.asarray(xy_tile, dtype=float)
    if len(pts) < 4:
        return []

    # PCA: 주축(레일 길이 방향) 탐색
    center = pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(pts - center, full_matrices=False)
    v_long = Vt[0]   # 길이 방향 단위벡터
    v_perp = Vt[1]   # 폭 방향 단위벡터

    local = pts - center
    t = local @ v_long   # 길이 방향 좌표

    t_min, t_max = t.min(), t.max()
    if t_max - t_min < spacing_px:
        # 너무 짧으면 무게중심 1점
        return [center.tolist()]

    n_bins = max(2, int((t_max - t_min) / spacing_px) + 1)
    bins = np.linspace(t_min, t_max, n_bins + 1)

    centerline = []
    s_all = local @ v_perp   # 폭 방향 좌표 (전체)

    for i in range(n_bins):
        mask = (t >= bins[i]) & (t <= bins[i + 1])
        if mask.sum() < 2:
            continue
        t_c = (bins[i] + bins[i + 1]) / 2
        s_vals = s_all[mask]
        # 폭 방향: 폴리곤 양 가장자리의 기하학적 중앙
        s_c = (s_vals.max() + s_vals.min()) / 2
        pt = center + t_c * v_long + s_c * v_perp
        centerline.append(pt.tolist())

    return centerline


# ── 설정 ──────────────────────────────────────────
SRC_TIF   = "drone_2cm/22)조치원(STA.127+570~131+300).tif"
MODEL_PT  = "runs/segment/output/yolo_train_2cm/rail_seg_v2/weights/best.pt"
OUT_DXF   = "output/rail_centerline_2cm_pca.dxf"
TILE      = 1024
OVERLAP   = 128
STEP      = TILE - OVERLAP
CONF      = 0.3
BLACK_THR = 0.4    # 검은 픽셀 비율 임계값
SPACING   = 10     # 중심선 샘플 간격 [픽셀]  10px ≈ 0.2m

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


# ── 타일 순회 ─────────────────────────────────────
total_polylines = 0
xs_range = range(0, W - TILE // 2, STEP)
ys_range = range(0, H - TILE // 2, STEP)
total_tiles = len(xs_range) * len(ys_range)
processed = 0

print(f"총 타일 수(예상): {total_tiles}, 처리 시작...")

for ty in ys_range:
    for tx in xs_range:
        tw = min(TILE, W - tx)
        th = min(TILE, H - ty)
        if tw < 64 or th < 64:
            continue

        win = Window(tx, ty, tw, th)
        data = src.read([1, 2, 3], window=win)
        img_arr = np.transpose(data, (1, 2, 0)).astype(np.uint8)

        black_ratio = np.mean(np.all(img_arr < 10, axis=2))
        if black_ratio > BLACK_THR:
            processed += 1
            continue

        pil_img = Image.fromarray(img_arr)
        results = model.predict(pil_img, imgsz=TILE, conf=CONF,
                                device='cuda:0', verbose=False)

        if not results or results[0].masks is None:
            processed += 1
            continue

        masks_data = results[0].masks.data.cpu().numpy()
        h_r, w_r = masks_data.shape[1], masks_data.shape[2]
        sx = tw / w_r
        sy = th / h_r

        for xy in results[0].masks.xy:
            if len(xy) < 4:
                continue

            # YOLO 좌표 → 타일 픽셀 좌표
            xy_tile = [(float(lx) * sx, float(ly) * sy) for lx, ly in xy]

            # PCA 슬라이싱으로 중심선 추출 (타일 픽셀 좌표)
            cl_tile = polygon_to_centerline(xy_tile, spacing_px=SPACING)
            if len(cl_tile) < 2:
                continue

            # overlap 가장자리 제거 + 세계 좌표 변환
            pts_world = []
            for lx, ly in cl_tile:
                if tx + tw < W and lx > (tw - OVERLAP // 2):
                    continue
                if ty + th < H and ly > (th - OVERLAP // 2):
                    continue
                pts_world.append(pixel_to_world(tx + lx, ty + ly))

            if len(pts_world) >= 2:
                msp.add_lwpolyline(pts_world,
                                   dxfattribs={"layer": "RAIL_CENTERLINE"})
                total_polylines += 1

        processed += 1
        if processed % 500 == 0:
            pct = processed / total_tiles * 100
            print(f"  진행: {processed}/{total_tiles} ({pct:.1f}%)"
                  f" | 폴리라인: {total_polylines}")

src.close()

doc.saveas(OUT_DXF)
print(f"\n완료: {total_polylines}개 폴리라인 → {OUT_DXF}")
