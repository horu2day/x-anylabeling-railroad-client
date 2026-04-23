"""
얇은 중공단면(도너츠 형태) 검출 및 표시 스크립트
드론 영상에서 전철주 상단 원형 단면을 찾아 표시

사용법:
  python detect_hollow_section.py <image> [--radius <px>] [--tol <px>] [--topk <n>]

  --radius : 3D 프로그램 줌 기준 단면 반지름 (픽셀). 미지정 시 자동 탐색.
  --tol    : radius ± 허용 오차 (기본 30px)
  --topk   : 표시할 최대 후보 수 (기본 3)
"""
import argparse
import cv2
import numpy as np
from pathlib import Path


def ring_score(edges: np.ndarray, cx: int, cy: int, r: int,
               H: int, W: int, inner_ratio: float = 0.60) -> float:
    """
    링(도넛) 특성 점수.
    - 링 영역(외곽~내곽) 엣지 강도 / 내부 엣지 강도 비율 (최대 5 클리핑)
    - 이미지 중심 근접도 보정 (중심에 가까울수록 가산점)
    Canny edges는 외부에서 한 번만 계산해 전달받음.
    """
    inner_r = max(int(r * inner_ratio), 3)

    mask_full  = np.zeros((H, W), np.uint8)
    mask_inner = np.zeros((H, W), np.uint8)
    cv2.circle(mask_full,  (cx, cy), r,       255, -1)
    cv2.circle(mask_inner, (cx, cy), inner_r, 255, -1)
    mask_ring = cv2.subtract(mask_full, mask_inner)

    if mask_ring.any() == 0 or mask_inner.any() == 0:
        return 0.0

    ring_edge  = float(edges[mask_ring  > 0].mean())
    inner_edge = float(edges[mask_inner > 0].mean()) + 1e-6
    edge_ratio = min(ring_edge / inner_edge, 5.0)

    dist_from_center = np.hypot(cx - W / 2, cy - H / 2)
    max_dist = np.hypot(W / 2, H / 2)
    center_bonus = 1.0 - dist_from_center / max_dist  # 0~1

    return edge_ratio * (0.7 + 0.3 * center_bonus)


def detect_hollow_section(
    image_path: str,
    output_path: str | None = None,
    radius_px: int | None = None,   # 3D 프로그램 줌 기준 알려진 반지름
    tol_px: int = 30,               # radius_px ± 허용 오차
    top_k: int = 3,
) -> str:
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"이미지 로드 실패: {image_path}")

    H, W = img.shape[:2]
    print(f"이미지 크기: {W}×{H}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    blurred = cv2.GaussianBlur(clahe.apply(gray), (7, 7), 2)

    # Canny 엣지를 미리 한 번만 계산 (ring_score에서 재사용)
    edges = cv2.Canny(blurred, 30, 90)

    # 반경 범위 결정
    if radius_px is not None:
        min_r = max(5, radius_px - tol_px)
        max_r = radius_px + tol_px
    else:
        short = min(W, H)
        min_r = max(10, short // 20)
        max_r = short // 3

    print(f"탐색 반경: {min_r} ~ {max_r} px"
          + (f"  (기준={radius_px}px ±{tol_px})" if radius_px else " (자동)"))

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.0,
        minDist=min_r * 3,
        param1=80,
        param2=35,
        minRadius=min_r,
        maxRadius=max_r,
    )

    result = img.copy()

    if circles is None:
        print("HoughCircles 미검출")
    else:
        circles = np.round(circles[0]).astype(int)
        print(f"HoughCircles 후보: {len(circles)}개 → 링 스코어 계산 중...")

        scored = sorted(
            [(ring_score(edges, cx, cy, r, H, W), cx, cy, r)
             for cx, cy, r in circles],
            reverse=True,
        )

        # 1위: 녹색(굵게), 2위: 노랑, 3위: 파랑
        colors = [(0, 255, 0), (0, 220, 255), (255, 120, 0)]

        print(f"\n상위 {min(top_k, len(scored))}개 중공단면 후보:")
        for rank, (score, cx, cy, r) in enumerate(scored[:top_k]):
            color = colors[rank % len(colors)]
            inner_r = max(int(r * 0.60), 5)
            lw = 3 if rank == 0 else 2

            cv2.circle(result, (cx, cy), r,       color, lw)
            cv2.circle(result, (cx, cy), inner_r, color, 1)
            cv2.circle(result, (cx, cy), 6,       (0, 0, 255), -1)
            arm = r + 15
            cv2.line(result, (cx - arm, cy), (cx + arm, cy), color, 1)
            cv2.line(result, (cx, cy - arm), (cx, cy + arm), color, 1)
            cv2.putText(result, f"#{rank+1} r={r}px s={score:.2f}",
                        (cx - r, cy - r - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            print(f"  #{rank+1}: 중심=({cx},{cy}), 반지름={r}px, 링스코어={score:.3f}")

    if output_path is None:
        p = Path(image_path)
        output_path = str(p.parent / (p.stem + "_hollow_detected" + p.suffix))

    cv2.imencode(Path(output_path).suffix, result)[1].tofile(output_path)
    print(f"\n결과 저장: {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="얇은 중공단면 검출")
    parser.add_argument("image", nargs="?",
                        default=r"d:\MYCLAUDE_PROJECT\x-anylabeling01\data\Message_2026-04-23T11_13_29+09_00.png")
    parser.add_argument("--radius", type=int, default=None,
                        help="3D 프로그램 줌 기준 단면 반지름(px). 미지정 시 자동 탐색.")
    parser.add_argument("--tol",    type=int, default=30,
                        help="radius ± 허용 오차 (기본 30px)")
    parser.add_argument("--topk",  type=int, default=3,
                        help="표시할 최대 후보 수 (기본 3)")
    args = parser.parse_args()

    detect_hollow_section(
        args.image,
        radius_px=args.radius,
        tol_px=args.tol,
        top_k=args.topk,
    )
