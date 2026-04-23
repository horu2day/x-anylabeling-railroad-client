"""
드론 영상에서 프레임 추출 → SAM3 서버로 모든 객체 세그멘테이션
Usage:
    1. SAM3 서버 시작: start_server.bat
    2. python tools/video_sam3_segment.py
"""

import base64
import cv2
import json
import numpy as np
import requests
import sys
from pathlib import Path

# === 설정 ===
VIDEO_PATH = Path("sample/rail.mp4")
OUTPUT_DIR = Path("output/video_segmentation")
SAM3_URL = "http://localhost:8000"
MODEL_ID = "segment_anything_3"
FRAME_INTERVAL = 30  # 30fps 영상에서 1초 간격

# 철도 시설물 + 일반 객체 프롬프트
PROMPTS = [
    "catenary pole",           # 전철주
    "junction box",            # 전기박스
    "utility box",             # 통신박스
    "rail track",              # 레일
    "fence",                   # 펜스
    "cable",                   # 전선/케이블
    "sign",                    # 표지판
    "building",                # 건물
    "vegetation",              # 식생
]

# 클래스별 색상 (BGR)
COLORS = {
    "catenary pole":      (0, 0, 255),      # 빨강
    "junction box":       (0, 165, 255),     # 주황
    "utility box":        (0, 255, 255),     # 노랑
    "rail track":         (255, 0, 0),       # 파랑
    "fence":              (255, 0, 255),     # 자홍
    "cable":              (0, 255, 0),       # 초록
    "sign":               (255, 255, 0),     # 시안
    "building":           (128, 128, 255),   # 연한 빨강
    "vegetation":         (0, 128, 0),       # 진한 초록
}


def extract_frames(video_path: Path, interval: int) -> list[tuple[int, np.ndarray]]:
    """동영상에서 일정 간격으로 프레임 추출"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {video_path.name} | {total} frames @ {fps:.0f}fps | {total/fps:.1f}s")

    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            frames.append((idx, frame))
        idx += 1

    cap.release()
    print(f"Extracted {len(frames)} frames (interval={interval})")
    return frames


def encode_frame(frame: np.ndarray) -> str:
    """프레임을 base64로 인코딩"""
    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return base64.b64encode(buffer).decode("utf-8")


def predict_sam3(image_b64: str, text_prompt: str) -> dict:
    """SAM3 서버에 예측 요청"""
    payload = {
        "model": MODEL_ID,
        "image": image_b64,
        "params": {
            "text_prompt": text_prompt,
            "conf_threshold": 0.25,
        },
    }
    try:
        resp = requests.post(f"{SAM3_URL}/v1/predict", json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        print(f"ERROR: SAM3 서버에 연결할 수 없습니다. start_server.bat을 먼저 실행하세요.")
        sys.exit(1)
    except Exception as e:
        print(f"  ERROR [{text_prompt}]: {e}")
        return {"success": False}


def draw_shapes_on_frame(frame: np.ndarray, shapes: list, prompt: str) -> np.ndarray:
    """세그멘테이션 결과를 프레임 위에 그리기"""
    overlay = frame.copy()
    color = COLORS.get(prompt, (200, 200, 200))

    for shape in shapes:
        points = np.array(shape["points"], dtype=np.int32)
        shape_type = shape.get("shape_type", "polygon")

        if shape_type == "polygon" and len(points) >= 3:
            cv2.fillPoly(overlay, [points], color)
            cv2.polylines(frame, [points], True, color, 2)
        elif shape_type == "rectangle" and len(points) == 2:
            cv2.rectangle(overlay, tuple(points[0]), tuple(points[1]), color, -1)
            cv2.rectangle(frame, tuple(points[0]), tuple(points[1]), color, 2)

        # 라벨 텍스트
        label = shape.get("label", prompt)
        score = shape.get("score")
        text = f"{label}" + (f" {score:.2f}" if score else "")
        if len(points) > 0:
            tx, ty = int(points[0][0]), int(points[0][1]) - 5
            cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 반투명 오버레이 블렌딩
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    return frame


def create_layer_image(frame_shape: tuple, shapes: list, prompt: str) -> np.ndarray:
    """클래스별 개별 레이어 이미지 (투명 배경 위 마스크)"""
    h, w = frame_shape[:2]
    color = COLORS.get(prompt, (200, 200, 200))
    bgr = np.zeros((h, w, 3), dtype=np.uint8)
    alpha = np.zeros((h, w), dtype=np.uint8)

    for shape in shapes:
        points = np.array(shape["points"], dtype=np.int32)
        if shape.get("shape_type") == "polygon" and len(points) >= 3:
            cv2.fillPoly(bgr, [points], color)
            cv2.fillPoly(alpha, [points], 180)

    layer = np.dstack([bgr, alpha])
    return layer


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 서버 상태 확인
    print("=== SAM3 서버 연결 확인 ===")
    try:
        health = requests.get(f"{SAM3_URL}/health", timeout=5)
        print(f"Server status: {health.json()}")
    except requests.exceptions.ConnectionError:
        print("ERROR: SAM3 서버가 실행 중이 아닙니다!")
        print("  → start_server.bat을 먼저 실행하세요.")
        sys.exit(1)

    # 2. 프레임 추출
    print("\n=== 프레임 추출 ===")
    frames = extract_frames(VIDEO_PATH, FRAME_INTERVAL)

    # 3. 각 프레임별, 각 프롬프트별 세그멘테이션
    all_results = {}

    for frame_idx, (fidx, frame) in enumerate(frames):
        print(f"\n=== Frame {fidx} ({frame_idx+1}/{len(frames)}) ===")

        frame_b64 = encode_frame(frame)
        frame_results = {}
        composite = frame.copy()

        for prompt in PROMPTS:
            print(f"  Segmenting: {prompt}...", end=" ")
            result = predict_sam3(frame_b64, prompt)

            if result.get("success") and result.get("data", {}).get("shapes"):
                shapes = result["data"]["shapes"]
                n = len(shapes)
                print(f"→ {n} objects found")
                frame_results[prompt] = shapes

                # 합성 이미지에 그리기
                composite = draw_shapes_on_frame(composite, shapes, prompt)

                # 개별 레이어 저장 (PNG with alpha)
                layer = create_layer_image(frame.shape, shapes, prompt)
                layer_name = prompt.replace(" ", "_")
                layer_path = OUTPUT_DIR / f"frame_{fidx:04d}_layer_{layer_name}.png"
                cv2.imwrite(str(layer_path), layer)
            else:
                print("→ no objects")

        # 합성 이미지 저장
        composite_path = OUTPUT_DIR / f"frame_{fidx:04d}_composite.jpg"
        cv2.imwrite(str(composite_path), composite)

        # 원본 프레임 저장 (비교용)
        original_path = OUTPUT_DIR / f"frame_{fidx:04d}_original.jpg"
        cv2.imwrite(str(original_path), frame)

        all_results[f"frame_{fidx}"] = {
            "frame_index": fidx,
            "detections": {
                prompt: len(shapes)
                for prompt, shapes in frame_results.items()
            },
        }

    # 4. 결과 요약 저장
    summary_path = OUTPUT_DIR / "segmentation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # 5. 범례 이미지 생성
    legend = np.zeros((len(PROMPTS) * 30 + 20, 300, 3), dtype=np.uint8)
    for i, (prompt, color) in enumerate(COLORS.items()):
        y = i * 30 + 20
        cv2.rectangle(legend, (10, y - 12), (30, y + 5), color, -1)
        cv2.putText(legend, prompt, (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imwrite(str(OUTPUT_DIR / "legend.jpg"), legend)

    print(f"\n=== 완료 ===")
    print(f"결과 저장: {OUTPUT_DIR}/")
    print(f"  - frame_XXXX_original.jpg   : 원본 프레임")
    print(f"  - frame_XXXX_composite.jpg  : 전체 세그멘테이션 합성")
    print(f"  - frame_XXXX_layer_*.png    : 클래스별 개별 레이어 (투명 배경)")
    print(f"  - segmentation_summary.json : 검출 요약")
    print(f"  - legend.jpg                : 색상 범례")


if __name__ == "__main__":
    main()
