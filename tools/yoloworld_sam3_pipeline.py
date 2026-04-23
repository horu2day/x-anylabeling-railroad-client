"""
YOLO-World + SAM3 반자동 레이블링 파이프라인
=============================================
YOLO-World로 bbox 검출 → SAM3로 polygon mask 생성 → 시각화 저장

사용법:
    python tools/yoloworld_sam3_pipeline.py --input sample/rail --output output/labeled
    python tools/yoloworld_sam3_pipeline.py --input sample/rail/frame_00000.jpg

SAM3 서버가 실행 중이어야 합니다:
    cd X-AnyLabeling-Server && uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

import argparse
import base64
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import requests

# ── 검출 대상 클래스 (YOLO-World 텍스트 프롬프트) ──────────────────────────
TARGET_CLASSES = [
    "catenary pole",       # 전철주 (세로 기둥)
    "catenary arm",        # 전철주 (가로 암)
    "junction box",        # 통신/전기 박스
    "fence",               # 펜스
]

# 클래스별 색상 (BGR)
CLASS_COLORS = {
    "catenary pole": (0, 200, 255),    # 주황
    "catenary arm":  (0, 100, 255),    # 빨강
    "junction box":  (255, 180, 0),    # 파랑
    "fence":         (0, 255, 100),    # 초록
}

SAM3_SERVER = "http://localhost:8000"
MODEL_ID = "segment_anything_3"


# ─────────────────────────────────────────────────────────────────────────────
# YOLO-World 초기화
# ─────────────────────────────────────────────────────────────────────────────
def load_yolo_world(model_size: str = "s"):
    """YOLO-World 모델 로드 (자동 다운로드)."""
    from ultralytics import YOLOWorld

    model_name = f"yolov8{model_size}-worldv2.pt"
    print(f"[YOLO-World] 모델 로드: {model_name}")
    model = YOLOWorld(model_name)
    model.set_classes(TARGET_CLASSES)
    print(f"[YOLO-World] 검출 클래스: {TARGET_CLASSES}")
    return model


def detect_with_yoloworld(model, image_bgr: np.ndarray, conf: float = 0.15):
    """YOLO-World로 bbox 검출. [(x1,y1,x2,y2,conf,class_name), ...] 반환."""
    results = model.predict(image_bgr, conf=conf, verbose=False)
    detections = []
    if results and len(results) > 0:
        r = results[0]
        boxes = r.boxes
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                score = float(box.conf[0].cpu())
                cls_idx = int(box.cls[0].cpu())
                cls_name = TARGET_CLASSES[cls_idx] if cls_idx < len(TARGET_CLASSES) else f"class_{cls_idx}"
                detections.append((float(x1), float(y1), float(x2), float(y2), score, cls_name))
    return detections


# ─────────────────────────────────────────────────────────────────────────────
# SAM3 서버 호출
# ─────────────────────────────────────────────────────────────────────────────
def encode_image(image_bgr: np.ndarray) -> str:
    """이미지를 base64 문자열로 인코딩."""
    _, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return base64.b64encode(buf).decode("utf-8")


def sam3_segment(image_bgr: np.ndarray, boxes: list, conf_threshold: float = 0.25):
    """SAM3 서버에 bbox 전달 → polygon masks 반환.

    Args:
        image_bgr: 원본 이미지
        boxes: [(x1,y1,x2,y2,score,class_name), ...]
        conf_threshold: SAM3 신뢰도 임계값

    Returns:
        shapes: [{"label": str, "points": [[x,y],...], "score": float}, ...]
    """
    marks = [
        {
            "type": "rectangle",
            "label": 1,
            "data": [b[0], b[1], b[2], b[3]],
        }
        for b in boxes
    ]

    payload = {
        "model": MODEL_ID,
        "image": encode_image(image_bgr),
        "params": {
            "marks": marks,
            "show_masks": True,
            "show_boxes": False,
            "conf_threshold": conf_threshold,
            "epsilon_factor": 0.002,
        },
    }

    try:
        resp = requests.post(f"{SAM3_SERVER}/v1/predict", json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.ConnectionError:
        print(f"  [ERROR] SAM3 서버에 연결할 수 없습니다: {SAM3_SERVER}")
        return []
    except Exception as e:
        print(f"  [ERROR] SAM3 호출 실패: {e}")
        return []

    if data.get("status") != "success":
        print(f"  [ERROR] SAM3 응답 오류: {data}")
        return []

    return data.get("data", {}).get("shapes", [])


# ─────────────────────────────────────────────────────────────────────────────
# 시각화
# ─────────────────────────────────────────────────────────────────────────────
def draw_results(image_bgr: np.ndarray, detections: list, shapes: list) -> np.ndarray:
    """bbox + mask를 이미지에 그리기."""
    vis = image_bgr.copy()
    overlay = image_bgr.copy()

    # SAM3 polygon masks 그리기
    for shape in shapes:
        pts = np.array(shape["points"], dtype=np.int32)
        # 첫점=끝점이면 마지막 제거
        if len(pts) > 1 and np.array_equal(pts[0], pts[-1]):
            pts = pts[:-1]
        label = shape.get("label", "unknown")
        color = CLASS_COLORS.get(label, (200, 200, 200))
        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(vis, [pts], True, color, 2)

    cv2.addWeighted(overlay, 0.35, vis, 0.65, 0, vis)

    # YOLO-World bbox + 라벨 그리기
    for x1, y1, x2, y2, score, cls_name in detections:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        color = CLASS_COLORS.get(cls_name, (200, 200, 200))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label_text = f"{cls_name} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(vis, label_text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    # 범례
    legend_y = 20
    for cls_name, color in CLASS_COLORS.items():
        cv2.rectangle(vis, (10, legend_y), (25, legend_y + 15), color, -1)
        cv2.putText(vis, cls_name, (30, legend_y + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        legend_y += 22

    return vis


# ─────────────────────────────────────────────────────────────────────────────
# 단일 이미지 처리
# ─────────────────────────────────────────────────────────────────────────────
def process_image(yolo_model, image_path: Path, output_dir: Path,
                  yolo_conf: float, sam3_conf: float, skip_sam3: bool) -> dict:
    """이미지 1장 처리. 결과 딕셔너리 반환."""
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        print(f"  [SKIP] 이미지 읽기 실패: {image_path}")
        return {}

    h, w = image_bgr.shape[:2]
    print(f"\n  이미지: {image_path.name} ({w}x{h})")

    # ── Step 1: YOLO-World 검출 ────────────────────────────────────────────
    detections = detect_with_yoloworld(yolo_model, image_bgr, conf=yolo_conf)
    print(f"  YOLO-World: {len(detections)}개 검출")
    for d in detections:
        print(f"    [{d[5]}] conf={d[4]:.3f}  bbox=({d[0]:.0f},{d[1]:.0f},{d[2]:.0f},{d[3]:.0f})")

    shapes = []
    if not skip_sam3 and detections:
        # ── Step 2: SAM3 mask 생성 ─────────────────────────────────────────
        print(f"  SAM3: {len(detections)}개 bbox → mask 요청 중...")
        shapes = sam3_segment(image_bgr, detections, conf_threshold=sam3_conf)
        print(f"  SAM3: {len(shapes)}개 mask 반환")
        for s in shapes:
            pts_count = len(s.get("points", []))
            print(f"    [{s.get('label')}] score={s.get('score', 0):.3f}  points={pts_count}")

    # ── Step 3: 시각화 저장 ───────────────────────────────────────────────
    vis = draw_results(image_bgr, detections, shapes)
    output_path = output_dir / f"{image_path.stem}_result.jpg"
    cv2.imwrite(str(output_path), vis)
    print(f"  저장: {output_path}")

    return {
        "image": image_path.name,
        "detections": [
            {"class": d[5], "conf": round(d[4], 3),
             "bbox": [round(d[0]), round(d[1]), round(d[2]), round(d[3])]}
            for d in detections
        ],
        "masks": len(shapes),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="YOLO-World + SAM3 파이프라인")
    parser.add_argument("--input", default="sample/rail",
                        help="이미지 파일 또는 폴더 경로")
    parser.add_argument("--output", default="output/yoloworld_sam3",
                        help="결과 저장 폴더")
    parser.add_argument("--model-size", default="s", choices=["s", "m", "l", "x"],
                        help="YOLO-World 모델 크기 (s/m/l/x)")
    parser.add_argument("--yolo-conf", type=float, default=0.10,
                        help="YOLO-World 검출 임계값 (기본 0.10)")
    parser.add_argument("--sam3-conf", type=float, default=0.20,
                        help="SAM3 마스크 임계값 (기본 0.20)")
    parser.add_argument("--skip-sam3", action="store_true",
                        help="SAM3 건너뛰고 YOLO-World bbox만 시각화")
    parser.add_argument("--server", default="http://localhost:8000",
                        help="SAM3 서버 주소")
    args = parser.parse_args()

    global SAM3_SERVER
    SAM3_SERVER = args.server

    # SAM3 서버 상태 확인
    if not args.skip_sam3:
        try:
            resp = requests.get(f"{SAM3_SERVER}/health", timeout=5)
            if resp.status_code == 200:
                print(f"[OK] SAM3 서버 연결: {SAM3_SERVER}")
            else:
                print(f"[WARN] SAM3 서버 응답 이상 (status={resp.status_code})")
        except Exception:
            print(f"[WARN] SAM3 서버 연결 실패 ({SAM3_SERVER}). --skip-sam3 로 bbox만 볼 수 있음.")
            ans = input("계속 진행하시겠습니까? (y/N): ").strip().lower()
            if ans != "y":
                sys.exit(0)

    # 입력 경로 처리
    input_path = Path(args.input)
    if input_path.is_file():
        image_files = [input_path]
    elif input_path.is_dir():
        image_files = sorted(
            list(input_path.glob("*.jpg")) +
            list(input_path.glob("*.jpeg")) +
            list(input_path.glob("*.png"))
        )
    else:
        print(f"[ERROR] 입력 경로를 찾을 수 없습니다: {input_path}")
        sys.exit(1)

    if not image_files:
        print(f"[ERROR] 이미지 파일 없음: {input_path}")
        sys.exit(1)

    print(f"\n총 {len(image_files)}개 이미지 처리 예정")

    # 출력 폴더 생성
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # YOLO-World 로드
    yolo_model = load_yolo_world(args.model_size)

    # 처리
    summary = []
    for img_path in image_files:
        result = process_image(
            yolo_model, img_path, output_dir,
            yolo_conf=args.yolo_conf,
            sam3_conf=args.sam3_conf,
            skip_sam3=args.skip_sam3,
        )
        if result:
            summary.append(result)

    # 요약 저장
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 통계 출력
    print("\n" + "="*50)
    print("처리 완료 요약")
    print("="*50)
    total_det = sum(len(r["detections"]) for r in summary)
    total_mask = sum(r["masks"] for r in summary)
    print(f"처리 이미지:  {len(summary)}장")
    print(f"YOLO 검출:   {total_det}개 (평균 {total_det/max(len(summary),1):.1f}/장)")
    print(f"SAM3 마스크: {total_mask}개 (평균 {total_mask/max(len(summary),1):.1f}/장)")

    # 클래스별 집계
    class_counts: dict = {}
    for r in summary:
        for d in r["detections"]:
            cls = d["class"]
            class_counts[cls] = class_counts.get(cls, 0) + 1
    if class_counts:
        print("\n클래스별 검출 수:")
        for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1]):
            print(f"  {cls:20s}: {cnt}개")

    print(f"\n결과 저장: {output_dir.resolve()}")
    print(f"요약 JSON:  {summary_path.resolve()}")


if __name__ == "__main__":
    main()
