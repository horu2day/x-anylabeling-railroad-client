"""
SAM3 배치 자동 레이블링 파이프라인
===================================
SAM3 서버에 텍스트 프롬프트로 이미지 배치 처리 →
X-AnyLabeling 호환 JSON annotation 파일 자동 생성

사용법:
    # 기본 (sample/rail 폴더, 서버 localhost:8000)
    python tools/sam3_batch_label.py

    # 폴더 지정
    python tools/sam3_batch_label.py --input sample/rail --output output/labels

    # conf 조정 (낮출수록 더 많이 검출, 오탐도 증가)
    python tools/sam3_batch_label.py --conf 0.20

SAM3 서버 실행:
    cd X-AnyLabeling-Server
    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

import argparse
import base64
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import requests

SAM3_SERVER = "http://localhost:8000"
MODEL_ID = "segment_anything_3"

# 검출 대상 + 한국어 레이블 매핑
TARGETS = {
    "pole":           "전철주_세로",
    "catenary arm":   "전철주_가로",
    "junction box":   "통신박스",
    "electrical box": "전기박스",
    "fence":          "펜스",
}

# 시각화 색상 (BGR)
COLORS = {
    "전철주_세로": (0, 200, 255),
    "전철주_가로": (0, 100, 255),
    "통신박스":   (255, 180, 0),
    "전기박스":   (100, 255, 200),
    "펜스":       (0, 255, 100),
}


def encode_image(image_bgr: np.ndarray) -> str:
    _, buf = cv2.imencode(".png", image_bgr)  # PNG: 무손실 풀해상도
    return base64.b64encode(buf).decode("utf-8")


def sam3_text_predict(image_bgr: np.ndarray, text_prompt: str, conf: float) -> list:
    """SAM3 텍스트 프롬프트로 segmentation. shapes 리스트 반환."""
    payload = {
        "model": MODEL_ID,
        "image": encode_image(image_bgr),
        "params": {
            "text_prompt": text_prompt,
            "show_masks": True,
            "show_boxes": False,
            "conf_threshold": conf,
            "epsilon_factor": 0.002,
        },
    }
    try:
        resp = requests.post(f"{SAM3_SERVER}/v1/predict", json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        if data.get("success"):
            return data.get("data", {}).get("shapes", [])
    except Exception as e:
        print(f"    [ERROR] SAM3 호출 실패: {e}")
    return []


def process_image(image_path: Path, conf: float, vis_dir: Path | None) -> dict:
    """이미지 1장: 모든 클래스 검출 → annotation dict 반환."""
    # 한글 경로 대응: np.fromfile + imdecode
    buf = np.fromfile(str(image_path), dtype=np.uint8)
    image_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if image_bgr is None:
        return {}
    h, w = image_bgr.shape[:2]

    all_shapes = []
    class_counts = {}

    for eng_label, kor_label in TARGETS.items():
        shapes = sam3_text_predict(image_bgr, eng_label, conf)
        # label 필드를 한국어로 교체
        for s in shapes:
            s["label"] = kor_label
        all_shapes.extend(shapes)
        if shapes:
            class_counts[kor_label] = len(shapes)

    # X-AnyLabeling JSON 형식
    annotation = {
        "version": "3.3.9",
        "flags": {},
        "shapes": [
            {
                "label": s["label"],
                "points": s["points"],
                "group_id": None,
                "shape_type": s.get("shape_type", "polygon"),
                "flags": {},
                "score": round(float(s.get("score", 0)), 4),
            }
            for s in all_shapes
        ],
        "imagePath": image_path.name,
        "imageData": None,
        "imageHeight": h,
        "imageWidth": w,
    }

    # 시각화
    if vis_dir is not None:
        vis = draw_vis(image_bgr, all_shapes)
        vis_path = vis_dir / f"{image_path.stem}_vis.jpg"
        cv2.imencode(".jpg", vis)[1].tofile(str(vis_path))

    return annotation, class_counts


def draw_vis(image_bgr: np.ndarray, shapes: list) -> np.ndarray:
    vis = image_bgr.copy()
    overlay = image_bgr.copy()

    for shape in shapes:
        pts = np.array(shape["points"], dtype=np.int32)
        if len(pts) > 1 and np.array_equal(pts[0], pts[-1]):
            pts = pts[:-1]
        label = shape.get("label", "unknown")
        color = COLORS.get(label, (200, 200, 200))
        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(vis, [pts], True, color, 2)

        # 레이블 텍스트
        cx = int(pts[:, 0].mean())
        cy = int(pts[:, 1].mean())
        cv2.putText(vis, label, (cx - 30, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.addWeighted(overlay, 0.3, vis, 0.7, 0, vis)

    # 범례
    y = 25
    for kor, color in COLORS.items():
        cv2.rectangle(vis, (10, y - 12), (24, y + 2), color, -1)
        cv2.putText(vis, kor, (28, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += 20

    return vis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="sample/rail")
    parser.add_argument("--output", default="output/sam3_labels",
                        help="JSON annotation 저장 폴더")
    parser.add_argument("--vis",    default="output/sam3_vis",
                        help="시각화 이미지 저장 폴더 ('' 로 비활성화)")
    parser.add_argument("--conf",   type=float, default=0.20,
                        help="SAM3 confidence threshold (기본 0.20)")
    parser.add_argument("--classes", nargs="+",
                        help="처리할 클래스 키 목록 (기본: 전체). 예: catenary_pole catenary_arm")
    parser.add_argument("--server", default="http://localhost:8000")
    args = parser.parse_args()

    global SAM3_SERVER, TARGETS
    SAM3_SERVER = args.server

    if args.classes:
        key_map = {
            "catenary_pole":  ("catenary pole",  "전철주_세로"),
            "concrete_pole":  ("concrete pole",  "전철주_세로"),
            "catenary_arm":   ("catenary arm",   "전철주_가로"),
            "junction_box":   ("junction box",   "통신박스"),
            "electrical_box": ("electrical box", "전기박스"),
            "fence":          ("fence",          "펜스"),
        }
        TARGETS = {key_map[k][0]: key_map[k][1] for k in args.classes if k in key_map}

    # 서버 확인
    try:
        r = requests.get(f"{SAM3_SERVER}/health", timeout=5)
        print(f"[OK] SAM3 서버 연결: {SAM3_SERVER}")
    except Exception:
        print(f"[ERROR] SAM3 서버 연결 실패: {SAM3_SERVER}")
        print("       서버를 먼저 실행하세요:")
        print("       cd X-AnyLabeling-Server && uvicorn app.main:app --port 8000")
        sys.exit(1)

    # 입력
    input_path = Path(args.input)
    if input_path.is_file():
        images = [input_path]
    elif input_path.is_dir():
        images = sorted(
            list(input_path.glob("*.jpg")) +
            list(input_path.glob("*.jpeg")) +
            list(input_path.glob("*.png"))
        )
    else:
        print(f"[ERROR] 경로 없음: {input_path}")
        sys.exit(1)

    if not images:
        print(f"[ERROR] 이미지 없음: {input_path}")
        sys.exit(1)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    vis_dir = None
    if args.vis:
        vis_dir = Path(args.vis)
        vis_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n처리 대상: {len(images)}장")
    print(f"검출 클래스: {list(TARGETS.values())}")
    print(f"SAM3 conf: {args.conf}")
    print(f"annotation 저장: {out_dir}")
    if vis_dir:
        print(f"시각화 저장: {vis_dir}")
    print()

    total_counts: dict = {v: 0 for v in TARGETS.values()}
    processed = 0

    for img_path in images:
        print(f"[{processed+1}/{len(images)}] {img_path.name}")
        result = process_image(img_path, args.conf, vis_dir)
        if not result:
            print(f"  [SKIP] 처리 실패")
            continue

        annotation, class_counts = result
        n = len(annotation["shapes"])
        print(f"  → {n}개 객체 검출: {class_counts if class_counts else '없음'}")

        # JSON 저장
        json_path = out_dir / f"{img_path.stem}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(annotation, f, ensure_ascii=False, indent=2)

        for k, v in class_counts.items():
            total_counts[k] = total_counts.get(k, 0) + v
        processed += 1

    # 요약
    print("\n" + "="*50)
    print(f"완료: {processed}/{len(images)}장 처리")
    print("\n클래스별 총 검출 수:")
    for cls, cnt in total_counts.items():
        avg = cnt / max(processed, 1)
        bar = "#" * min(cnt, 30)
        print(f"  {cls:12s}: {cnt:4d}개  평균 {avg:.1f}/장  {bar}")
    print(f"\nJSON 저장: {out_dir.resolve()}")
    if vis_dir:
        print(f"시각화:   {vis_dir.resolve()}")
    print("\n다음 단계:")
    print("  X-AnyLabeling → Open Image Folder → annotation 폴더 선택")
    print("  → 오탐/미탐 수동 수정 → Export → YOLO11-seg 학습")


if __name__ == "__main__":
    main()
