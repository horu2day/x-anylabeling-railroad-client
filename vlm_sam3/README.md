# VLM + SAM3 철도 객체 검출 파이프라인

## 아키텍처

기존 서버 기반(`X-AnyLabeling-Server`) 구조와 달리, **독립 실행형 2-Stage 파이프라인**입니다.

```
┌─────────────────────────────────────────────────────────┐
│  Stage 1: VLM (Vision Language Model)                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ Florence-2   │  │ Qwen2.5-VL   │  │ Claude/GPT-4V│  │
│  │ (~1.5GB)     │  │ (~7GB)       │  │ (API, 0GB)   │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  │
│         └─────────────────┼─────────────────┘           │
│                    객체 검출 결과                          │
│              (class_name, bbox, confidence)              │
└─────────────────────┬───────────────────────────────────┘
                      │  VLM 언로드 (순차 모드)
                      ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 2: SAM3 (Segment Anything 3)                     │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Concept Prompt (text) + Box Prompt (bbox)       │   │
│  │  → 정밀 segmentation mask 생성                    │   │
│  │  (~6-8GB VRAM)                                    │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────┬───────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────┐
│  Output                                                  │
│  ├── masks/          바이너리 마스크 PNG                   │
│  ├── overlays/       마스크 오버레이 시각화                  │
│  ├── vlm_detections/ VLM bbox 시각화                      │
│  ├── annotations/    X-AnyLabeling JSON                   │
│  └── coco_annotations.json  COCO 형식                     │
└─────────────────────────────────────────────────────────┘
```

## 기존 파이프라인과의 차이

| 항목 | 기존 (서버 기반) | 새로운 (VLM+SAM3) |
|------|----------------|-------------------|
| 아키텍처 | FastAPI 서버 + HTTP 호출 | 독립 실행 스크립트 |
| 검출 모델 | YOLO-World / SAM3 text | VLM (Florence-2/Qwen/Claude) |
| 장점 | 실시간, GUI 연동 | 장면 이해력, 유연한 프롬프트 |
| VRAM | 상시 ~8GB 점유 | 순차 로딩으로 12GB 가능 |
| SAM3 연동 | HTTP API (간접) | Python API (직접 로딩) |
| 환경 | `.venv` | `.venv_vlm` (별도) |

## 환경 설치

```bash
# 프로젝트 루트에서 실행
vlm_sam3\setup_env.bat
```

또는 수동 설치:
```bash
python -m venv .venv_vlm
.venv_vlm\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r vlm_sam3\requirements.txt
```

## 사용법

### 기본 실행 (Florence-2 + SAM3, 순차 로딩)
```bash
.venv_vlm\Scripts\activate
python vlm_sam3/pipeline.py
```

### VLM 백엔드 변경
```bash
# Qwen2.5-VL (더 강력한 장면 이해, 순차 로딩 필수)
python vlm_sam3/pipeline.py --vlm qwen2.5vl

# Claude Vision API (VRAM 0GB, API 비용 발생)
python vlm_sam3/pipeline.py --vlm claude --mode api_vlm

# GPT-4V API
python vlm_sam3/pipeline.py --vlm openai --mode api_vlm
```

### 입출력 경로 지정
```bash
python vlm_sam3/pipeline.py --input 노선영상/extracted_frames --output vlm_sam3/output
```

### SAM3 프롬프트 모드
```bash
# text: VLM이 식별한 클래스명으로 concept segmentation
python vlm_sam3/pipeline.py --prompt-mode text

# bbox: VLM bbox를 SAM3 box prompt로 전달
python vlm_sam3/pipeline.py --prompt-mode bbox

# both: text + bbox 결합 (기본, 최고 정확도)
python vlm_sam3/pipeline.py --prompt-mode both
```

### Confidence 조정
```bash
python vlm_sam3/pipeline.py --conf 0.15   # 더 많이 검출 (오탐 증가)
python vlm_sam3/pipeline.py --conf 0.40   # 정밀 검출 (미탐 증가)
```

## 파이프라인 모드 (VRAM 전략)

| 모드 | 설명 | 필요 VRAM |
|------|------|-----------|
| `sequential` | VLM→언로드→SAM3 (기본) | ~8GB (RTX 3060 OK) |
| `concurrent` | VLM+SAM3 동시 로딩 | ~16GB+ (RTX 4080+) |
| `api_vlm` | API VLM + 로컬 SAM3 | ~8GB (SAM3만) |

## 설정 파일

`vlm_sam3/config.yaml`에서 모든 파라미터를 조정할 수 있습니다.
CLI 옵션은 설정 파일 값을 오버라이드합니다.

## 검출 대상 객체

config.yaml의 `railway.target_objects`에서 관리:
- railway track (레일)
- railway signal (신호기)
- catenary pole (전주)
- catenary wire (전차선)
- junction box (함체)
- railroad switch (분기기)
- sleeper (침목)
- fence (울타리)
- platform (승강장)

## 출력물 활용

1. **X-AnyLabeling에서 검수**: `annotations/` 폴더를 X-AnyLabeling으로 열어 수동 검수/수정
2. **YOLO 학습**: COCO JSON → YOLO format 변환 후 학습
3. **DXF 변환**: 기존 `tools/rail_centerline_dxf.py`와 연계하여 마스크→DXF
