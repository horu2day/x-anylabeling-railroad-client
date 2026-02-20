# X-AnyLabeling + SAM3 설정 및 사용 가이드

Meta의 SAM3 (Segment Anything Model 3, 2025년 11월 출시)를 활용한 제로샷 객체 인식 데스크탑 애플리케이션.

## 아키텍처

```
[X-AnyLabeling GUI]  <--HTTP-->  [X-AnyLabeling-Server:8000]  <--GPU-->  [SAM3 Model]
 (.venv_client)                   (.venv)                                 (sam3.pt 3.3GB)
```

- **서버 (.venv)**: SAM3 모델을 GPU에서 실행. CUDA PyTorch 필요
- **클라이언트 (.venv_client)**: GUI 전용. PyTorch 불필요
- **두 환경을 반드시 분리** (같은 venv 사용 시 CUDA DLL 충돌 WinError 1114 발생)

---

## 전제 조건

| 항목 | 요구사항 |
|------|----------|
| OS | Windows 10/11 |
| Python | **3.12 이상** |
| GPU | NVIDIA GPU (VRAM 12GB 이상 권장) |
| NVIDIA 드라이버 | 최신 버전 |
| CUDA Toolkit | **12.6** (https://developer.nvidia.com/cuda-12-6-0-download-archive) |
| Git | 설치 필요 |
| Hugging Face | SAM3 접근 권한 필요 (Gated 모델) |

---

## 설치 순서

### Step 1: 프로젝트 준비

```powershell
cd D:\MYCLAUDE_PROJECT\x-anylabeling01

# X-AnyLabeling-Server 클론 (이미 있으면 생략)
git clone https://github.com/CVHub520/X-AnyLabeling-Server.git
```

### Step 2: 서버 환경 설치 (.venv)

```powershell
# 가상환경 생성
python -m venv .venv

# 1) PyTorch + CUDA 12.6 설치 (반드시 먼저!)
.venv\Scripts\pip.exe install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 2) Windows용 Triton 설치 (SAM3 필수)
.venv\Scripts\pip.exe install "triton-windows<3.7"

# 3) 나머지 서버 의존성 설치
.venv\Scripts\pip.exe install -r requirements.txt

# 4) X-AnyLabeling-Server 설치
.venv\Scripts\pip.exe install -e X-AnyLabeling-Server
```

### Step 3: 클라이언트 환경 설치 (.venv_client)

```powershell
# 별도 가상환경 생성
python -m venv .venv_client

# 클라이언트만 설치
.venv_client\Scripts\pip.exe install -r requirements-xanylabeling.txt
```

### Step 4: Hugging Face 로그인 및 SAM3 모델 다운로드

**Hugging Face 토큰:** https://huggingface.co/settings/tokens 에서 발급
(토큰은 보안상 문서에 포함하지 않음 - 개인 보관)

```powershell
# HF 로그인 (최초 1회)
.venv\Scripts\python.exe -c "from huggingface_hub import login; login(token='YOUR_HF_TOKEN')"

# SAM3 모델 다운로드 (~3.3GB, Gated 모델 - 접근 권한 필요)
# 접근 권한: https://huggingface.co/facebook/sam3 에서 "Request access" 후 승인 대기
.venv\Scripts\python.exe -c "from huggingface_hub import hf_hub_download; hf_hub_download('facebook/sam3', 'sam3.pt', local_dir='X-AnyLabeling-Server', token='YOUR_HF_TOKEN')"
```

CLIP 어휘 파일도 필요:
```powershell
.venv\Scripts\python.exe -c "import urllib.request; urllib.request.urlretrieve('https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz', 'X-AnyLabeling-Server/bpe_simple_vocab_16e6.txt.gz')"
```

### Step 5: CUDA 설치 확인

```powershell
.venv\Scripts\python.exe -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"
```

출력 예시: `CUDA: True, Device: NVIDIA GeForce RTX 3060`

---

## 실행 방법

### 터미널 1 - 서버 시작

```powershell
cd D:\MYCLAUDE_PROJECT\x-anylabeling01\X-AnyLabeling-Server
..\.venv\Scripts\python.exe -m app.main
```

→ `7/8 model(s) loaded` 메시지 확인될 때까지 대기

### 터미널 2 - 클라이언트 시작

```powershell
D:\MYCLAUDE_PROJECT\x-anylabeling01\.venv_client\Scripts\xanylabeling.exe
```

### 클라이언트에서 서버 연결

1. 툴바의 **Remote-server** 버튼 클릭
2. URL: `http://localhost:8000`
3. API Key: 비워두기
4. Connect 클릭
5. 드롭다운에서 **Segment Anything 3** 선택

---

## 사용법

### 모드 1: 텍스트 프롬프트 (주요 기능)

1. 이미지 열기 (File → Open 또는 드래그 앤 드롭)
2. 왼쪽 패널의 **텍스트 입력창**에 찾고 싶은 물체 이름 입력
   - 단어: `car`, `person`, `tree`
   - 구문: `red car`, `person wearing a hat`
   - 설명: `traffic light on the pole`
   - 여러 개: `car, person, tree` (쉼표 구분)
3. **Send** 버튼 클릭
4. 결과 확인 → **Finish Object**로 확정

SAM3는 CLIP 기반 텍스트 인코더를 사용하므로 **영어 구문/설명**이 가장 잘 동작합니다.

### 모드 2: 시각 프롬프트 (박스 지정)

텍스트로 잘 안 잡히는 물체(특수 장비, 컨트롤 박스 등)에 유용합니다.

1. **Add Pos Rect** 클릭
2. 이미지에서 대상 물체를 **드래그로 사각형** 그리기 (포함할 영역)
3. (선택) **Add Neg Rect** → 제외할 영역 드래그
4. **Run** 클릭 → SAM3가 해당 영역을 정밀 세그멘테이션
5. **Finish Object** 클릭 → 어노테이션 확정
6. 다음 물체 반복

### 위젯 설명

| 버튼/슬라이더 | 기능 |
|---------------|------|
| **Send** | 텍스트 프롬프트 실행 |
| **Add Pos Rect** | 포함할 영역 박스 그리기 |
| **Add Neg Rect** | 제외할 영역 박스 그리기 |
| **Run** | 박스 프롬프트로 세그멘테이션 실행 |
| **Clear** | 모든 프롬프트/결과 초기화 |
| **Finish Object** | 현재 객체 확정, 다음 객체로 이동 |
| **Conf** 슬라이더 | 신뢰도 임계값 (기본 0.25, 높이면 정확한 것만, 낮추면 더 많이 검출) |
| **Mask Fineness** | 마스크 정밀도 (기본 10) |
| **Preserve Annotations** | 체크하면 기존 라벨 유지 |

### 회전 사각형 (OBB) 그리기

- 메뉴: **Edit → Create Rotation** (또는 단축키 `O`)
- 4개의 꼭짓점을 순서대로 클릭하여 회전된 사각형 생성
- 자동 감지: 드롭다운에서 **YOLO11n OBB** 모델 선택

---

## SAM3 전체 워크플로우

```
SAM3로 어노테이션 생성 (텍스트/박스 프롬프트로 빠르게 라벨링)
    ↓
라벨 데이터 저장 (JSON 파일)
    ↓
이 데이터로 YOLO 등 경량 모델 학습 (별도 과정)
    ↓
학습된 모델로 자동 인식
```

SAM3 자체가 학습하는 것이 아니라, SAM3는 **라벨링 도구**입니다.
박스/텍스트로 빠르게 어노테이션을 만들고, 그 데이터를 모아서 별도 모델(YOLO 등)을 학습시키는 것이 전체 워크플로우입니다.

---

## SAM2 vs SAM3 차이

| 항목 | SAM2 | SAM3 |
|------|------|------|
| 텍스트 프롬프트 | 지원 안 함 | CLIP 기반 텍스트 인코더 내장 |
| 시각 프롬프트 | 포인트/박스/마스크 | 포인트/박스/마스크 |
| 제로샷 인식 | 시각 프롬프트만 가능 | 텍스트로 물체 검색 가능 |
| 출시 | 2024년 | 2025년 11월 |

---

## 설정 파일

### 서버 모델 설정 (configs/models.yaml)

```yaml
enabled_models:
  - yolo11n
  - yolo11n_seg
  - yolo11n_pose
  - yolo11n_obb
  - yolo11n_track
  - segment_anything_3
  - pp_doclayout_v3
  - paddleocr_vl_1_5
```

### SAM3 모델 설정 (configs/auto_labeling/segment_anything_3.yaml)

```yaml
params:
  bpe_path: "D:/MYCLAUDE_PROJECT/x-anylabeling01/X-AnyLabeling-Server/bpe_simple_vocab_16e6.txt.gz"
  model_path: "D:/MYCLAUDE_PROJECT/x-anylabeling01/X-AnyLabeling-Server/sam3.pt"
  device: "cuda:0"
  conf_threshold: 0.25
  show_boxes: false      # true: 박스 표시, false: 외곽선만
  show_masks: true
  epsilon_factor: 0.001
```

**다른 PC로 옮길 때** `bpe_path`와 `model_path`의 절대경로를 새 PC에 맞게 수정해야 합니다.

---

## 설치된 패키지 버전 (참고)

| 패키지 | 서버 (.venv) | 클라이언트 (.venv_client) |
|--------|-------------|-------------------------|
| torch | 2.10.0+cu126 | 미설치 |
| torchvision | 0.25.0+cu126 | 미설치 |
| triton-windows | 3.6.0.post25 | 미설치 |
| x-anylabeling-cvhub | 3.3.9 | 3.3.9 |
| numpy | 2.3.5 | 1.26.4 |
| Python | 3.12 | 3.12 |

---

## 트러블슈팅

### DLL 초기화 오류 (WinError 1114)
서버와 클라이언트가 같은 venv를 사용하면 CUDA DLL 충돌 발생.
→ 반드시 `.venv`(서버)와 `.venv_client`(클라이언트)를 분리

### CUDA out of memory
이미지 해상도가 너무 크면 VRAM 부족 (61GB 요구 등).
→ 이미지를 1920x1080 이하로 리사이즈 후 사용

### Torch not compiled with CUDA enabled
서버의 PyTorch가 CPU 버전으로 바뀐 경우.
→ 재설치:
```powershell
.venv\Scripts\pip.exe install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

### No prompt provided 경고
텍스트 없이 Run 버튼을 클릭한 경우. 텍스트를 입력하고 Send를 누르거나, Add Pos Rect로 박스를 그린 후 Run 클릭.
