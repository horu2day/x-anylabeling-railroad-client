# X-AnyLabeling + SAM3 개발환경 셋업 가이드

> 처음부터 (VSCode, Python, venv 없는 상태) 전체 환경 구성 절차

---

## 목차

1. [사전 요구사항](#1-사전-요구사항)
2. [Python 설치](#2-python-설치)
3. [VSCode 설치 및 설정](#3-vscode-설치-및-설정)
4. [프로젝트 클론 또는 열기](#4-프로젝트-클론-또는-열기)
5. [Server 환경 (.venv) 구성](#5-server-환경-venv-구성)
6. [Client 환경 (.venv_client) 구성](#6-client-환경-venv_client-구성)
7. [모델 파일 준비](#7-모델-파일-준비)
8. [서버 실행](#8-서버-실행)
9. [클라이언트 실행 및 서버 연결](#9-클라이언트-실행-및-서버-연결)
10. [VSCode 개발 설정](#10-vscode-개발-설정)
11. [동작 확인 (Health Check)](#11-동작-확인-health-check)
12. [트러블슈팅](#12-트러블슈팅)

---

## 1. 사전 요구사항

| 항목 | 요구 사항 | 비고 |
|------|----------|------|
| OS | Windows 10/11 64-bit | |
| GPU | NVIDIA GPU (CUDA 지원) | RTX 3060 이상 권장 |
| CUDA Toolkit | 12.6 | GPU 드라이버와 호환 필요 |
| RAM | 16GB 이상 | SAM3 모델 로드 시 ~8GB 사용 |
| 디스크 여유 | 20GB 이상 | 모델 파일 ~3.3GB + venv |
| 인터넷 | 필요 | pip 설치, 모델 다운로드 |

### CUDA 드라이버 확인

```powershell
nvidia-smi
```

출력 상단에 `CUDA Version: 12.x` 확인. 없으면 [NVIDIA 드라이버](https://www.nvidia.com/drivers) 최신버전 설치.

---

## 2. Python 설치

**Python 3.12.x (64-bit)** 필수 (3.10 이상이면 동작하나 3.12 권장)

1. [python.org/downloads](https://www.python.org/downloads/) 에서 **Python 3.12.x Windows installer (64-bit)** 다운로드
2. 설치 시 반드시 **"Add Python to PATH"** 체크
3. **"Install Now"** 클릭

설치 확인:

```powershell
python --version
# Python 3.12.x
```

---

## 3. VSCode 설치 및 설정

### VSCode 설치

1. [code.visualstudio.com](https://code.visualstudio.com/) 에서 다운로드
2. 설치 시 **"Add to PATH"**, **"Open with Code"** 체크 권장

### 필수 확장 프로그램 설치

VSCode 실행 후 Extensions (Ctrl+Shift+X) 에서 설치:

| 확장명 | ID | 용도 |
|--------|-----|------|
| Python | `ms-python.python` | Python 지원 |
| Pylance | `ms-python.vscode-pylance` | IntelliSense |
| Ruff | `charliermarsh.ruff` | Linting |
| YAML | `redhat.vscode-yaml` | configs/*.yaml 편집 |
| GitLens | `eamodio.gitlens` | Git 히스토리 |
| REST Client | `humao.rest-client` | API 테스트 (선택) |

---

## 4. 프로젝트 클론 또는 열기

### 기존 프로젝트 열기

```powershell
code D:\MYCLAUDE_PROJECT\x-anylabeling01
```

### 새로 클론하는 경우

```powershell
cd D:\MYCLAUDE_PROJECT
git clone <repo_url> x-anylabeling01
cd x-anylabeling01
```

> X-AnyLabeling-Server 서브 디렉토리가 없으면:
> ```powershell
> git clone https://github.com/CVHub520/X-AnyLabeling X-AnyLabeling-Server
> ```

---

## 5. Server 환경 (.venv) 구성

> **서버용 가상환경** - PyTorch(CUDA), FastAPI, SAM3, YOLO 등 포함
> 프로젝트 루트(`D:\MYCLAUDE_PROJECT\x-anylabeling01`)에서 실행

### Step 1. 가상환경 생성

```powershell
cd D:\MYCLAUDE_PROJECT\x-anylabeling01
python -m venv .venv
```

### Step 2. PyTorch (CUDA 12.6) 설치

> 일반 `pip install torch`는 CPU 버전이 설치됨. 반드시 CUDA 인덱스 사용.

```powershell
.venv\Scripts\pip.exe install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

설치 확인:

```powershell
.venv\Scripts\python.exe -c "import torch; print('CUDA:', torch.cuda.is_available(), '| Version:', torch.version.cuda)"
# CUDA: True | Version: 12.6
```

### Step 3. 나머지 패키지 설치

```powershell
.venv\Scripts\pip.exe install -r requirements.txt
```

### Step 4. X-AnyLabeling-Server 패키지 설치 (editable)

```powershell
.venv\Scripts\pip.exe install -e X-AnyLabeling-Server
```

### 전체 설치 시간

약 10~20분 (인터넷 속도에 따라 상이)

---

## 6. Client 환경 (.venv_client) 구성

> **클라이언트용 별도 가상환경** - PyQt5 GUI 전용, torch 없음
> 서버 venv와 분리하는 이유: torch DLL이 같이 있으면 **WinError 1114** 발생

### Step 1. 가상환경 생성

```powershell
cd D:\MYCLAUDE_PROJECT\x-anylabeling01
python -m venv .venv_client
```

### Step 2. 패키지 설치

```powershell
.venv_client\Scripts\pip.exe install -r requirements-xanylabeling.txt
```

> `requirements-xanylabeling.txt` 핵심 내용:
> ```
> x-anylabeling-cvhub>=3.3.9
> numpy<=1.26.4    # 반드시 1.26.4 이하
> # torch/torchvision 없음 (의도적)
> ```

### 설치 확인

```powershell
.venv_client\Scripts\xanylabeling.exe --version
```

---

## 7. 모델 파일 준비

### SAM3 모델 (sam3.pt, ~3.3GB)

**방법 A: HuggingFace CLI로 다운로드**

```powershell
# huggingface_hub 설치 (서버 venv에 이미 포함됨)
.venv\Scripts\python.exe -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='facebook/sam3',
    filename='sam3.pt',
    local_dir='X-AnyLabeling-Server'
)
print('다운로드 완료')
"
```

**방법 B: 수동 다운로드**

1. [huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3) 접속
2. `sam3.pt` 파일 다운로드
3. `D:\MYCLAUDE_PROJECT\x-anylabeling01\X-AnyLabeling-Server\` 에 저장

### YOLO 모델 (선택, 없으면 자동 다운로드)

```
X-AnyLabeling-Server/
├── sam3.pt                        ← SAM3 (필수, 수동)
├── yolo11n.pt                     ← YOLO detection (자동 다운로드)
├── yolo11n-seg.pt                 ← YOLO segmentation (자동 다운로드)
└── bpe_simple_vocab_16e6.txt.gz   ← CLIP 어휘 (이미 포함)
```

### 모델 경로 설정 확인

`X-AnyLabeling-Server/configs/auto_labeling/segment_anything_3.yaml` 열어서 경로 확인:

```yaml
params:
  bpe_path: "D:/MYCLAUDE_PROJECT/x-anylabeling01/X-AnyLabeling-Server/bpe_simple_vocab_16e6.txt.gz"
  model_path: "D:/MYCLAUDE_PROJECT/x-anylabeling01/X-AnyLabeling-Server/sam3.pt"
  device: "cuda:0"
```

> 경로가 다르면 절대경로로 수정

---

## 8. 서버 실행

### 방법 A: 배치 파일 (권장)

```powershell
start_server.bat
```

### 방법 B: 직접 실행

```powershell
cd D:\MYCLAUDE_PROJECT\x-anylabeling01\X-AnyLabeling-Server
..\\.venv\Scripts\python.exe -m app.main
```

### 방법 C: 커스텀 옵션

```powershell
.venv\Scripts\python.exe X-AnyLabeling-Server/app/cli.py \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 1 \
  --config X-AnyLabeling-Server/configs/server.yaml \
  --models-config X-AnyLabeling-Server/configs/models.yaml
```

### 서버 정상 기동 로그 확인

```
INFO:     Started server process [XXXX]
INFO:     Waiting for application startup.
INFO:     Loading model: segment_anything_3 ...
INFO:     Model loaded: segment_anything_3
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

---

## 9. 클라이언트 실행 및 서버 연결

### 방법 A: 배치 파일

```powershell
start_client.bat
```

### 방법 B: 직접 실행

```powershell
.venv_client\Scripts\xanylabeling.exe
```

### 서버 연결 설정

1. 툴바에서 **"Remote-server"** 버튼 클릭 (또는 메뉴 → Model → Remote Server)
2. 설정 입력:
   - **URL**: `http://localhost:8000`
   - **API Key**: 비워두기 (기본 설정은 인증 없음)
3. **Connect** 클릭
4. 모델 드롭다운에서 **"Segment Anything 3"** 선택

### SAM3 사용법

| 모드 | 방법 |
|------|------|
| 텍스트 프롬프트 | 영어로 객체 설명 입력 후 Send (예: `railway signal`, `catenary wire`) |
| 박스 프롬프트 | Add Positive Rect 클릭 후 객체 위에 사각형 드로우 |
| 포인트 프롬프트 | 클릭으로 포함/제외 포인트 지정 |

---

## 10. VSCode 개발 설정

### Python 인터프리터 선택

`Ctrl+Shift+P` → **"Python: Select Interpreter"**:
- 서버 개발: `.venv\Scripts\python.exe` 선택
- 클라이언트 개발: `.venv_client\Scripts\python.exe` 선택

### 권장 workspace 설정 (.vscode/settings.json)

```json
{
    "python.defaultInterpreterPath": "${workspaceFolder}/.venv/Scripts/python.exe",
    "python.terminal.activateEnvironment": true,
    "editor.formatOnSave": true,
    "[python]": {
        "editor.defaultFormatter": "charliermarsh.ruff"
    },
    "ruff.enable": true,
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        ".venv": true,
        ".venv_client": true
    }
}
```

### 디버그 설정 (.vscode/launch.json)

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Server: Run FastAPI",
            "type": "debugpy",
            "request": "launch",
            "module": "app.main",
            "cwd": "${workspaceFolder}/X-AnyLabeling-Server",
            "python": "${workspaceFolder}/.venv/Scripts/python.exe",
            "env": {
                "PYTHONPATH": "${workspaceFolder}/X-AnyLabeling-Server"
            }
        }
    ]
}
```

---

## 11. 동작 확인 (Health Check)

### 브라우저에서

`http://localhost:8000/health` 접속 → 아래와 같이 응답:

```json
{
  "status": "healthy",
  "models_loaded": 8,
  "timestamp": "2026-03-12T..."
}
```

### PowerShell에서

```powershell
# 서버 상태
Invoke-RestMethod -Uri "http://localhost:8000/health"

# 모델 목록
Invoke-RestMethod -Uri "http://localhost:8000/v1/models"
```

### Python에서 테스트

```python
import requests, base64

# 테스트 이미지 (임의)
with open("test.jpg", "rb") as f:
    img_b64 = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()

resp = requests.post("http://localhost:8000/v1/predict", json={
    "model": "segment_anything_3",
    "image": img_b64,
    "params": {"text_prompt": "person", "conf_threshold": 0.25}
})
print(resp.json())
```

---

## 12. 트러블슈팅

### CUDA 사용 불가 (CUDA: False)

```powershell
# torch가 CPU 버전인지 확인
.venv\Scripts\python.exe -c "import torch; print(torch.__version__)"
# 출력에 +cu126 없으면 CUDA 버전 아님

# 재설치
.venv\Scripts\pip.exe uninstall torch torchvision -y
.venv\Scripts\pip.exe install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

### WinError 1114 (DLL 오류)

클라이언트 venv에 torch가 설치된 경우 발생:

```powershell
# 클라이언트 venv 초기화 후 재설치
rm -rf .venv_client
python -m venv .venv_client
.venv_client\Scripts\pip.exe install -r requirements-xanylabeling.txt
# requirements-xanylabeling.txt 에 torch 없어야 함
```

### numpy 버전 오류 (클라이언트)

```powershell
.venv_client\Scripts\pip.exe install "numpy<=1.26.4" --force-reinstall
```

### sam3.pt 파일 없음

```
FileNotFoundError: .../X-AnyLabeling-Server/sam3.pt
```

→ [7. 모델 파일 준비](#7-모델-파일-준비) 섹션 참고하여 다운로드

### 포트 충돌 (Address already in use)

```powershell
# 8000 포트 사용 프로세스 확인
netstat -ano | findstr :8000

# 프로세스 종료 (PID 확인 후)
taskkill /PID <PID번호> /F
```

### 모델 로딩 느림

SAM3 첫 로딩은 GPU 메모리 할당으로 **30초~2분** 소요 정상.
로그에 `Model loaded: segment_anything_3` 뜰 때까지 대기.

---

## 프로젝트 디렉토리 구조 요약

```
D:\MYCLAUDE_PROJECT\x-anylabeling01\
├── .venv\                          ← 서버 가상환경 (Python 3.12, torch+CUDA)
├── .venv_client\                   ← 클라이언트 가상환경 (Python 3.12, GUI전용)
├── X-AnyLabeling-Server\           ← FastAPI 서버 소스
│   ├── app\                        ← 애플리케이션 코드
│   ├── configs\                    ← 서버/모델 설정 YAML
│   ├── sam3.pt                     ← SAM3 모델 (수동 다운로드 필요)
│   └── bpe_simple_vocab_16e6.txt.gz
├── src\                            ← 커스텀 소스 (oblique_mosaic 등)
├── requirements.txt                ← 서버 의존성
├── requirements-xanylabeling.txt   ← 클라이언트 의존성
├── start_server.bat                ← 서버 시작 스크립트
├── start_client.bat                ← 클라이언트 시작 스크립트
└── DEV_SETUP.md                    ← 이 문서
```

---

## 빠른 시작 체크리스트

- [ ] Python 3.12 설치 + PATH 등록
- [ ] VSCode 설치 + Python 확장 설치
- [ ] `.venv` 생성 + torch(CUDA) + requirements.txt 설치
- [ ] `.venv_client` 생성 + requirements-xanylabeling.txt 설치
- [ ] `sam3.pt` 파일을 `X-AnyLabeling-Server/` 에 위치
- [ ] `segment_anything_3.yaml` 에서 model_path 절대경로 확인
- [ ] `start_server.bat` 실행 → 로그에서 모델 로딩 확인
- [ ] `start_client.bat` 실행 → http://localhost:8000 으로 서버 연결
- [ ] `http://localhost:8000/health` 에서 status: healthy 확인
