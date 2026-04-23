@echo off
REM ============================================================
REM VLM + SAM3 파이프라인 환경 설치 스크립트
REM 기존 .venv (서버), .venv_client (GUI)와 별도 환경
REM ============================================================

echo.
echo ============================================================
echo  VLM + SAM3 Railway Pipeline - Environment Setup
echo  GPU: RTX 3060 12GB (sequential model loading)
echo ============================================================
echo.

REM 1) 가상환경 생성
if not exist ".venv_vlm" (
    echo [1/5] 가상환경 생성: .venv_vlm
    python -m venv .venv_vlm
) else (
    echo [1/5] 가상환경 이미 존재: .venv_vlm
)

REM 활성화
call .venv_vlm\Scripts\activate.bat

REM 2) PyTorch + CUDA 12.6
echo.
echo [2/5] PyTorch + CUDA 12.6 설치...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

REM 3) 기본 의존성
echo.
echo [3/5] 기본 의존성 설치...
pip install -r vlm_sam3\requirements.txt

REM 4) SAM3 패키지 (X-AnyLabeling-Server에서 사용하는 것과 동일)
echo.
echo [4/5] SAM3 패키지 확인...
if exist "X-AnyLabeling-Server\sam3.pt" (
    echo   [OK] sam3.pt 모델 파일 확인됨
) else (
    echo   [WARN] sam3.pt 모델 파일이 없습니다!
    echo         다운로드: huggingface-cli download facebook/sam3 --local-dir X-AnyLabeling-Server
    echo         또는:    pip install huggingface_hub ^&^& python -c "from huggingface_hub import hf_hub_download; hf_hub_download('facebook/sam3', 'sam3.pt', local_dir='X-AnyLabeling-Server')"
)

REM 5) 설치 확인
echo.
echo [5/5] 설치 확인...
python -c "import torch; print(f'  PyTorch: {torch.__version__}'); print(f'  CUDA: {torch.cuda.is_available()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"})')"
python -c "import transformers; print(f'  Transformers: {transformers.__version__}')"
python -c "import cv2; print(f'  OpenCV: {cv2.__version__}')"
python -c "import supervision; print(f'  Supervision: {supervision.__version__}')" 2>nul || echo   Supervision: not installed (optional)

echo.
echo ============================================================
echo  설치 완료!
echo.
echo  사용법:
echo    .venv_vlm\Scripts\activate
echo    python vlm_sam3/pipeline.py
echo.
echo  또는:
echo    start_vlm_sam3.bat
echo ============================================================

pause
