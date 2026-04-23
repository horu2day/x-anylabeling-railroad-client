@echo off
REM VLM + SAM3 파이프라인 실행
REM 기본: Florence-2 + SAM3, 순차 로딩 모드

call .venv_vlm\Scripts\activate.bat

echo ============================================================
echo  VLM + SAM3 Railway Object Detection Pipeline
echo  Mode: %1
echo ============================================================

if "%1"=="" (
    python vlm_sam3/pipeline.py --config vlm_sam3/config.yaml
) else (
    python vlm_sam3/pipeline.py --config vlm_sam3/config.yaml %*
)

pause
