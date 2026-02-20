심층 기술 가이드: 미인식 객체 훈련을 위한 실무 파이프라인
본 장에서는 가장 추천하는 솔루션인 X-AnyLabeling을 중심으로, 실제 미인식 객체를 훈련하는 단계별(Step-by-Step) 기술 가이드를 제공한다.

5.1 1단계: 환경 설정 및 모델 준비 (Setup)
SAM 3의 훈련 기능을 활용하기 위해서는 단순 추론보다 높은 사양의 하드웨어 준비가 선행되어야 한다.

하드웨어: NVIDIA GPU (VRAM 12GB 이상 권장, 훈련 시 24GB 이상 권장). CUDA 12.x 드라이버 설치.

소프트웨어 설치:

Bash
# Python 3.10+ 환경 생성
conda create -n sam3_labeling python=3.10
conda activate sam3_labeling
# GPU 가속 지원 버전 설치
pip install "x-anylabeling-cvhub[cuda12]"
모델 로드: X-AnyLabeling 실행 후 좌측 사이드바의 'Brain' 아이콘을 클릭하여 모델 리스트를 연다. 'Segment Anything Models' 카테고리에서 **'SAM 3 (Large)'**를 선택한다. 최초 선택 시 모델 가중치가 자동으로 다운로드된다. 이때 인터넷 연결이 필요하지만, 이후에는 오프라인으로 작동한다.

5.2 2단계: 미인식 객체 데이터 구축 (Cold Start & Annotation)
훈련 데이터가 전무한 'Cold Start' 상황에서 SAM 3를 활용해 데이터를 구축하는 과정이다.

이미지 로드: 훈련할 대상 객체가 포함된 이미지 폴더를 연다.

텍스트 프롬프트 시도: Ctrl+A를 눌러 AI 모드를 켜고, 텍스트 프롬프트 창에 객체에 대한 설명을 입력한다 (예: "scratched surface"). SAM 3가 이를 인식한다면 운이 좋은 경우이며, 즉시 마스크가 생성된다.

시각적 프롬프트 및 수정: 텍스트로 인식이 안 되거나 부정확할 경우, 해당 객체 영역에 박스를 그리거나 점을 찍는다(Positive Point). 잘못 포함된 영역은 우클릭(Negative Point)으로 제거한다.

수동 보정: 생성된 마스크가 완벽하지 않다면 'Edit' 모드로 전환하여 폴리곤의 점을 이동시키거나 브러시로 다듬는다.

클래스 정의: 인식된 객체에 정확한 클래스 명(예: scratch_defect)을 할당하고 저장(Save)한다. 이 과정이 반복되면서 고품질의 'Ground Truth' 데이터가 쌓이게 된다.

5.3 3단계: 모델 훈련 및 루프 (Training Loop)
데이터가 일정량(초기 50~100장) 모였다면 훈련을 시작한다.

훈련 설정: UI 상단의 'Train' 아이콘을 클릭한다. X-AnyLabeling은 내부적으로 Ultralytics YOLO 트레이너를 호출한다.

모델 선택: 훈련할 모델 아키텍처를 선택한다 (예: yolov8m-seg). 여기서 중요한 점은 SAM 3 모델 자체를 미세 조정하는 옵션이 있다면 선택하되, 로컬 자원이 부족하다면 SAM 3로 만든 데이터로 YOLO-Seg 모델을 학습시키는 것이 현실적으로 더 빠르고 효율적이라는 점이다.   

파라미터 튜닝: Epochs(반복 횟수), Batch Size, Image Size 등을 설정한다. 미인식 객체의 특성에 따라 데이터 증강(Augmentation) 옵션을 켜거나 끈다.

학습 실행: 'Start Training'을 누르면 로컬 GPU에서 학습이 진행되며, 실시간으로 손실(Loss) 그래프가 표시된다.

모델 교체 및 검증: 학습이 끝나면 새로운 모델 파일(.pt 또는 .onnx)이 생성된다. 이를 X-AnyLabeling에 커스텀 모델로 로드한다. 이제 이전에 인식되지 않았던 객체가 새로운 모델을 통해 자동으로 인식되는지 확인한다. 인식률이 낮다면 2단계로 돌아가 데이터를 추가하고 다시 3단계를 반복한다(Active Learning Loop).