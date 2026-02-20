# 철도 시설물 커스텀 YOLO 학습 가이드

SAM3 박스 프롬프트로 어노테이션을 생성하고, YOLO 커스텀 모델을 학습하는 전체 워크플로우.

## 전체 흐름

```
드론 이미지 수집
    ↓
SAM3 박스 프롬프트로 어노테이션 (라벨링)
    ↓
YOLO 형식으로 내보내기
    ↓
YOLO 커스텀 모델 학습
    ↓
학습된 모델로 자동 탐지
```

---

## STEP 1: 이미지 수집 및 폴더 구성

```
D:\railway_dataset\
├── images\
│   ├── img001.jpg
│   ├── img002.jpg
│   └── ...
└── labels\        ← 나중에 자동 생성됨
```

클라이언트에서 **File → Open Dir** → `images` 폴더 선택

---

## STEP 2: SAM3로 박스 프롬프트 어노테이션

### 컨트롤박스 라벨링 방법

1. 이미지에서 컨트롤박스가 보이면
2. 왼쪽 패널 **Add Pos Rect** 클릭
3. 컨트롤박스 위에 드래그로 사각형 그리기
4. **Run** 클릭 → SAM3가 정밀 외곽선 자동 생성
5. 라벨명 입력 창이 뜨면 `control_box` 입력
6. **Finish Object** 클릭 → 확정
7. 같은 이미지에 여러 개면 2~6 반복
8. **Ctrl+S** 저장 → `.json` 파일 자동 생성

### 추가 팁

- 결과가 마음에 안 들면 **Add Neg Rect**로 제외 영역 추가 후 **Run** 재실행
- **Mask Fineness** 슬라이더를 높이면 더 정밀한 외곽선
- **Preserve Annotations** 체크하면 이전 라벨 유지하면서 추가 가능

### 권장 클래스명 (영어, 소문자, 언더스코어)

```
rail
railway_sleeper
catenary_pole
control_box
insulator
rail_switch
signal_light
```

---

## STEP 3: 충분한 데이터 확보

| 클래스 | 최소 권장 이미지 수 |
|--------|-------------------|
| 단순한 형태 (레일, 침목) | 50장 이상 |
| 중간 복잡도 (전철주, 교량) | 100장 이상 |
| 복잡/소형 (컨트롤박스, 애자) | 200장 이상 |

> SAM3 박스 프롬프트를 쓰면 수동 폴리곤 작업보다 **5~10배 빠르게** 라벨링 가능

---

## STEP 4: YOLO 형식으로 내보내기

1. 라벨링 완료 후 메뉴: **File → Export → Export YOLO-Seg Annotations**
2. 출력 폴더 선택: `D:\railway_dataset\yolo_export\`
3. 클래스 파일 선택 또는 자동 생성 확인
4. **Export** 클릭

내보내기 결과:
```
D:\railway_dataset\yolo_export\
├── images\
│   ├── img001.jpg
│   └── img002.jpg
├── labels\
│   ├── img001.txt      ← YOLO 형식 좌표
│   └── img002.txt
└── classes.txt         ← 클래스 목록
```

`labels/img001.txt` 내용 예시:
```
0 0.452 0.312 0.478 0.298 0.501 0.334 0.477 0.348   ← control_box 폴리곤
1 0.123 0.456 0.234 0.456 0.234 0.512 0.123 0.512   ← catenary_pole 폴리곤
```

---

## STEP 5: YOLO 학습 데이터셋 구성

내보낸 폴더를 train/val로 분리:

```
D:\railway_dataset\yolo_train\
├── dataset.yaml
├── train\
│   ├── images\   ← 전체의 80%
│   └── labels\
└── val\
    ├── images\   ← 전체의 20%
    └── labels\
```

`dataset.yaml` 파일 생성:

```yaml
path: D:/railway_dataset/yolo_train
train: train/images
val: val/images

nc: 7   # 클래스 수
names:
  0: rail
  1: railway_sleeper
  2: catenary_pole
  3: control_box
  4: insulator
  5: rail_switch
  6: signal_light
```

---

## STEP 6: YOLO 커스텀 학습

서버 환경(.venv)에서 실행:

```powershell
D:\MYCLAUDE_PROJECT\x-anylabeling01\.venv\Scripts\python.exe -c "
from ultralytics import YOLO
model = YOLO('yolo11n-seg.pt')
model.train(
    data='D:/railway_dataset/yolo_train/dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=8,
    device=0,
    project='D:/railway_dataset/runs',
    name='railway_seg'
)
"
```

학습 완료 후 모델 위치:
```
D:\railway_dataset\runs\railway_seg\weights\best.pt
```

---

## STEP 7: 학습된 모델을 클라이언트에 로드

**클라이언트에서 로컬 모델 로드:**
1. 모델 드롭다운 → **Load Custom Model**
2. `best.pt` 파일 선택
3. 이후 이미지에서 자동으로 컨트롤박스, 애자 등을 탐지

---

## 단계별 소요 시간 요약

| 단계 | 작업 | 소요 시간 |
|------|------|----------|
| 1 | 이미지 폴더 구성 | 10분 |
| 2 | SAM3 박스 프롬프트로 라벨링 | 이미지당 1~3분 |
| 3 | 200장 라벨링 완료 | 3~6시간 |
| 4 | YOLO-Seg 내보내기 | 5분 |
| 5 | dataset.yaml 작성 | 5분 |
| 6 | YOLO 학습 (GPU) | 30분~2시간 |
| 7 | 모델 등록 및 테스트 | 10분 |

---

## 참고: 바운딩박스(HBB) vs 세그멘테이션(Seg) 선택 기준

| 용도 | 권장 형식 | 명령어 |
|------|----------|--------|
| 단순 탐지 (있다/없다) | YOLO-HBB | `YOLO('yolo11n.pt')` |
| 외곽선 필요 (면적, 형태 분석) | YOLO-Seg | `YOLO('yolo11n-seg.pt')` |
| 회전된 객체 (드론 항공뷰) | YOLO-OBB | `YOLO('yolo11n-obb.pt')` |

드론 항공영상에서 철도 시설물은 **회전되어 보이는 경우**가 많으므로 OBB도 고려하세요.
Export 시 **Export YOLO-Obb Annotations** 선택 후 `yolo11n-obb.pt`로 학습하면 됩니다.
