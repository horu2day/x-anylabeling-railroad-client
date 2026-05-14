# 전철주(Catenary Pole) 라멘형 구조 검출 알고리즘 시도 이력

## 문제 정의

**목표**: 드론 사선 촬영 이미지에서 SAM3.1/YOLO가 생성한 catenary_pole 폴리곤들을 분석하여
라멘형(門자형) 전철주 — 수직 기둥 2개 + 수평 빔 1-2개 — 를 식별한다.

**입력**
- 원본 드론 이미지 (4K급 사선 촬영)
- YOLO 포맷 라벨 `.txt` (class 1 = catenary_pole, 정규화된 폴리곤 좌표)

**핵심 난점**
- SAM3.1 폴리곤 경계가 지글지글함(noisy) → 마스크 품질이 낮음
- 드론 사선 촬영 → 수직 기둥도 방사형으로 기울어져 보임
- 일부 기둥은 빔에 가려짐 / 도로교에 가려짐 → 폴리곤 미검출
- 하나의 전철주가 여러 폴리곤 조각으로 분리 검출됨
- 여러 전철주가 하나의 그룹으로 합쳐지기도 함

---

## 데이터 현황 (DJI_20260306100900_0034 기준)

| 그룹 | 판별 결과 | 비고 |
|------|-----------|------|
| C10 | 라멘 ✓ | 명확히 보임 |
| C13 | 라멘 ✓ | 명확히 보임 |
| C14 | 라멘 ✓ | 명확히 보임 |
| C25 | 라멘 ✓ | 명확히 보임 |
| C20 | 라멘 ✓ | 명확히 보임 |
| C22 | C20의 부속 소형 | 작음 |
| C12 | 라멘이나 검출 불완전 | 카메라 정면 방향, 빔과 기둥 각도 거의 동일 → 뒤쪽 기둥 미검출 |
| C19 | 라멘이나 검출 불완전 | 도로교에 가림, 35번 폴리곤(C18 소속)이 원래 C19에 속해야 함 |

---

## 시도한 접근 방식

### 1. Skeleton + deg≥3 분기점 검출

**방법**
1. 그룹 내 폴리곤 union 마스크 → `dilate(11) / erode(7)` → `cv2.ximgproc.thinning()`
2. 8방향 이웃 카운트(deg map) → `deg ≥ 3` 픽셀을 분기점으로 판정
3. `cluster_centroids`로 클러스터링 → 분기점 위치 추출

**결과**: 실패  
폴리곤 경계 노이즈 때문에 skeleton 선을 따라 spurious deg≥3 픽셀이 수십 개 발생.  
실제 T/X junction 2개가 artifact 속에 묻힘.

---

### 2. Crossing Number(CN) 필터

**방법**  
deg≥3 픽셀 중 8방향 시계 순서로 0→1 전환 횟수(CN)를 계산:
- CN = 1: endpoint
- CN = 2: 직선상 픽셀 (stair-step artifact 포함)
- CN ≥ 3: 위상학적 진짜 분기점

`CN ≥ 3`만 진짜 branch로 분류, `deg≥3 & CN<2`는 spurious로 분류.

**결과**: 실패  
모든 spurious deg≥3 픽셀이 CN=3으로 나옴.  
→ 노이즈가 stair-step이 아닌 **spur(짧은 가지)**를 만들고 있었기 때문.  
spur 기저점은 위상학적으로 진짜 분기점(CN=3)이라 필터링 불가.

---

### 3. Spur Pruning (현재 적용 중)

**방법**
1. skeleton에서 CN≥3 픽셀(branch) 제거 → arm component 분리
2. 각 arm component 픽셀 수 < `min_arm_px(=30)` 이면 spur → 제거
3. 제거된 arm의 centroid를 분홍 원으로 표시

**결과**: 부분 개선  
짧은 spur는 제거되나, 폴리곤 품질 자체가 낮아 여전히 spurious branch 발생.  
spur 제거 후에도 실제 junction 판별이 불안정함.

**파라미터**: `min_arm_px=30`

---

### 4. Top-down Tree Topology (현재 적용 중)

**방법**  
endpoint들을 topology로 연결하는 알고리즘:
1. 최상단 endpoint T 선택 (y 최솟값)
2. T에서 left-most, right-most endpoint로 연결
3. 더 긴 쪽 끝점에서 y+ 방향 미연결 노드로 체이닝

**결과**: 부분 동작  
라멘형 단순 구조에선 그럴듯하게 연결되나, spurious branch/endpoint가 섞이면 오결합 발생.

---

### 5. Virtual Endpoint 보완

**방법**  
endpoint가 3개만 검출된 경우(정상 라멘은 4개), bbox 방향 분석으로 누락된 방향을 찾아  
가장 극단에 위치한 branch를 virtual endpoint로 추가.

**결과**: 부분 동작  
branch 검출이 노이즈투성이일 때 올바른 branch를 찾지 못해 오동작.

---

### 6. 폴리곤 수직/수평 분류 — 절대 각도 방식

**방법**  
`cv2.minAreaRect`로 장축 각도 계산 → 45° 기준 V/H 분류.

**결과**: 실패  
드론 사선 촬영에서 수직 기둥도 이미지 내에서 기울어져 보이므로  
절대 각도 기준으로는 모든 폴리곤이 V로 분류됨.

---

### 7. 폴리곤 수직/수평 분류 — Radial 방향 방식 (현재 적용 중)

**방법**  
이미지 중심(W/2, H/2)에서 폴리곤 중심 방향을 radial 벡터로 삼고,  
minAreaRect 장축과 radial 벡터 사이의 cos 유사도로 분류:
- `cos_sim > 0.7` → **V** (수직 기둥: radial 방향으로 기울어짐)
- `cos_sim ≤ 0.7` → **H** (수평 빔: radial ⊥ 방향)

**결과**: 양호  
V/H 분포가 명확히 분리됨(V: ~40개, H: ~22개).  
V+H 혼재 그룹을 ★라멘? 후보로 자동 태깅.

**한계**: threshold 0.7은 경험적 값. poly 7(0.621), poly 14(0.683) 등 경계선 케이스 존재.

---

## 현재 파이프라인 구조

```
YOLO .txt 라벨 (class 1 필터)
  ↓
폴리곤 파싱 → minAreaRect radial 분류 (V/H/?)
  ↓
union mask → dilate(11)/erode(7) → thinning → skeleton
  ↓
Spur pruning (min_arm_px=30) → skel_pruned
  ↓
connectedComponents → 그룹(comp) 분리
  ↓
[그룹별]
  - branch 검출 (deg≥3, cluster_radius=7)
  - endpoint 검출 (deg=1, cluster_radius=5)
  - virtual endpoint 보완 (endpoint=3인 경우)
  - Top-down tree 연결선
  ↓
출력: 그룹별 색상 폴리곤 + Comp번호 표시
```

---

## 근본적 문제점 요약

| 문제 | 원인 | 영향 |
|------|------|------|
| Spurious branch | 폴리곤 경계 noisy → spur 발생 | Junction 검출 신뢰도 낮음 |
| 그룹 병합 | 인접 전철주 폴리곤이 skeleton으로 연결됨 | C10, C20 등이 과도하게 큰 그룹 |
| 그룹 분리 | 하나의 전철주 폴리곤이 skeleton 단절 | 일부 기둥이 별도 그룹으로 분류 |
| 폴리곤 미검출 | 가림/각도 등으로 SAM이 검출 실패 | C12 뒤쪽 기둥, C19 기둥 누락 |

---

## 미시도 방향 (참고)

- 폴리곤 smoothing 전처리 후 skeleton 재계산
- skeleton 대신 폴리곤 proximity graph (거리 기반 인접성)로 그룹핑
- V/H 분류 + bbox 공간 관계(위/아래/옆)로 라멘 구조 직접 판정
- 이미지에서 vanishing point 추정 → 더 정확한 수직 방향 기준 산출
