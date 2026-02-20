# 철도 드론영상 탐지 오브젝트 목록

드론 항공영상에서 SAM3 등 AI 모델로 탐지 가능한 철도 시설물 분류

---

## 1. 궤도 (Track Components)

| 한국어 | 영어 (프롬프트) | 설명 | 항공뷰 특징 | 난이도 |
|--------|----------------|------|------------|--------|
| 철로 / 레일 | `rail`, `railroad track` | 강철 레일 2줄 | 평행한 은색/갈색 선 | 쉬움 |
| 침목 | `railway sleeper`, `railroad tie` | 레일을 지지하는 가로 구조물 | 레일 사이 규칙적인 가로선 | 쉬움 |
| 자갈 도상 | `ballast`, `gravel bed` | 침목 아래 자갈층 | 회색 자갈 텍스처 | 중간 |
| 콘크리트 도상 | `concrete track bed` | 콘크리트로 된 도상 | 회색 매끈한 면 | 중간 |
| 분기기 / 선로전환기 | `railway switch`, `turnout`, `rail junction` | 선로 방향을 바꾸는 장치 | Y자 분기 형태 | 중간 |
| 레일 이음매 | `rail joint`, `fishplate` | 레일 연결 부위 | 레일 위 작은 금속판 | 어려움 |
| 레일 체결장치 | `rail fastener`, `rail clip` | 레일과 침목 고정 장치 | 침목 위 작은 금속물 | 어려움 |

---

## 2. 전기 시설 (Electrical Infrastructure)

| 한국어 | 영어 (프롬프트) | 설명 | 항공뷰 특징 | 난이도 |
|--------|----------------|------|------------|--------|
| 전철주 | `electric pole`, `catenary pole`, `overhead line pole` | 가선을 지지하는 기둥 | 일정 간격의 세로 기둥 | 쉬움 |
| 가선 / 전차선 | `catenary wire`, `overhead wire`, `contact wire` | 전동차에 전기 공급하는 전선 | 레일 위 가느다란 선 | 어려움 |
| 급전선 | `feeder cable`, `feeder wire` | 변전소에서 가선으로 전력 공급 | 굵은 전선 | 어려움 |
| 지지애자 | `insulator`, `suspension insulator` | 전선 절연 및 지지 | 전철주 상단 흰색/갈색 물체 | 어려움 |
| 가선 브래킷 | `catenary bracket`, `wire support arm` | 전철주에서 가선을 지지하는 팔 | 전철주에서 수평으로 뻗은 구조 | 중간 |
| 변전소 | `substation`, `railway substation` | 전력 공급 시설 | 큰 건물 + 변압기 | 쉬움 |
| 변압기 | `transformer` | 전압 변환 장치 | 박스형 대형 장비 | 중간 |

---

## 3. 신호 / 제어 시설 (Signaling & Control)

| 한국어 | 영어 (프롬프트) | 설명 | 항공뷰 특징 | 난이도 |
|--------|----------------|------|------------|--------|
| 철도 신호기 | `railway signal`, `signal light` | 열차 운행 신호 | 기둥 위 등화 장치 | 중간 |
| 컨트롤 박스 | `control box`, `relay box`, `trackside cabinet` | 신호/전기 제어 박스 | 소형 금속 박스 | 중간 |
| 신호 기계실 | `signal room`, `relay room` | 신호 장비 수용 건물 | 소형 건물 | 쉬움 |
| 접속함 / 단자함 | `junction box`, `terminal box` | 케이블 접속 박스 | 작은 박스형 | 어려움 |
| 케이블 트레이 | `cable tray`, `cable duct` | 케이블 배선 지지대 | 선로 옆 줄 형태 | 중간 |
| ATC 지상자 | `ATC transponder`, `balise` | 자동열차제어 지상 장치 | 레일 사이 작은 노란 박스 | 어려움 |
| 열차 검지기 | `train detector`, `axle counter` | 열차 통과 감지 장치 | 레일 근처 소형 장비 | 어려움 |

---

## 4. 건널목 / 안전 시설 (Level Crossings & Safety)

| 한국어 | 영어 (프롬프트) | 설명 | 항공뷰 특징 | 난이도 |
|--------|----------------|------|------------|--------|
| 건널목 차단기 | `level crossing barrier`, `crossing gate` | 건널목 차량 통제 | 도로와 교차하는 줄무늬 바 | 쉬움 |
| 건널목 | `level crossing`, `railroad crossing` | 도로와 선로 교차 지점 | 선로와 도로 교차 패턴 | 쉬움 |
| 차막이 / 차단장치 | `buffer stop`, `end of track` | 선로 끝 충돌 방지 | 선로 끝 구조물 | 중간 |
| 방호울타리 | `safety fence`, `rail fence` | 선로 침범 방지 울타리 | 선로 양옆 줄 형태 | 중간 |
| 경고표지 | `warning sign`, `track sign` | 각종 경고/안내 표지 | 작은 표지판 | 어려움 |

---

## 5. 토목 구조물 (Civil Structures)

| 한국어 | 영어 (프롬프트) | 설명 | 항공뷰 특징 | 난이도 |
|--------|----------------|------|------------|--------|
| 철도교 / 교량 | `railway bridge`, `rail viaduct` | 하천/도로 위 교량 | 다리 구조 + 선로 | 쉬움 |
| 터널 입구 | `tunnel portal`, `tunnel entrance` | 터널 갱구 | 반원형 콘크리트 | 중간 |
| 옹벽 | `retaining wall` | 토사 유입 방지벽 | 선로 옆 콘크리트벽 | 중간 |
| 배수로 / 측구 | `drainage ditch`, `side ditch` | 우수 배수 시설 | 선로 옆 홈 형태 | 중간 |
| 암거 | `culvert` | 도로/하천 횡단 배수관 | 선로 아래 원통/박스 | 어려움 |
| 노반 | `trackbed`, `subgrade` | 선로 기초 토공 | 선로 주변 정형화된 토지 | 어려움 |

---

## 6. 역 시설 (Station Facilities)

| 한국어 | 영어 (프롬프트) | 설명 | 항공뷰 특징 | 난이도 |
|--------|----------------|------|------------|--------|
| 승강장 | `platform`, `station platform` | 여객 승하차 시설 | 선로 옆 긴 구조물 | 쉬움 |
| 역사 건물 | `station building` | 역 주 건물 | 큰 건물 | 쉬움 |
| 승강장 지붕 | `platform canopy`, `station roof` | 승강장 덮개 | 흰색/금속 지붕 구조 | 쉬움 |
| 육교 / 과선교 | `footbridge`, `pedestrian overpass` | 선로 위 보행자 통로 | 선로 가로지르는 교량 | 쉬움 |

---

## 7. 유지보수 장비 (Maintenance Equipment)

| 한국어 | 영어 (프롬프트) | 설명 | 항공뷰 특징 | 난이도 |
|--------|----------------|------|------------|--------|
| 궤도 작업차 | `maintenance vehicle`, `track inspection car` | 선로 유지보수 차량 | 선로 위 특수 차량 | 중간 |
| 고소작업차 | `overhead line maintenance vehicle` | 가선 유지보수 차량 | 크레인형 차량 | 중간 |
| 자재 적치장 | `material storage`, `depot` | 자재 보관 구역 | 정형화된 보관 구역 | 쉬움 |

---

## 8. 이상 상태 (Anomalies / Defects)

| 한국어 | 영어 (프롬프트) | 설명 | 항공뷰 특징 | 난이도 |
|--------|----------------|------|------------|--------|
| 선로 침수 | `flooded track`, `waterlogged rail` | 선로 물 고임 | 선로 위 반사 수면 | 중간 |
| 토사 유입 | `landslide`, `soil intrusion` | 사면 토사가 선로 침범 | 선로 위 갈색 토사 | 중간 |
| 식생 침범 | `vegetation encroachment`, `overgrown track` | 잡초/나무가 선로 침범 | 선로 위/옆 녹색 식물 | 중간 |
| 이물질 | `debris`, `foreign object on track` | 선로 위 이물질 | 선로 위 비정형 물체 | 어려움 |
| 적설 | `snow on track` | 선로 위 적설 | 흰색으로 덮인 선로 | 쉬움 |
| 레일 파손 | `broken rail`, `damaged rail` | 레일 균열/파손 | 레일 연속성 단절 | 어려움 |

---

## SAM3 프롬프트 추천

### 기본 탐지 세트
```
rail, railway sleeper, electric pole, control box, catenary wire
```

### 안전점검용
```
flooded track, vegetation encroachment, debris, landslide, broken rail
```

### 시설물 점검용
```
catenary pole, insulator, signal light, relay box, cable tray, junction box
```

### 구조물 점검용
```
railway bridge, retaining wall, tunnel portal, drainage ditch, culvert
```

---

## 탐지 난이도 요약

| 난이도 | 오브젝트 | 이유 |
|--------|----------|------|
| **쉬움** | 레일, 침목, 전철주, 교량, 건널목, 승강장 | 크고 뚜렷한 형태, 규칙적 패턴 |
| **중간** | 분기기, 신호기, 컨트롤박스, 옹벽, 배수로 | 크기가 작거나 배경과 유사 |
| **어려움** | 레일 체결장치, 애자, 지상자, ATC발리스 | 매우 작거나 선로와 색상 유사 |

---

## 드론 촬영 권장 조건

| 항목 | 권장값 |
|------|--------|
| 비행 고도 | 30~80m (시설물 점검), 80~150m (노선 전체) |
| 해상도 | 4K 이상 (소형 시설물 탐지 시) |
| GSD | 2cm/px 이하 (레일 체결장치 탐지 시) |
| 촬영 시간 | 맑은 날 오전 10시~오후 2시 (그림자 최소화) |
| 오버랩 | 전방 80%, 측방 60% 이상 (정사영상 생성 시) |
