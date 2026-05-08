# 클래스 목록 (누적)

> 이 파일은 프로젝트에서 다루는 모든 검출 클래스를 중복 없이 누적합니다.
> 최초 작성: 2026-05-08

---

## 철도 인프라

| 클래스 (EN) | 클래스 (KR) | SAM3 프롬프트 키워드 | 비고 |
|---|---|---|---|
| `railway` | 철도 레일 | railroad track, railway rail, steel rail | 선로 전체 |
| `catenary_pole` | 전철주 | railway catenary pole, overhead line pole, catenary mast | 가선 지지 기둥 |
| `bracket` | 브라켓/암 | catenary pole top, horizontal cantilever beam, bracket arm | 전철주 상단 가로 암 |
| `wire` | 전선/가선 | overhead wire, catenary wire, electric cable line | 가공 전선 |
| `bridge` | 교량/교각 | bridge, overpass, viaduct | 철도·도로 교량 포함 |

## 도로 인프라

| 클래스 (EN) | 클래스 (KR) | SAM3 프롬프트 키워드 | 비고 |
|---|---|---|---|
| `highway` | 고속도로/도로 | highway road, expressway asphalt, paved road lane | 포장 도로 전반 |
| `guardrail` | 가드레일 | guardrail, highway barrier, road fence, crash barrier | 도로 방호벽 |

## 이동 객체

| 클래스 (EN) | 클래스 (KR) | SAM3 프롬프트 키워드 | 비고 |
|---|---|---|---|
| `vehicle` | 차량 | car, truck, vehicle, automobile | 승용차·트럭 포함 |

## 토지 피복

| 클래스 (EN) | 클래스 (KR) | SAM3 프롬프트 키워드 | 비고 |
|---|---|---|---|
| `building` | 건물 | building, house, rooftop, structure | 지붕면 기준 검출 |
| `farmland` | 농지 | farmland, agricultural field, cropland, vegetable garden | 밭·논 포함 |
| `vegetation` | 식생 | trees, forest, shrubs, vegetation, bushes | 나무·수풀 |

---

_총 11개 클래스 (2026-05-08 기준)_
