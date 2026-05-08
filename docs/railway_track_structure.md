# 철도 궤도 구조 (Railway Track Structure)

> 출처: 표준 단면도 (Permanent Way cross-section diagram)
> 등록: 2026-05-08

---

## 구조 계층 (위→아래)

```
         Ballast shoulder (도상 어깨)
              │
    ┌─────────┴──────────────┐
    │  Rails (레일)           │  ← Track Structure
    │  Sleepers (침목)        │
    │  Ballast & sub-ballast  │
    │  (자갈도상 + 하부도상)   │
    └────────────────────────┘
    │  Blanket (차단층, 선택)  │  ← Formation
    │  Subgrade (노반)         │
    └────────────────────────┘
         Subsoil / Natural Ground (자연 지반)
```

## 각 요소 설명

| 용어 (EN) | 용어 (KR) | 설명 |
|---|---|---|
| Rails | 레일 | 열차 하중을 직접 받는 강철 종방향 빔 |
| Sleepers | 침목 | 레일을 지지하는 횡방향 콘크리트/목재 부재 |
| Ballast & sub-ballast | 자갈도상 | 침목 하부의 쇄석층. 하중 분산 및 배수 역할 |
| Ballast shoulder | 도상 어깨 | 도상 측면 경사부 |
| Cess | 측구 공간 | 선로 측면 유지보수·배수 공간 |
| Blanket | 차단층 | 노반 보호용 모래·자갈 혼합층 (선택적) |
| Subgrade | 노반 | 궤도 기초를 이루는 다짐 토공층 |
| Subsoil / Natural Ground | 자연 지반 | 공사 전 원래 지반 |

## 드론 항공뷰 검출 대상

드론 정사영상에서 가시적인 요소:

| 클래스 | 가시성 | 비고 |
|---|---|---|
| `railway` (레일) | 중 | 얇은 두 줄 금속선 — 고해상도에서만 식별 |
| `sleeper` (침목) | 중 | 레일에 수직인 직사각형 패턴 |
| `ballast` (자갈도상) | 높음 | 회색 쇄석 질감, 가장 넓은 면적 |
| `ballast shoulder` | 낮음 | 도상 측면 경사 — 별도 클래스 미지정 |
| Blanket, Subgrade | 없음 | 지하 매설층 — 항공뷰 불가 |
