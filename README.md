# Expedia Hotel Recommendations

사용자의 검색 행동 데이터를 기반으로 예약 가능성이 높은 **호텔 클러스터 Top-5를 예측**하는 추천 시스템입니다.

> Kaggle — [Expedia Hotel Recommendations](https://www.kaggle.com/competitions/expedia-hotel-recommendations) 대회 데이터 활용  
> 최종 평가 지표: **MAP@5 = 0.4111**

---

## 프로젝트 구조

```
expedia-hotel-recommendations/
├── expedia_train_test.py     # 최종 모델 (MAP@5: 0.4111)
├── practice_train_test.py    # 초기 베이스라인 (MAP@5: 0.4023)
├── train.csv                 # 학습 데이터 (37M rows, 미포함)
├── destinations.csv          # 목적지 LSI 피처 (62,106 rows, 미포함)
├── feature_importance.png    # LightGBM 피처 중요도
├── ensemble_diagram.svg      # Score Fusion 앙상블 구조도
├── project_framework.svg     # 전체 프로젝트 프레임워크
└── data_schema.svg           # 데이터 스키마 다이어그램
```

---

## 문제 정의

- **입력**: 사용자 검색 1건 (출발지, 목적지, 일정, 인원 등)
- **출력**: 예약 가능성이 높은 `hotel_cluster` 번호 상위 5개
- **클래스 수**: 100개 (hotel_cluster 0 ~ 99)
- **평가 지표**: MAP@5 — 정답이 1위면 1.0점, 2위면 0.5점, 5위 밖이면 0점

---

## 데이터

| 파일 | rows | 설명 |
|------|------|------|
| `train.csv` | 37,000,000 | 2013~2014년 사용자 검색·예약 이벤트 |
| `destinations.csv` | 62,106 | 목적지별 LSI 잠재 의미 피처 149개 |

> 본 프로젝트는 메모리/속도 제약으로 `train.csv`에서 **200만 행**을 사용합니다.

---

## 접근 방식

### 하이브리드 앙상블 (LightGBM + 룰 기반 맵)

```
train.csv + destinations.csv
         ↓
     전처리 · 피처 엔지니어링
         ↓              ↓
   LightGBM        룰 기반 맵 8종
   (100클래스)     (예약 히스토리 집계)
         ↓              ↓
       Score Matrix 가중 합산
                ↓
          Top-5 클러스터 반환
```

### 피처 엔지니어링

**날짜/일정**
- `stay_duration`, `lead_time`, `booking_window_bucket`
- `is_weekend_checkin`, `month`, `quarter`, `day_of_week`

**사용자 히스토리**
- `user_fav_cluster`, `user_booking_cnt`, `user_avg_stay`
- `user_fav_country`, `user_fav_market`, `user_market_match`

### 룰 기반 맵 (8종)

| 맵 | 키 | 가중치 |
|----|---|--------|
| leakage | (도시, 거리) | **8.0** (rank-0만) |
| user_dest | (유저ID, 목적지ID) | 5.0 |
| user_market | (유저ID, 시장) | 2.0 |
| adv_pop | (목적지, 국가, 시장) | 1.0 |
| pkg_dest | (패키지여부, 목적지) | 0.8 |
| month_dest | (월, 목적지) | 0.4 |
| basic_pop | (목적지) | 0.4 |
| market_pop | (시장) | 0.2 |

> **핵심 인사이트**: `(user_location_city, orig_destination_distance)` 조합은 호텔 건물을 거의 특정할 수 있어 가장 강력한 시그널입니다. 단, 1위 클러스터에만 적용하고 2~5위는 제외합니다.

---

## 실험 결과

| 버전 | MAP@5 | 변화 | 주요 변경 |
|------|------|------|---------|
| 베이스라인 | 0.4023 | — | LGB + 맵 5종, leakage 3.0 |
| 1차 시도 | 0.3946 | ▼ -0.0077 | leakage 10.0 전 rank 적용 → 역효과 |
| **최종** | **0.4111** | **▲ +0.0088** | leakage rank-0 전용 8.0, 맵 3종 추가 |

---

## 실행 방법

### 요구사항

```bash
pip install pandas numpy lightgbm scikit-learn matplotlib
```

### 데이터 경로 설정

[expedia_train_test.py](expedia_train_test.py) 상단의 `base_path`를 수정합니다.

```python
base_path = "C:/your/path/to/expedia-hotel-recommendations/"
```

### 실행

```bash
python expedia_train_test.py
```

실행 완료 시 MAP@5 점수와 `feature_importance.png`가 출력됩니다.

---

## 주요 발견

1. **강한 도메인 시그널 > 정교한 ML 모델**  
   `(도시, 거리)` 조합 하나가 LGB 모델보다 훨씬 큰 기여를 합니다.

2. **가중치 크기보다 적용 범위가 중요**  
   leakage를 10.0으로 올렸을 때 rank 1~4에도 적용되어 오히려 성능이 하락했습니다. 강한 시그널일수록 1위 예측에만 집중해야 합니다.

3. **계층적 fallback 구조**  
   개인화(유저 히스토리) → 목적지 인기도 → 시장 인기도 순으로 fallback하여 커버리지와 정확도를 동시에 확보합니다.

---

## 개발 환경

| 항목 | 버전 |
|------|------|
| Python | 3.x |
| LightGBM | 최신 |
| pandas | 최신 |
| scikit-learn | 최신 |
| OS | Windows 11 |
