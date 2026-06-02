import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. 메모리 절감
# ============================================================
def reduce_mem_usage(df):
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min, c_max = df[col].min(), df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
        else:
            df[col] = df[col].astype('category')
    return df


# ============================================================
# 2. 데이터 불러오기 (2M)
# ============================================================
print("📂 데이터를 불러오는 중...")
base_path = "C:/Users/82108/Downloads/expedia-hotel-recommendations/"
train = pd.read_csv(base_path + "train.csv", nrows=2_000_000)
destinations = pd.read_csv(base_path + "destinations.csv")

train = reduce_mem_usage(train)
destinations = reduce_mem_usage(destinations)
print(f"  train shape: {train.shape}, destinations shape: {destinations.shape}")


# ============================================================
# 3. 날짜 전처리
# ============================================================
def process_dates(df):
    df['date_time'] = pd.to_datetime(df['date_time'])
    df['srch_ci']   = pd.to_datetime(df['srch_ci'], errors='coerce')
    df['srch_co']   = pd.to_datetime(df['srch_co'], errors='coerce')
    df['month']           = df['date_time'].dt.month
    df['year']            = df['date_time'].dt.year
    df['quarter']         = df['date_time'].dt.quarter
    df['day_of_week']     = df['date_time'].dt.dayofweek
    df['stay_duration']   = (df['srch_co'] - df['srch_ci']).dt.days
    df['lead_time']       = (df['srch_ci'] - df['date_time']).dt.days
    df['ci_day_of_week']      = df['srch_ci'].dt.dayofweek
    df['is_weekend_checkin']  = (df['ci_day_of_week'] >= 5).astype(np.int8)
    df['booking_window_bucket'] = pd.cut(
        df['lead_time'].fillna(-1),
        bins=[-999, 0, 7, 30, 90, 9999],
        labels=False
    ).astype('float32')
    df.drop(['date_time', 'srch_ci', 'srch_co'], axis=1, inplace=True)
    return df

train = process_dates(train)
train['orig_destination_distance'] = train['orig_destination_distance'].round(4)
train = pd.merge(train, destinations, on='srch_destination_id', how='left')
train.reset_index(drop=True, inplace=True)


# ============================================================
# 4. 결측치 처리 + 파생 피처
# ============================================================
train['dist_null'] = train['orig_destination_distance'].isnull().astype(np.int8)
train['orig_destination_distance'] = train['orig_destination_distance'].fillna(-1)
train['stay_duration'] = train['stay_duration'].fillna(-1)
train['lead_time']     = train['lead_time'].fillna(-1)
train.fillna(0, inplace=True)
train.loc[train['stay_duration'] < 0, 'stay_duration'] = 0

train['adults_children_ratio'] = (
    train['srch_adults_cnt'] / (train['srch_children_cnt'] + 1)
).astype(np.float32)
train['total_travelers'] = (
    train['srch_adults_cnt'] + train['srch_children_cnt']
).astype(np.int8)
train['is_domestic'] = (
    train['user_location_country'] == train['hotel_country']
).astype(np.int8)


# ============================================================
# 5. 유저 히스토리 피처
# ============================================================
print("👤 유저 히스토리 피처 생성 중...")
bookings = train[train['is_booking'] == 1]

def mode_first(x):
    return x.value_counts().index[0]

user_fav           = bookings.groupby('user_id')['hotel_cluster'].agg(mode_first).rename('user_fav_cluster')
user_booking_cnt   = bookings.groupby('user_id').size().rename('user_booking_cnt')
user_avg_stay      = bookings.groupby('user_id')['stay_duration'].mean().rename('user_avg_stay')
user_fav_country   = bookings.groupby('user_id')['hotel_country'].agg(mode_first).rename('user_fav_country')
user_fav_market    = bookings.groupby('user_id')['hotel_market'].agg(mode_first).rename('user_fav_market')
user_fav_dest_type = bookings.groupby('user_id')['srch_destination_type_id'].agg(mode_first).rename('user_fav_dest_type')

for feat in [user_fav, user_booking_cnt, user_avg_stay,
             user_fav_country, user_fav_market, user_fav_dest_type]:
    train = train.merge(feat, on='user_id', how='left')

train['user_fav_cluster']   = train['user_fav_cluster'].fillna(-1).astype(np.float32)
train['user_booking_cnt']   = train['user_booking_cnt'].fillna(0).astype(np.float32)
train['user_avg_stay']      = train['user_avg_stay'].fillna(-1).astype(np.float32)
train['user_fav_country']   = train['user_fav_country'].fillna(-1).astype(np.float32)
train['user_fav_market']    = train['user_fav_market'].fillna(-1).astype(np.float32)
train['user_fav_dest_type'] = train['user_fav_dest_type'].fillna(-1).astype(np.float32)
train['user_country_match'] = (train['user_fav_country'] == train['hotel_country']).astype(np.int8)
train['user_market_match']  = (train['user_fav_market']  == train['hotel_market']).astype(np.int8)


# ============================================================
# 6. 룰베이스 맵 생성 (예약 5배 가중)
# ============================================================
def get_rule_maps(df):
    df = df.copy()
    df['score'] = df['is_booking'] * 5 + 1

    def top5_map(agg_df, key_cols, cluster_col='hotel_cluster', score_col='score'):
        agg_df = agg_df.copy()
        agg_df['_rank'] = agg_df.groupby(key_cols)[score_col].rank(
            ascending=False, method='first'
        )
        top5 = agg_df[agg_df['_rank'] <= 5].sort_values(key_cols + ['_rank'])
        return top5.groupby(key_cols)[cluster_col].apply(list).to_dict()

    # Leakage (도시+거리) — 예약만
    leaks = (
        df[df['is_booking'] == 1]
        .groupby(['user_location_city', 'orig_destination_distance', 'hotel_cluster'])
        .size().reset_index(name='score')
    )
    leak_map = top5_map(leaks, ['user_location_city', 'orig_destination_distance'])

    # 정밀 매칭 (목적지+국가+시장)
    agg_adv = df.groupby(
        ['srch_destination_id', 'hotel_country', 'hotel_market', 'hotel_cluster']
    )['score'].sum().reset_index()
    adv_pop_map = top5_map(agg_adv, ['srch_destination_id', 'hotel_country', 'hotel_market'])

    # 일반 매칭 (목적지 ID)
    agg_basic = df.groupby(['srch_destination_id', 'hotel_cluster'])['score'].sum().reset_index()
    basic_pop_map = top5_map(agg_basic, ['srch_destination_id'])

    # 시장 단위
    agg_market = df.groupby(['hotel_market', 'hotel_cluster'])['score'].sum().reset_index()
    market_pop_map = top5_map(agg_market, ['hotel_market'])

    # 목적지 타입+국가
    agg_dtype = df.groupby(
        ['srch_destination_type_id', 'hotel_country', 'hotel_cluster']
    )['score'].sum().reset_index()
    dtype_map = top5_map(agg_dtype, ['srch_destination_type_id', 'hotel_country'])

    # 유저+목적지 히스토리 — 예약만
    user_dest_agg = (
        df[df['is_booking'] == 1]
        .groupby(['user_id', 'srch_destination_id', 'hotel_cluster'])
        .size().reset_index(name='score')
    )
    user_dest_map = top5_map(user_dest_agg, ['user_id', 'srch_destination_id'])

    # ★ 유저+시장 히스토리 (신규) — 예약만
    user_market_agg = (
        df[df['is_booking'] == 1]
        .groupby(['user_id', 'hotel_market', 'hotel_cluster'])
        .size().reset_index(name='score')
    )
    user_market_map = top5_map(user_market_agg, ['user_id', 'hotel_market'])

    # ★ is_package + 목적지 (신규)
    pkg_agg = df.groupby(
        ['is_package', 'srch_destination_id', 'hotel_cluster']
    )['score'].sum().reset_index()
    pkg_dest_map = top5_map(pkg_agg, ['is_package', 'srch_destination_id'])

    # ★ 월 + 목적지 시즌성 (신규)
    month_dest_agg = df.groupby(
        ['month', 'srch_destination_id', 'hotel_cluster']
    )['score'].sum().reset_index()
    month_dest_map = top5_map(month_dest_agg, ['month', 'srch_destination_id'])

    # ★ 글로벌 인기 top5 (최종 fallback)
    global_top5 = (
        df[df['is_booking'] == 1]
        .groupby('hotel_cluster').size()
        .sort_values(ascending=False).head(5).index.tolist()
    )

    return (leak_map, adv_pop_map, basic_pop_map, market_pop_map, dtype_map,
            user_dest_map, user_market_map, pkg_dest_map, month_dest_map, global_top5)

print("🔥 맵 생성 중...")
(leak_map, adv_pop_map, basic_pop_map, market_pop_map, dtype_map,
 user_dest_map, user_market_map, pkg_dest_map, month_dest_map, global_top5) = get_rule_maps(train)


# ============================================================
# 7. 모델 학습 (파라미터 개선)
# ============================================================
drop_cols = ['hotel_cluster', 'user_id', 'cnt', 'score']
X = train.drop(drop_cols, axis=1, errors='ignore')
y = train['hotel_cluster']

for col in X.select_dtypes(['category']).columns:
    X[col] = X[col].cat.codes

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

val_user_ids = train.loc[X_val.index, 'user_id'].values

model = lgb.LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=63,
    max_bin=63,
    min_child_samples=50,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.6,
    reg_alpha=0.1,
    reg_lambda=1.0,
    objective='multiclass',
    n_jobs=-1,
    random_state=42,
    verbose=-1
)

print("🚀 모델 학습 시작...")
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(stopping_rounds=30), lgb.log_evaluation(100)]
)
print("✅ 학습 완료!")


# ============================================================
# 8. 벡터화 앙상블
# ============================================================
y_probs = model.predict_proba(X_val).astype(np.float32)

X_val_r = X_val.reset_index(drop=True)
n_val = len(X_val_r)
N_CLUSTERS = 100

score_matrix = y_probs.copy()

def build_boost_df(pop_map, key_cols, boost_weight):
    rows = []
    for key, clusters in pop_map.items():
        k = key if isinstance(key, tuple) else (key,)
        for rank, cluster in enumerate(clusters):
            rows.append((*k, int(cluster), boost_weight / (rank + 1)))
    if not rows:
        return pd.DataFrame(columns=key_cols + ['cluster', 'boost'])
    return pd.DataFrame(rows, columns=key_cols + ['cluster', 'boost'])

def build_top1_boost_df(pop_map, key_cols, boost_weight):
    """rank 0만 부스트 — leakage처럼 1위가 거의 확실한 경우 전용"""
    rows = []
    for key, clusters in pop_map.items():
        if not clusters:
            continue
        k = key if isinstance(key, tuple) else (key,)
        rows.append((*k, int(clusters[0]), boost_weight))
    if not rows:
        return pd.DataFrame(columns=key_cols + ['cluster', 'boost'])
    return pd.DataFrame(rows, columns=key_cols + ['cluster', 'boost'])

def apply_map_boost(score_mat, val_df, boost_df, key_cols):
    if boost_df.empty:
        return
    val_keyed = val_df[key_cols].copy()
    val_keyed['_row'] = np.arange(len(val_keyed))
    merged = val_keyed.merge(boost_df, on=key_cols, how='inner')
    if merged.empty:
        return
    np.add.at(
        score_mat,
        (merged['_row'].values, merged['cluster'].values.astype(np.int32)),
        merged['boost'].values.astype(np.float32)
    )

print("⚡ 벡터화 앙상블 적용 중...")

# Leakage: rank-0만 8.0 (rank 1~4 제거 — 틀렸을 때 top-5를 오염시키던 원인)
leak_boost_df = build_top1_boost_df(leak_map, ['user_location_city', 'orig_destination_distance'], 8.0)
if not leak_boost_df.empty:
    val_leak = pd.DataFrame({
        'user_location_city':        X_val_r['user_location_city'].values,
        'orig_destination_distance': X_val_r['orig_destination_distance'].values,
        '_row': np.arange(n_val)
    })
    merged_leak = val_leak.merge(leak_boost_df,
                                 on=['user_location_city', 'orig_destination_distance'],
                                 how='inner')
    if not merged_leak.empty:
        np.add.at(score_matrix,
                  (merged_leak['_row'].values, merged_leak['cluster'].values.astype(np.int32)),
                  merged_leak['boost'].values.astype(np.float32))

# 유저+목적지 히스토리 (5.0 유지)
ud_boost_df = build_boost_df(user_dest_map, ['user_id', 'srch_destination_id'], 5.0)
if not ud_boost_df.empty:
    val_ud = pd.DataFrame({
        'user_id':             val_user_ids,
        'srch_destination_id': X_val_r['srch_destination_id'].values,
        '_row': np.arange(n_val)
    })
    merged_ud = val_ud.merge(ud_boost_df, on=['user_id', 'srch_destination_id'], how='inner')
    if not merged_ud.empty:
        np.add.at(score_matrix,
                  (merged_ud['_row'].values, merged_ud['cluster'].values.astype(np.int32)),
                  merged_ud['boost'].values.astype(np.float32))

# 유저+시장 히스토리 (신규, 2.0)
um_boost_df = build_boost_df(user_market_map, ['user_id', 'hotel_market'], 2.0)
if not um_boost_df.empty:
    val_um = pd.DataFrame({
        'user_id':     val_user_ids,
        'hotel_market': X_val_r['hotel_market'].values,
        '_row': np.arange(n_val)
    })
    merged_um = val_um.merge(um_boost_df, on=['user_id', 'hotel_market'], how='inner')
    if not merged_um.empty:
        np.add.at(score_matrix,
                  (merged_um['_row'].values, merged_um['cluster'].values.astype(np.int32)),
                  merged_um['boost'].values.astype(np.float32))

# 목적지+국가+시장 (1.0)
apply_map_boost(score_matrix, X_val_r,
                build_boost_df(adv_pop_map, ['srch_destination_id', 'hotel_country', 'hotel_market'], 1.0),
                ['srch_destination_id', 'hotel_country', 'hotel_market'])

# ★ is_package + 목적지 (신규, 0.8)
apply_map_boost(score_matrix, X_val_r,
                build_boost_df(pkg_dest_map, ['is_package', 'srch_destination_id'], 0.8),
                ['is_package', 'srch_destination_id'])

# ★ 월 + 목적지 (신규, 0.4)
apply_map_boost(score_matrix, X_val_r,
                build_boost_df(month_dest_map, ['month', 'srch_destination_id'], 0.4),
                ['month', 'srch_destination_id'])

# 목적지 단독 (0.4)
apply_map_boost(score_matrix, X_val_r,
                build_boost_df(basic_pop_map, ['srch_destination_id'], 0.4),
                ['srch_destination_id'])

# 시장 단독 (0.2)
apply_map_boost(score_matrix, X_val_r,
                build_boost_df(market_pop_map, ['hotel_market'], 0.2),
                ['hotel_market'])

# 목적지 타입+국가 (0.2)
apply_map_boost(score_matrix, X_val_r,
                build_boost_df(dtype_map, ['srch_destination_type_id', 'hotel_country'], 0.2),
                ['srch_destination_type_id', 'hotel_country'])

# 유저 즐겨찾기 (1.5 유지)
fav_clusters = X_val_r['user_fav_cluster'].values.astype(int)
valid_mask = (fav_clusters >= 0) & (fav_clusters < N_CLUSTERS)
score_matrix[np.where(valid_mask)[0], fav_clusters[valid_mask]] += 1.5

# ★ 글로벌 인기 fallback (score가 모두 0인 행)
row_max = score_matrix.max(axis=1)
zero_rows = np.where(row_max == 0)[0]
if len(zero_rows) > 0:
    for rank, gc in enumerate(global_top5):
        score_matrix[zero_rows, int(gc)] += 0.1 / (rank + 1)

# 최종 top-5
final_top_5_preds = np.argsort(score_matrix, axis=1)[:, -5:][:, ::-1]


# ============================================================
# 9. MAP@5 평가
# ============================================================
def quick_map5(y_true, y_pred_top5):
    scores = [
        1 / (np.where(p == t)[0][0] + 1) if t in p else 0
        for t, p in zip(y_true, y_pred_top5)
    ]
    return np.mean(scores)

map5_score = quick_map5(y_val.values, final_top_5_preds)
print("\n" + "=" * 50)
print(f"🏆 최종 MAP@5 점수: {map5_score:.4f}")
print("=" * 50)


# ============================================================
# 10. 피처 중요도 시각화
# ============================================================
feat_imp = pd.Series(model.feature_importances_, index=X_train.columns)
top20 = feat_imp.sort_values(ascending=False).head(20)

plt.figure(figsize=(10, 6))
top20.sort_values().plot(kind='barh', color='steelblue')
plt.title('Feature Importance (Top 20)', fontsize=14)
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
plt.show()
print("📊 피처 중요도 저장 완료: feature_importance.png")
