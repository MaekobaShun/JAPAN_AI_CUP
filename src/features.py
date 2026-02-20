"""
特徴量生成ロジック
"""
import pandas as pd
import numpy as np

def create_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    特徴量を生成する

    Args:
        data: 前処理済みデータ
    Returns:
        特徴量データフレーム（user_idを含む）
    """
    # データ全体の最新日付を取得（基準日として使用）
    latest_date = data["date"].max()
    
    # 各ユーザーの最後の来店日を取得
    last_visit = data.groupby("user_id")["date"].max().reset_index()
    last_visit.columns = ["user_id", "last_visit_date"]
    
    # 1. 最終購入日からの経過日数
    last_visit["days_since_last_purchase"] = (latest_date - last_visit["last_visit_date"]).dt.days
    
    # 2. 平均購入間隔を計算
    # 各ユーザーの購入日をソートして、購入間隔を計算
    def calc_avg_purchase_interval(group):
        dates = group["date"].sort_values().unique()
        if len(dates) <= 1:
            return np.nan
        # numpy.diffで差分を計算（numpy.timedelta64が返される）
        intervals = np.diff(dates)
        if len(intervals) == 0:
            return np.nan
        # numpy.timedelta64を日数に変換
        avg_interval = intervals.mean()
        if pd.isna(avg_interval):
            return np.nan
        # numpy.timedelta64を日数（float）に変換
        return avg_interval / np.timedelta64(1, 'D')
    
    avg_intervals = data.groupby("user_id").apply(calc_avg_purchase_interval).reset_index()
    avg_intervals.columns = ["user_id", "avg_purchase_interval_days"]
    
    # 3. 最終購入日が平均購入間隔に比べて何倍か
    last_visit = pd.merge(last_visit, avg_intervals, on="user_id", how="left")
    last_visit["last_purchase_to_avg_interval_ratio"] = (
        last_visit["days_since_last_purchase"] / last_visit["avg_purchase_interval_days"]
    )
    # 平均購入間隔が0またはNaNの場合はNaNを設定
    last_visit["last_purchase_to_avg_interval_ratio"] = last_visit["last_purchase_to_avg_interval_ratio"].replace([np.inf, -np.inf], np.nan)
    
    # 4. 直近30日の活動量（金額、来店）
    date_30_days_ago = latest_date - pd.Timedelta(days=30)
    recent_30_days = data[data["date"] >= date_30_days_ago].copy()
    
    recent_30_features = recent_30_days.groupby("user_id").agg({
        "total_price": "sum",
        "date": "nunique"  # 来店回数（ユニークな日付数）
    }).reset_index()
    recent_30_features.columns = ["user_id", "recent_30_days_total_price", "recent_30_days_visit_count"]
    
    # 5. 30日～60日前の金額、来店
    date_60_days_ago = latest_date - pd.Timedelta(days=60)
    period_30_60_days = data[
        (data["date"] >= date_60_days_ago) & (data["date"] < date_30_days_ago)
    ].copy()
    
    period_30_60_features = period_30_60_days.groupby("user_id").agg({
        "total_price": "sum",
        "date": "nunique"  # 来店回数（ユニークな日付数）
    }).reset_index()
    period_30_60_features.columns = ["user_id", "period_30_60_days_total_price", "period_30_60_days_visit_count"]
    
    # 既存の特徴量を作成
    features = pd.DataFrame(data.groupby("user_id").agg({
        "date": ["count"],
        "average_unit_price": ["sum"]
    }).reset_index().to_numpy())

    features.columns = ["user_id", "date_count", "average_unit_price_sum"]

    features["average_unit_price_sum"] = pd.to_numeric(features["average_unit_price_sum"], errors="coerce")
    features["date_count"] = pd.to_numeric(features["date_count"], errors="coerce")

    # すべての特徴量をマージ
    features = pd.merge(features, last_visit[["user_id", "days_since_last_purchase", "avg_purchase_interval_days", "last_purchase_to_avg_interval_ratio"]], on="user_id", how="left")
    features = pd.merge(features, recent_30_features, on="user_id", how="left")
    features = pd.merge(features, period_30_60_features, on="user_id", how="left")
    
    # 欠損値を0で埋める（該当期間に購入がなかった場合）
    features["recent_30_days_total_price"] = features["recent_30_days_total_price"].fillna(0)
    features["recent_30_days_visit_count"] = features["recent_30_days_visit_count"].fillna(0)
    features["period_30_60_days_total_price"] = features["period_30_60_days_total_price"].fillna(0)
    features["period_30_60_days_visit_count"] = features["period_30_60_days_visit_count"].fillna(0)
    
    # 対数変換: last_purchase_to_avg_interval_ratio の log1p
    features["log_ratio"] = np.log1p(features["last_purchase_to_avg_interval_ratio"])
    
    return features


def prepare_train_data(
    train_flag: pd.DataFrame,
    features: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series]:
    """
    学習データを準備する
    
    Args:
        train_flag: 学習フラグデータ（user_id, churnを含む）
        features: 特徴量データ（user_idを含む）
    
    Returns:
        (X_train, y_train) のタプル
    """
    X_train = pd.merge(train_flag, features, on="user_id", how="left")
    X_train = X_train.drop(["user_id", "churn"], axis=1)
    y_train = train_flag["churn"]
    
    return X_train, y_train


def prepare_test_data(
    sample_submission: pd.DataFrame,
    features: pd.DataFrame
) -> pd.DataFrame:
    """
    テストデータを準備する
    
    Args:
        sample_submission: サブミッションテンプレート（user_idを含む）
        features: 特徴量データ（user_idを含む）
    
    Returns:
        X_test（user_idを除く）
    """
    X_test = pd.merge(sample_submission[["user_id"]], features, on="user_id", how="left")
    X_test = X_test.drop(["user_id"], axis=1)
    
    return X_test


def filter_features_r_only(
    X_train: pd.DataFrame,
    remove_days_since_last_purchase: bool = False,
    remove_period_30_60_visit_count: bool = False,
    remove_all_price_features: bool = False
) -> pd.DataFrame:
    """
    Rのみでモデルを作るための特徴量フィルタリング
    
    Args:
        X_train: 特徴量データ
        remove_days_since_last_purchase: days_since_last_purchaseを削除するか
        remove_period_30_60_visit_count: period_30_60_days_visit_countを削除するか
        remove_all_price_features: 金額系特徴量を全て削除するか
    
    Returns:
        フィルタリング後の特徴量データ
    """
    X_filtered = X_train.copy()
    
    # ① days_since_last_purchase を削除
    if remove_days_since_last_purchase and "days_since_last_purchase" in X_filtered.columns:
        X_filtered = X_filtered.drop(columns=["days_since_last_purchase"])
    
    # ② period_30_60_days_visit_count を削除
    if remove_period_30_60_visit_count and "period_30_60_days_visit_count" in X_filtered.columns:
        X_filtered = X_filtered.drop(columns=["period_30_60_days_visit_count"])
    
    # ③ 金額系特徴量を全て削除
    if remove_all_price_features:
        price_features = [
            "recent_30_days_total_price",
            "period_30_60_days_total_price",
            "average_unit_price_sum"
        ]
        for feat in price_features:
            if feat in X_filtered.columns:
                X_filtered = X_filtered.drop(columns=[feat])
    
    return X_filtered


def create_features_rf(data: pd.DataFrame) -> pd.DataFrame:
    """
    RFMのR（Recency）とF（Frequency）のコア特徴量のみを使用
    
    🔴 削除: log_ratio, txn_count_all, txn_per_visit, recent30_visit_ratio, 
            spend_per_visit, period_30_60_days_total_price, period_30_60_days_visit_count,
            recent_30_days_total_price, average_unit_price_sum
    
    🟢 残す: date_count, last_purchase_to_avg_interval_ratio, recent_30_days_visit_count,
            days_since_last_purchase, avg_purchase_interval_days
    
    🟡 追加: ratio_x_datecount (ratio × date_count)
    
    Args:
        data: 前処理済みデータ
    Returns:
        特徴量データフレーム（user_idを含む）
    """
    # 既存の特徴量を生成
    features = create_features(data)
    
    # 🔴 不要な特徴量を削除
    features_to_remove = [
        "log_ratio",
        "txn_count_all",
        "txn_per_visit",
        "recent30_visit_ratio",
        "spend_per_visit",
        "period_30_60_days_total_price",
        "period_30_60_days_visit_count",
        "recent_30_days_total_price",
        "average_unit_price_sum",
    ]
    
    for feat in features_to_remove:
        if feat in features.columns:
            features = features.drop(columns=[feat])
    
    # 🟡 新しい特徴量を追加: ratio_x_datecount
    features["ratio_x_datecount"] = (
        features["last_purchase_to_avg_interval_ratio"] * features["date_count"]
    )
    
    return features
