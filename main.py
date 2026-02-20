from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

REF_DATE = pd.Timestamp("2025-02-03")
CAT_COLS = ["age_category", "sex", "user_stage"]


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    input_dir = Path(__file__).resolve().parent / "input"
    data = pd.read_csv(input_dir / "data.csv")
    train_flag = pd.read_csv(input_dir / "train_flag.csv")
    return data, train_flag


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["date"] = pd.to_datetime(data["date"].astype(str), format="%Y%m%d")

    ms = pd.to_numeric(data["membership_start_ym"], errors="coerce").astype("Int64")
    data["membership_start_ym"] = pd.to_datetime(
        ms.astype(str), format="%Y%m", errors="coerce"
    )

    hist = data[data["date"] < REF_DATE].copy()

    def agg_window(df: pd.DataFrame, days: int) -> pd.DataFrame:
        start = REF_DATE - pd.Timedelta(days=days)
        w = df[df["date"] >= start]
        g = w.groupby("user_id").agg(
            txn_count=("total_price", "size"),
            spend_sum=("total_price", "sum"),
            amount_sum=("amount", "sum"),
            avg_unit_price=("average_unit_price", "mean"),
        )
        g.columns = [f"{c}_{days}d" for c in g.columns]
        return g

    last_date = hist.groupby("user_id")["date"].max()
    recency = (REF_DATE - last_date).dt.days.to_frame("days_since_last_purchase")

    base = hist.groupby("user_id").agg(
        txn_count_all=("total_price", "size"),
        spend_sum_all=("total_price", "sum"),
        amount_sum_all=("amount", "sum"),
        avg_unit_price_all=("average_unit_price", "mean"),
        unique_items_all=("jan_cd", "nunique"),
        unique_cat1_all=("item_category_cd_1", "nunique"),
    )

    w7 = agg_window(hist, 7)
    w14 = agg_window(hist, 14)
    w30 = agg_window(hist, 30)
    w60 = agg_window(hist, 60)
    w90 = agg_window(hist, 90)

    hist_sorted = hist.sort_values(["user_id", "date"])
    profile_last = hist_sorted.groupby("user_id").tail(1).set_index("user_id")

    profile = pd.DataFrame(index=profile_last.index)
    profile["age_category"] = profile_last["age_category"].fillna("unknown").astype("category")
    profile["sex"] = profile_last["sex"].fillna("unknown").astype("category")
    profile["user_stage"] = profile_last["user_stage"].fillna("unknown").astype("category")
    profile["user_flag_ec"] = (
        pd.to_numeric(profile_last["user_flag_ec"], errors="coerce").fillna(0).astype(int)
    )

    for k in range(1, 7):
        col = f"user_flag_{k}"
        profile[col] = pd.to_numeric(profile_last[col], errors="coerce").fillna(0).astype(int)

    ms_last = profile_last["membership_start_ym"]
    m = ms_last.dt.year * 12 + ms_last.dt.month
    profile["membership_months"] = ((REF_DATE.year * 12 + REF_DATE.month) - m).astype(float)
    profile["membership_months"] = profile["membership_months"].fillna(-1.0)

    x = (
        profile.join(recency, how="left")
        .join(base, how="left")
        .join(w7, how="left")
        .join(w14, how="left")
        .join(w30, how="left")
        .join(w60, how="left")
        .join(w90, how="left")
    )

    x["spend_prev30d"] = (x["spend_sum_60d"] - x["spend_sum_30d"]).clip(lower=0)
    x["txn_prev30d"] = (x["txn_count_60d"] - x["txn_count_30d"]).clip(lower=0)
    x["spend_trend_30d"] = x["spend_sum_30d"] - x["spend_prev30d"]
    x["txn_trend_30d"] = x["txn_count_30d"] - x["txn_prev30d"]
    x["spend_ratio_30d_prev30d"] = x["spend_sum_30d"] / (x["spend_prev30d"] + 1)
    x["txn_ratio_30d_prev30d"] = x["txn_count_30d"] / (x["txn_prev30d"] + 1)
    x["activity_decay_30d_over_90d"] = x["txn_count_30d"] / (x["txn_count_90d"] + 1)

    hist_sorted2 = hist.sort_values(["user_id", "date"]).reset_index(drop=True)
    interval_days = hist_sorted2.groupby("user_id")["date"].diff().dt.days
    interval_feat = interval_days.groupby(hist_sorted2["user_id"]).agg(
        mean_purchase_interval="mean",
        std_purchase_interval="std",
        last_purchase_interval="last",
    )
    x = x.join(interval_feat, how="left")

    num_cols = x.select_dtypes(include=[np.number]).columns
    x[num_cols] = x[num_cols].fillna(0)

    return x.reset_index()


def train_and_score(train_df: pd.DataFrame) -> float:
    y = train_df["churn"].astype(int)
    x_train_full = train_df.drop(columns=["churn", "user_id"])

    for col in CAT_COLS:
        x_train_full[col] = x_train_full[col].astype("category")

    for col in CAT_COLS:
        if "unknown" not in x_train_full[col].cat.categories:
            x_train_full[col] = x_train_full[col].cat.add_categories(["unknown"])
        x_train_full[col] = x_train_full[col].fillna("unknown")

    num_cols = x_train_full.select_dtypes(include=[np.number]).columns
    x_train_full[num_cols] = x_train_full[num_cols].fillna(0)

    x_tr, x_val, y_tr, y_val = train_test_split(
        x_train_full,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = lgb.LGBMClassifier(
        n_estimators=5000,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1,
    )

    model.fit(
        x_tr,
        y_tr,
        eval_set=[(x_val, y_val)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(200, verbose=False)],
    )

    val_pred = model.predict_proba(x_val)[:, 1]
    return float(roc_auc_score(y_val, val_pred))


def main() -> None:
    data, train_flag = load_inputs()
    x = build_features(data)
    train_df = train_flag[["user_id", "churn"]].merge(x, on="user_id", how="left")
    auc = train_and_score(train_df)
    print(f"Validation AUC (churn=1): {auc}")


if __name__ == "__main__":
    main()
