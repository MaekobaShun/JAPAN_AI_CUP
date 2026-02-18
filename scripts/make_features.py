# make_features.py
# Usage:
#   python make_features.py --input train.csv --ref_date 2024-04-01 --out_dir out
#
# It will create:
#   out/features_YYYYMMDD.csv        (user-level features)
#   out/labels_YYYYMMDD.csv          (user-level labels; optional if future data exists)
#   out/train_table_YYYYMMDD.csv     (features + target; target=0 for users w/o future rows)

import argparse
import os
import numpy as np
import pandas as pd

REQUIRED_COLS = [
    "jan_cd","item_name","item_spec",
    "item_category_cd_1","item_category_cd_2","item_category_cd_3","item_category_name",
    "average_unit_price","amount","total_price",
    "user_id","date","store_deli","user_flag_ec","membership_start_ym",
    "age_category","sex","user_stage",
    "user_flag_1","user_flag_2","user_flag_3","user_flag_4","user_flag_5","user_flag_6"
]

def load_transactions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d", errors="coerce")
    if df["date"].isna().any():
        bad = df[df["date"].isna()].head(5)
        raise ValueError(f"Bad date rows found. Example:\n{bad}")

    ms = pd.to_numeric(df["membership_start_ym"], errors="coerce").astype("Int64")
    df["membership_start_ym"] = pd.to_datetime(ms.astype(str), format="%Y%m", errors="coerce")

    for c in ["average_unit_price","amount","total_price","user_flag_ec",
              "user_flag_1","user_flag_2","user_flag_3","user_flag_4","user_flag_5","user_flag_6",
              "item_category_cd_1","item_category_cd_2","item_category_cd_3"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in ["store_deli","age_category","sex","user_stage","item_category_name","item_name","item_spec"]:
        df[c] = df[c].fillna("unknown").astype(str)

    return df


def _encode_store_deli(s: pd.Series) -> pd.DataFrame:
    text = s.astype(str)
    is_store = text.str.contains("店").astype(int)
    is_delivery = text.str.contains("宅|配|デリ").astype(int)

    return pd.DataFrame({
        "store_only": (is_store & (1 - is_delivery)).astype(int),
        "delivery_only": (is_delivery & (1 - is_store)).astype(int),
        "both_store_delivery": (is_store & is_delivery).astype(int),
        "store_deli_unknown": (text.eq("unknown")).astype(int),
    }, index=s.index)


def _category_entropy(counts: np.ndarray) -> float:
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log(p)).sum())


def build_user_features(
    df: pd.DataFrame,
    ref_date: pd.Timestamp,
    windows_days=(7, 30, 60, 90, 180),
    cat_col="item_category_cd_1",
    topk_cat=10,
) -> pd.DataFrame:

    if not isinstance(ref_date, pd.Timestamp):
        ref_date = pd.Timestamp(ref_date)

    hist = df[df["date"] < ref_date].copy()
    hist = hist.sort_values(["user_id", "date"])

    # latest profile per user
    last_profile = hist.groupby("user_id").tail(1).set_index("user_id")

    profile = pd.DataFrame(index=last_profile.index)
    profile["age_category"] = last_profile["age_category"].astype("category")
    profile["sex"] = last_profile["sex"].astype("category")
    profile["user_stage"] = last_profile["user_stage"].astype("category")
    profile["user_flag_ec"] = last_profile["user_flag_ec"].fillna(0).astype(int)

    for k in range(1, 7):
        col = f"user_flag_{k}"
        profile[col] = last_profile[col].fillna(0).astype(int)

    ms = last_profile["membership_start_ym"]
    profile["membership_months"] = (
        (ref_date.year * 12 + ref_date.month) - (ms.dt.year * 12 + ms.dt.month)
    ).astype("float").fillna(-1.0)

    profile = pd.concat([profile, _encode_store_deli(last_profile["store_deli"])], axis=1)

    # daily aggregation
    daily = (
        hist.groupby(["user_id", "date"], as_index=False)
            .agg(
                txn_count=("total_price", "size"),
                amount_sum=("amount", "sum"),
                spend_sum=("total_price", "sum"),
                unit_price_mean=("average_unit_price", "mean"),
            )
            .sort_values(["user_id", "date"])
    )

    # recency
    last_date = daily.groupby("user_id")["date"].max()
    recency = (ref_date - last_date).dt.days.rename("days_since_last_purchase")

    # rolling window sums
    feats = []
    daily_idxed = daily.set_index("date")

    for w in windows_days:
        rolled = (
            daily_idxed.groupby("user_id")[["txn_count","amount_sum","spend_sum"]]
                      .rolling(f"{w}D", closed="both")
                      .sum()
                      .reset_index()
        )
        last_rolled = rolled.groupby("user_id").tail(1).set_index("user_id")

        f = pd.DataFrame(index=last_rolled.index)
        f[f"txn_count_{w}d"] = last_rolled["txn_count"].fillna(0)
        f[f"amount_sum_{w}d"] = last_rolled["amount_sum"].fillna(0)
        f[f"spend_sum_{w}d"] = last_rolled["spend_sum"].fillna(0)
        f[f"avg_spend_per_txn_{w}d"] = (
            f[f"spend_sum_{w}d"] / f[f"txn_count_{w}d"].replace(0, np.nan)
        ).fillna(0.0)

        feats.append(f)

    txn_feats = pd.concat(feats, axis=1).sort_index()

    # trend: 30d vs prev30d (using 60d)
    if 30 in windows_days and 60 in windows_days:
        txn_feats["spend_prev30d"] = (txn_feats["spend_sum_60d"] - txn_feats["spend_sum_30d"]).clip(lower=0)
        txn_feats["txn_prev30d"] = (txn_feats["txn_count_60d"] - txn_feats["txn_count_30d"]).clip(lower=0)
        txn_feats["spend_trend_30d"] = txn_feats["spend_sum_30d"] - txn_feats["spend_prev30d"]
        txn_feats["txn_trend_30d"] = txn_feats["txn_count_30d"] - txn_feats["txn_prev30d"]
        txn_feats["spend_ratio_30d_prev30d"] = (
            txn_feats["spend_sum_30d"] / txn_feats["spend_prev30d"].replace(0, np.nan)
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # overall stats
    overall = hist.groupby("user_id").agg(
        unit_price_mean_all=("average_unit_price", "mean"),
        unit_price_std_all=("average_unit_price", "std"),
        spend_mean_all=("total_price", "mean"),
        spend_std_all=("total_price", "std"),
        unique_items_all=("jan_cd", "nunique"),
        unique_cat1_all=(cat_col, "nunique"),
        txn_count_all=("total_price", "size"),
        spend_sum_all=("total_price", "sum"),
    ).fillna(0.0)

    # channel ratios
    store_text = hist[["user_id","store_deli"]].copy()
    store_text["is_store"] = store_text["store_deli"].astype(str).str.contains("店").astype(int)
    store_text["is_delivery"] = store_text["store_deli"].astype(str).str.contains("宅|配|デリ").astype(int)
    store_ratio = store_text.groupby("user_id")[["is_store","is_delivery"]].mean().rename(
        columns={"is_store":"store_ratio_all", "is_delivery":"delivery_ratio_all"}
    )

    # category preference (top-k)
    cat_hist = hist.copy()
    if 90 in windows_days:
        cat_hist = cat_hist[cat_hist["date"] >= (ref_date - pd.Timedelta(days=90))]

    top_cats = (
        cat_hist[cat_col].dropna()
                .astype(int)
                .value_counts()
                .head(topk_cat).index.tolist()
    )

    cat_hist["cat_in_top"] = cat_hist[cat_col].where(cat_hist[cat_col].isin(top_cats), other=-1)
    cat_pivot = (
        cat_hist.pivot_table(
            index="user_id", columns="cat_in_top",
            values="total_price", aggfunc="size", fill_value=0
        )
    )
    cat_pivot.columns = [f"cat1_cnt_{int(c)}" for c in cat_pivot.columns]

    cat_counts = cat_pivot.to_numpy()
    total_cnt = cat_counts.sum(axis=1, keepdims=True)
    top_ratio = np.where(total_cnt > 0, cat_counts.max(axis=1, keepdims=True) / total_cnt, 0.0).ravel()
    entropy = [_category_entropy(cat_counts[i]) for i in range(cat_counts.shape[0])]
    cat_stats = pd.DataFrame(
        {"top_category_ratio_90d": top_ratio, "category_entropy_90d": entropy},
        index=cat_pivot.index
    )

    # seasonality (as-of ref_date)
    m = ref_date.month
    season = pd.DataFrame(index=profile.index)
    season["month"] = m
    season["month_sin"] = np.sin(2*np.pi*m/12.0)
    season["month_cos"] = np.cos(2*np.pi*m/12.0)
    season["is_bonus_season"] = int(m in (6, 12))
    season["is_year_end"] = int(m in (12, 1))

    X = (
        profile
        .join(recency, how="outer")
        .join(txn_feats, how="outer")
        .join(overall, how="outer")
        .join(store_ratio, how="outer")
        .join(cat_pivot, how="outer")
        .join(cat_stats, how="outer")
        .join(season, how="left")
    )

    # fill numeric NAs
    num_cols = X.select_dtypes(include=[np.number]).columns
    X[num_cols] = X[num_cols].fillna(0.0)

    # keep user_id index
    X.index.name = "user_id"
    return X


def build_label(df: pd.DataFrame, ref_date: pd.Timestamp, horizon_days=30, threshold=100) -> pd.Series:
    if not isinstance(ref_date, pd.Timestamp):
        ref_date = pd.Timestamp(ref_date)
    end_date = ref_date + pd.Timedelta(days=horizon_days)

    fut = df[(df["date"] >= ref_date) & (df["date"] < end_date)]
    y = fut.groupby("user_id")["total_price"].sum().ge(threshold).astype(int)
    y.name = "target"
    y.index.name = "user_id"
    return y


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Input CSV (transactions)")
    p.add_argument("--ref_date", required=True, help="Feature cutoff date, e.g. 2024-04-01")
    p.add_argument("--out_dir", default="out", help="Output directory")
    p.add_argument("--topk_cat", type=int, default=10, help="Top-K categories for preference features")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = load_transactions(args.input)
    ref_date = pd.Timestamp(args.ref_date)

    X = build_user_features(df, ref_date, topk_cat=args.topk_cat)
    feat_path = os.path.join(args.out_dir, f"features_{ref_date.strftime('%Y%m%d')}.csv")
    X.reset_index().to_csv(feat_path, index=False)

    # label may be empty if your input has no future period (e.g., test set)
    y = build_label(df, ref_date)
    lab_path = os.path.join(args.out_dir, f"labels_{ref_date.strftime('%Y%m%d')}.csv")
    y.reset_index().to_csv(lab_path, index=False)

    train_table = X.join(y, how="left").fillna({"target": 0}).reset_index()
    train_path = os.path.join(args.out_dir, f"train_table_{ref_date.strftime('%Y%m%d')}.csv")
    train_table.to_csv(train_path, index=False)

    print("DONE")
    print(f"features:   {feat_path}  shape={X.shape}")
    print(f"labels:     {lab_path}   positives={int(y.sum())} / users={y.shape[0]}")
    print(f"train_table:{train_path} shape={train_table.shape}")


if __name__ == "__main__":
    main()
