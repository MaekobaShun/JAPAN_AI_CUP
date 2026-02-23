# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

# Avoid matplotlib cache permission issues on Windows
_mpl_dir = Path(__file__).resolve().parent / ".matplotlib"
_mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(_mpl_dir)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


REF_DATE = pd.Timestamp("2025-02-03")
WINDOWS = [7, 14, 30, 60, 90]
CORR_METHOD = "pearson"

DATA_CANDIDATES = [
    ("input/data.csv", "input/train_flag.csv", "input/sample_submission.csv"),
    ("inputs/data.csv", "inputs/train_flag.csv", "inputs/sample_submission.csv"),
    ("data/raw/data.csv", "data/raw/train_flag.csv", "data/raw/sample_submission.csv"),
    ("raw/data.csv", "raw/train_flag.csv", "raw/sample_submission.csv"),
]

ITEM_COL_CANDIDATES = ["item_id", "product_id", "jan_code", "sku_id", "item_code"]
CAT1_COL_CANDIDATES = ["cat1", "category1", "category_1", "cat_1", "major_category"]
AMOUNT_COL_CANDIDATES = ["amount", "qty", "quantity"]


def get_project_root() -> Path:
    return Path(__file__).resolve().parent


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _find_data_file(project_root: Path) -> Path:
    for data_rel, _, _ in DATA_CANDIDATES:
        p = project_root / data_rel
        if p.exists():
            return p
    tried = "\n".join([f"- {x[0]}" for x in DATA_CANDIDATES])
    raise FileNotFoundError(f"data.csv not found. Tried:\n{tried}")


def _first_existing_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    required = ["user_id", "date", "total_price"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    if np.issubdtype(data["date"].dtype, np.number):
        data["date"] = pd.to_datetime(
            data["date"].astype("Int64").astype(str), format="%Y%m%d", errors="coerce"
        )
    else:
        data["date"] = pd.to_datetime(data["date"], errors="coerce")

    data["total_price"] = pd.to_numeric(data["total_price"], errors="coerce").fillna(0.0)

    amount_col = _first_existing_column(data, AMOUNT_COL_CANDIDATES)
    if amount_col is not None:
        data["amount"] = pd.to_numeric(data[amount_col], errors="coerce")
    else:
        data["amount"] = np.nan

    if "average_unit_price" not in data.columns:
        if amount_col is not None:
            denom = data["amount"].replace(0, np.nan)
            data["average_unit_price"] = data["total_price"] / denom
        else:
            data["average_unit_price"] = np.nan
    data["average_unit_price"] = pd.to_numeric(data["average_unit_price"], errors="coerce")

    return data


def _window_agg(hist: pd.DataFrame, days: int, amount_exists: bool) -> pd.DataFrame:
    start = REF_DATE - pd.Timedelta(days=days)
    w = hist[hist["date"] >= start]

    g = w.groupby("user_id").agg(
        **{
            f"txn_count_{days}d": ("total_price", "size"),
            f"spend_sum_{days}d": ("total_price", "sum"),
        }
    ).reset_index()

    if amount_exists:
        g_amt = w.groupby("user_id").agg(
            **{f"amount_sum_{days}d": ("amount", "sum")}
        ).reset_index()
        g = pd.merge(g, g_amt, on="user_id", how="left")
        g[f"avg_unit_price_{days}d"] = g[f"spend_sum_{days}d"] / g[f"amount_sum_{days}d"].replace(0, np.nan)
    else:
        g_price = w.groupby("user_id").agg(
            **{f"avg_unit_price_{days}d": ("average_unit_price", "mean")}
        ).reset_index()
        g = pd.merge(g, g_price, on="user_id", how="left")
        g[f"amount_sum_{days}d"] = np.nan

    return g


def _interval_stats(hist: pd.DataFrame) -> pd.DataFrame:
    def per_user(d: pd.DataFrame) -> pd.Series:
        dates = np.sort(d["date"].dropna().unique())
        if len(dates) <= 1:
            return pd.Series(
                {
                    "mean_purchase_interval": np.nan,
                    "std_purchase_interval": np.nan,
                    "last_purchase_interval": np.nan,
                }
            )
        intervals = np.diff(dates) / np.timedelta64(1, "D")
        return pd.Series(
            {
                "mean_purchase_interval": float(np.mean(intervals)),
                "std_purchase_interval": float(np.std(intervals)) if len(intervals) >= 2 else np.nan,
                "last_purchase_interval": float(intervals[-1]),
            }
        )

    return hist.groupby("user_id").apply(per_user).reset_index()


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    hist = data[data["date"] < REF_DATE].copy()
    amount_exists = hist["amount"].notna().any()

    item_col = _first_existing_column(hist, ITEM_COL_CANDIDATES)
    cat1_col = _first_existing_column(hist, CAT1_COL_CANDIDATES)

    base_agg = {
        "txn_count_all": ("total_price", "size"),
        "spend_sum_all": ("total_price", "sum"),
        "avg_unit_price_all": ("average_unit_price", "mean"),
    }
    if amount_exists:
        base_agg["amount_sum_all"] = ("amount", "sum")

    base = hist.groupby("user_id").agg(**base_agg).reset_index()
    if not amount_exists:
        base["amount_sum_all"] = np.nan

    if item_col is not None:
        item_df = hist.groupby("user_id")[item_col].nunique().reset_index(name="unique_items_all")
        base = pd.merge(base, item_df, on="user_id", how="left")
    else:
        base["unique_items_all"] = np.nan

    if cat1_col is not None:
        cat_df = hist.groupby("user_id")[cat1_col].nunique().reset_index(name="unique_cat1_all")
        base = pd.merge(base, cat_df, on="user_id", how="left")
    else:
        base["unique_cat1_all"] = np.nan

    last_visit = hist.groupby("user_id")["date"].max().reset_index()
    last_visit["days_since_last_purchase"] = (REF_DATE - last_visit["date"]).dt.days
    last_visit = last_visit[["user_id", "days_since_last_purchase"]]
    feats = pd.merge(base, last_visit, on="user_id", how="left")

    for d in WINDOWS:
        w = _window_agg(hist, d, amount_exists=amount_exists)
        feats = pd.merge(feats, w, on="user_id", how="left")

    cur30 = hist[hist["date"] >= REF_DATE - pd.Timedelta(days=30)]
    prev30 = hist[(hist["date"] >= REF_DATE - pd.Timedelta(days=60)) & (hist["date"] < REF_DATE - pd.Timedelta(days=30))]

    cur30_agg = cur30.groupby("user_id").agg(
        cur_spend_30d=("total_price", "sum"),
        cur_txn_30d=("total_price", "size"),
    ).reset_index()
    prev30_agg = prev30.groupby("user_id").agg(
        spend_prev30d=("total_price", "sum"),
        txn_prev30d=("total_price", "size"),
    ).reset_index()

    feats = pd.merge(feats, cur30_agg, on="user_id", how="left")
    feats = pd.merge(feats, prev30_agg, on="user_id", how="left")

    feats["spend_prev30d"] = feats["spend_prev30d"].fillna(0.0)
    feats["txn_prev30d"] = feats["txn_prev30d"].fillna(0.0)
    feats["cur_spend_30d"] = feats["cur_spend_30d"].fillna(0.0)
    feats["cur_txn_30d"] = feats["cur_txn_30d"].fillna(0.0)

    feats["spend_trend_30d"] = feats["cur_spend_30d"] - feats["spend_prev30d"]
    feats["txn_trend_30d"] = feats["cur_txn_30d"] - feats["txn_prev30d"]
    feats["spend_ratio_30d_prev30d"] = (feats["cur_spend_30d"] + 1.0) / (feats["spend_prev30d"] + 1.0)
    feats["txn_ratio_30d_prev30d"] = (feats["cur_txn_30d"] + 1.0) / (feats["txn_prev30d"] + 1.0)
    feats["activity_decay_30d_over_90d"] = (feats["txn_count_30d"] + 1.0) / (feats["txn_count_90d"] + 1.0)

    interval_df = _interval_stats(hist)
    feats = pd.merge(feats, interval_df, on="user_id", how="left")

    feats = feats.replace([np.inf, -np.inf], np.nan)
    feats = feats.drop(columns=["cur_spend_30d", "cur_txn_30d"], errors="ignore")
    return feats


def target_feature_names() -> List[str]:
    cols: List[str] = [
        "days_since_last_purchase",
        "txn_count_all",
        "spend_sum_all",
        "amount_sum_all",
        "avg_unit_price_all",
        "unique_items_all",
        "unique_cat1_all",
    ]
    for d in WINDOWS:
        cols.extend(
            [
                f"txn_count_{d}d",
                f"spend_sum_{d}d",
                f"amount_sum_{d}d",
                f"avg_unit_price_{d}d",
            ]
        )
    cols.extend(
        [
            "spend_prev30d",
            "txn_prev30d",
            "spend_trend_30d",
            "txn_trend_30d",
            "spend_ratio_30d_prev30d",
            "txn_ratio_30d_prev30d",
            "activity_decay_30d_over_90d",
            "mean_purchase_interval",
            "std_purchase_interval",
            "last_purchase_interval",
        ]
    )
    return cols


def plot_corr_heatmap(corr: pd.DataFrame, out_png: Path) -> None:
    n = len(corr.columns)
    fig_size = max(10, int(n * 0.45))

    plt.figure(figsize=(fig_size, fig_size))
    im = plt.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04, label=f"{CORR_METHOD} correlation")
    plt.xticks(range(n), corr.columns, rotation=90, fontsize=8)
    plt.yticks(range(n), corr.columns, fontsize=8)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def make_pairwise_corr_table(corr: pd.DataFrame) -> pd.DataFrame:
    pairs = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v = corr.iat[i, j]
            pairs.append(
                {
                    "feature_a": cols[i],
                    "feature_b": cols[j],
                    "corr": float(v),
                    "abs_corr": float(abs(v)),
                }
            )
    out = pd.DataFrame(pairs).sort_values("abs_corr", ascending=False).reset_index(drop=True)
    return out


def main() -> None:
    root = get_project_root()
    out_dir = ensure_dir(root / "output" / "feature_correlation")

    data_path = _find_data_file(root)
    print(f"Loading data: {data_path}")
    data = pd.read_csv(data_path)
    data = clean_data(data)

    feats = build_features(data)
    requested_cols = target_feature_names()
    use_cols = [c for c in requested_cols if c in feats.columns]
    missing_cols = [c for c in requested_cols if c not in feats.columns]

    if missing_cols:
        print(f"Missing requested features ({len(missing_cols)}):")
        for c in missing_cols:
            print(f"  - {c}")

    corr_input = feats[use_cols].copy()

    # Drop constant columns because correlation becomes undefined
    non_constant_cols = [c for c in corr_input.columns if corr_input[c].nunique(dropna=True) > 1]
    dropped_constant = [c for c in corr_input.columns if c not in non_constant_cols]
    if dropped_constant:
        print(f"Dropped constant features ({len(dropped_constant)}):")
        for c in dropped_constant:
            print(f"  - {c}")

    if len(non_constant_cols) < 2:
        raise RuntimeError("Not enough non-constant features to compute correlation.")

    corr = corr_input[non_constant_cols].corr(method=CORR_METHOD)
    corr_csv = out_dir / "feature_corr_matrix.csv"
    heatmap_png = out_dir / "feature_corr_heatmap.png"
    pair_csv = out_dir / "feature_corr_pairs_sorted.csv"
    feat_csv = out_dir / "feature_table_for_corr.csv"

    corr.to_csv(corr_csv)
    plot_corr_heatmap(corr, heatmap_png)

    pair_df = make_pairwise_corr_table(corr)
    pair_df.to_csv(pair_csv, index=False)

    feats[["user_id"] + use_cols].to_csv(feat_csv, index=False)

    print("Saved:")
    print(f"  - {corr_csv}")
    print(f"  - {heatmap_png}")
    print(f"  - {pair_csv}")
    print(f"  - {feat_csv}")

    print("\nTop 15 absolute correlations:")
    print(pair_df.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
