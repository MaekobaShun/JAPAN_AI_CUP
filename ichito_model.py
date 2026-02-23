# run_pipeline.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict

# avoid permission error for matplotlib cache directory on Windows
_mpl_dir = Path(__file__).resolve().parent / ".matplotlib"
_mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(_mpl_dir)

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import matplotlib.pyplot as plt


# =========================
# CONFIG
# =========================
REF_DATE = pd.Timestamp("2025-02-03")

# ここをいじって window を整理する
WINDOWS = [7, 14, 28, 56, 84]   # 例: [7, 21, 45, 90] とかに変える

# SHAPで削る閾値（雑に強めでいい）
# 例: 下位30%を削る / mean_abs_shap < 1e-4 を削る など
SHAP_DROP_MODE = "quantile"     # "quantile" or "threshold"
SHAP_DROP_Q = 0.30              # 下位30%削る（強め）
SHAP_DROP_THRESHOLD = 1e-4      # thresholdモード用

# ほぼ欠損 or ほぼ定数 を落とす（ノイズ削除）
MISSING_RATE_MAX = 0.995        # 99.5%欠損以上は捨てる
LOW_VARIANCE_EPS = 1e-12        # 分散がこれ以下は捨てる（数値列）

# LightGBM base params（まずはシンプル）
LGB_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 5.0,
    "seed": 42,
    "verbosity": -1,
}

CV_CONFIG = {
    "n_splits": 5,
    "shuffle": True,
    "random_state": 42,
    "num_boost_round": 12000,
    "early_stopping_rounds": 600,
    "log_evaluation_period": 200,
}

DATA_CANDIDATES = [
    ("input/data.csv", "input/train_flag.csv", "input/sample_submission.csv"),
    ("inputs/data.csv", "inputs/train_flag.csv", "inputs/sample_submission.csv"),
    ("data/raw/data.csv", "data/raw/train_flag.csv", "data/raw/sample_submission.csv"),
    ("raw/data.csv", "raw/train_flag.csv", "raw/sample_submission.csv"),
]


# =========================
# utils
# =========================
def get_project_root() -> Path:
    return Path(__file__).resolve().parent


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_output_path(name: str) -> Path:
    return get_project_root() / "output" / name


def _find_input_files(project_root: Path) -> Tuple[Path, Path, Path]:
    for a, b, c in DATA_CANDIDATES:
        pa, pb, pc = project_root / a, project_root / b, project_root / c
        if pa.exists() and pb.exists() and pc.exists():
            return pa, pb, pc
    tried = "\n".join([f"- {x[0]}, {x[1]}, {x[2]}" for x in DATA_CANDIDATES])
    raise FileNotFoundError(
        "CSVが見つからない。\n"
        "DATA_CANDIDATES を確認して配置を修正して。\n"
        f"{tried}"
    )


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = get_project_root()
    data_path, train_flag_path, sub_path = _find_input_files(root)
    data = pd.read_csv(data_path)
    train_flag = pd.read_csv(train_flag_path)
    sample_submission = pd.read_csv(sub_path)
    return data, train_flag, sample_submission


# =========================
# preprocessing
# =========================
def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    if "date" not in data.columns:
        raise KeyError("data is missing 'date' column.")
    if "user_id" not in data.columns:
        raise KeyError("data is missing 'user_id' column.")
    if "total_price" not in data.columns:
        raise KeyError("data is missing 'total_price' column.")

    # date
    if np.issubdtype(data["date"].dtype, np.number):
        data["date"] = pd.to_datetime(
            data["date"].astype("Int64").astype(str),
            format="%Y%m%d",
            errors="coerce",
        )
    else:
        data["date"] = pd.to_datetime(data["date"], errors="coerce")

    # numeric
    data["total_price"] = pd.to_numeric(data["total_price"], errors="coerce").fillna(0)

    if "amount" in data.columns:
        data["amount"] = pd.to_numeric(data["amount"], errors="coerce")

    if "average_unit_price" not in data.columns:
        if "amount" in data.columns:
            amt = data["amount"].replace(0, np.nan)
            data["average_unit_price"] = data["total_price"] / amt
        else:
            data["average_unit_price"] = np.nan
    data["average_unit_price"] = pd.to_numeric(data["average_unit_price"], errors="coerce")

    return data


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace(0, np.nan)


# =========================
# features
# =========================
def _window_agg(hist: pd.DataFrame, days: int) -> pd.DataFrame:
    start = REF_DATE - pd.Timedelta(days=days)
    w = hist[hist["date"] >= start]

    agg_dict = {
        "txn_count": ("total_price", "size"),
        "visit_days": ("date", "nunique"),
        "spend_sum": ("total_price", "sum"),
        "spend_mean": ("total_price", "mean"),
        "spend_std": ("total_price", "std"),
        "spend_max": ("total_price", "max"),
        "unit_price_mean": ("average_unit_price", "mean"),
        "unit_price_std": ("average_unit_price", "std"),
    }
    if "amount" in w.columns:
        agg_dict.update(
            {
                "amount_sum": ("amount", "sum"),
                "amount_mean": ("amount", "mean"),
                "amount_max": ("amount", "max"),
            }
        )

    g = w.groupby("user_id").agg(**agg_dict).reset_index()
    prefix = f"w{days}_"
    g.columns = ["user_id"] + [prefix + c for c in g.columns if c != "user_id"]
    return g


def _prev_window_agg(hist: pd.DataFrame, days: int) -> pd.DataFrame:
    end = REF_DATE - pd.Timedelta(days=days)
    start = REF_DATE - pd.Timedelta(days=2 * days)
    w = hist[(hist["date"] >= start) & (hist["date"] < end)]

    g = w.groupby("user_id").agg(
        prev_spend_sum=("total_price", "sum"),
        prev_visit_days=("date", "nunique"),
        prev_txn_count=("total_price", "size"),
    ).reset_index()

    g.columns = ["user_id", f"prev{days}_spend_sum", f"prev{days}_visit_days", f"prev{days}_txn_count"]
    return g


def _recency_features(hist: pd.DataFrame) -> pd.DataFrame:
    last_visit = hist.groupby("user_id")["date"].max().reset_index()
    last_visit.columns = ["user_id", "last_visit_date"]
    last_visit["days_since_last_purchase"] = (REF_DATE - last_visit["last_visit_date"]).dt.days
    return last_visit[["user_id", "days_since_last_purchase"]]


def _interval_features(hist: pd.DataFrame) -> pd.DataFrame:
    def per_user(d: pd.DataFrame) -> pd.Series:
        dates = np.sort(d["date"].dropna().unique())
        if len(dates) <= 1:
            return pd.Series(
                {
                    "avg_interval_days": np.nan,
                    "std_interval_days": np.nan,
                    "min_interval_days": np.nan,
                    "max_interval_days": np.nan,
                    "last_interval_days": np.nan,
                    "last3_interval_mean": np.nan,
                }
            )
        intervals = np.diff(dates) / np.timedelta64(1, "D")
        last_interval = float(intervals[-1]) if len(intervals) >= 1 else np.nan
        last3 = intervals[-3:] if len(intervals) >= 3 else intervals
        return pd.Series(
            {
                "avg_interval_days": float(np.mean(intervals)),
                "std_interval_days": float(np.std(intervals)) if len(intervals) >= 2 else np.nan,
                "min_interval_days": float(np.min(intervals)),
                "max_interval_days": float(np.max(intervals)),
                "last_interval_days": last_interval,
                "last3_interval_mean": float(np.mean(last3)) if len(last3) > 0 else np.nan,
            }
        )

    feat = hist.groupby("user_id").apply(per_user).reset_index()
    return feat


def create_features(data: pd.DataFrame) -> pd.DataFrame:
    hist = data[data["date"] < REF_DATE].copy()

    base = hist.groupby("user_id").agg(
        txn_count_all=("total_price", "size"),
        visit_days_all=("date", "nunique"),
        spend_sum_all=("total_price", "sum"),
        spend_mean_all=("total_price", "mean"),
        spend_std_all=("total_price", "std"),
        spend_max_all=("total_price", "max"),
        unit_price_mean_all=("average_unit_price", "mean"),
        unit_price_std_all=("average_unit_price", "std"),
    ).reset_index()

    if "amount" in hist.columns:
        amt = hist.groupby("user_id").agg(
            amount_sum_all=("amount", "sum"),
            amount_mean_all=("amount", "mean"),
            amount_max_all=("amount", "max"),
        ).reset_index()
        base = pd.merge(base, amt, on="user_id", how="left")

    base["txn_per_visit_all"] = _safe_div(base["txn_count_all"], base["visit_days_all"])
    base["spend_per_visit_all"] = _safe_div(base["spend_sum_all"], base["visit_days_all"])

    feats = base

    # windows
    for d in WINDOWS:
        feats = pd.merge(feats, _window_agg(hist, d), on="user_id", how="left")

    # prev window（28が存在する前提で差分を作るので、WINDOWSに28が無いなら prev28は作らない）
    if 28 in WINDOWS:
        prev28 = _prev_window_agg(hist, 28)
        feats = pd.merge(feats, prev28, on="user_id", how="left")

        feats["diff28_spend"] = feats["w28_spend_sum"] - feats["prev28_spend_sum"]
        feats["ratio28_spend"] = (feats["w28_spend_sum"] + 1.0) / (feats["prev28_spend_sum"] + 1.0)

        feats["diff28_visit_days"] = feats["w28_visit_days"] - feats["prev28_visit_days"]
        feats["ratio28_visit_days"] = (feats["w28_visit_days"] + 1.0) / (feats["prev28_visit_days"] + 1.0)

        feats["diff28_txn"] = feats["w28_txn_count"] - feats["prev28_txn_count"]
        feats["ratio28_txn"] = (feats["w28_txn_count"] + 1.0) / (feats["prev28_txn_count"] + 1.0)

    # recency + interval
    feats = pd.merge(feats, _recency_features(hist), on="user_id", how="left")
    feats = pd.merge(feats, _interval_features(hist), on="user_id", how="left")

    # fill (window系だけゼロ埋め)
    for c in [col for col in feats.columns if col.startswith("w") or col.startswith("prev")]:
        feats[c] = feats[c].fillna(0)

    # inf -> nan
    feats = feats.replace([np.inf, -np.inf], np.nan)

    return feats


def prepare_train_data(train_flag: pd.DataFrame, features: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.merge(train_flag[["user_id", "churn"]], features, on="user_id", how="left", sort=False)
    df = df.set_index("user_id").loc[train_flag["user_id"]].reset_index()

    y = df["churn"].astype(int)
    X = df.drop(["user_id", "churn"], axis=1)
    return X, y


def prepare_test_data(sample_submission: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    df = pd.merge(sample_submission[["user_id"]], features, on="user_id", how="left", sort=False)
    df = df.set_index("user_id").loc[sample_submission["user_id"]].reset_index()
    X = df.drop(["user_id"], axis=1)
    return X


# =========================
# feature pruning
# =========================
def drop_noise_features(X: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    drop_cols: List[str] = []

    # too many missing
    miss_rate = X.isna().mean()
    drop_cols += miss_rate[miss_rate >= MISSING_RATE_MAX].index.tolist()

    # low variance (numeric only)
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    if len(num_cols) > 0:
        var = X[num_cols].var(axis=0, skipna=True)
        drop_cols += var[var <= LOW_VARIANCE_EPS].index.tolist()

    drop_cols = sorted(set(drop_cols))
    X2 = X.drop(columns=drop_cols, errors="ignore")
    return X2, drop_cols


def train_lgb_cv(X: pd.DataFrame, y: pd.Series) -> Tuple[List[lgb.Booster], np.ndarray, float]:
    skf = StratifiedKFold(
        n_splits=CV_CONFIG["n_splits"],
        shuffle=CV_CONFIG["shuffle"],
        random_state=CV_CONFIG["random_state"],
    )

    X = X.copy()
    for c in X.columns:
        if X[c].dtype == "object":
            X[c] = X[c].astype("category")

    oof = np.zeros(len(X), dtype=float)
    models: List[lgb.Booster] = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        dtrain = lgb.Dataset(X_tr, label=y_tr, free_raw_data=False)
        dvalid = lgb.Dataset(X_va, label=y_va, free_raw_data=False)

        model = lgb.train(
            params=LGB_PARAMS,
            train_set=dtrain,
            valid_sets=[dtrain, dvalid],
            valid_names=["train", "valid"],
            num_boost_round=CV_CONFIG["num_boost_round"],
            callbacks=[
                lgb.early_stopping(CV_CONFIG["early_stopping_rounds"], verbose=False),
                lgb.log_evaluation(CV_CONFIG["log_evaluation_period"]),
            ],
        )
        pred_va = model.predict(X_va, num_iteration=model.best_iteration)
        oof[va_idx] = pred_va
        models.append(model)

        auc = roc_auc_score(y_va, pred_va)
        print(f"[fold {fold}] valid AUC: {auc:.6f} best_iter={model.best_iteration}")

    auc_all = roc_auc_score(y, oof)
    return models, oof, auc_all


def shap_importance(models: List[lgb.Booster], X: pd.DataFrame, out_dir: Path, sample_size: int = 1500) -> pd.DataFrame:
    ensure_dir(out_dir)
    import shap

    Xs = X.sample(min(sample_size, len(X)), random_state=42)
    explainer = shap.TreeExplainer(models[0])
    sv = explainer.shap_values(Xs)

    if isinstance(sv, list):
        sv = sv[1] if len(sv) > 1 else sv[0]

    mean_abs = np.abs(sv).mean(axis=0)
    imp = pd.DataFrame({"feature": X.columns.tolist(), "mean_abs_shap": mean_abs})
    imp = imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    imp.to_csv(out_dir / "shap_importance.csv", index=False)

    # plot top30
    top = imp.head(30).iloc[::-1]
    plt.figure()
    plt.barh(top["feature"], top["mean_abs_shap"])
    plt.title("Top SHAP importance (mean |SHAP|)")
    plt.tight_layout()
    plt.savefig(out_dir / "shap_top30.png", dpi=150)
    plt.close()

    # plot bottom30
    bot = imp.tail(30)
    plt.figure()
    plt.barh(bot["feature"], bot["mean_abs_shap"])
    plt.title("Bottom SHAP importance (mean |SHAP|)")
    plt.tight_layout()
    plt.savefig(out_dir / "shap_bottom30.png", dpi=150)
    plt.close()

    return imp


def select_drop_by_shap(imp: pd.DataFrame) -> List[str]:
    if SHAP_DROP_MODE == "threshold":
        return imp.loc[imp["mean_abs_shap"] < SHAP_DROP_THRESHOLD, "feature"].tolist()
    # quantile
    q = float(imp["mean_abs_shap"].quantile(SHAP_DROP_Q))
    return imp.loc[imp["mean_abs_shap"] <= q, "feature"].tolist()


# =========================
# main
# =========================
def main():
    root = get_project_root()
    out_root = ensure_dir(get_output_path("shap_prune"))
    sub_dir = ensure_dir(get_output_path("submissions"))

    print(f"REF_DATE: {REF_DATE.date()}")
    print(f"WINDOWS: {WINDOWS}")

    print("\n[1/5] Loading data...")
    data, train_flag, sample_submission = load_data()

    print("[2/5] Preprocessing...")
    data = clean_data(data)

    print("[3/5] Creating features...")
    feats = create_features(data)
    X_train, y_train = prepare_train_data(train_flag, feats)
    X_test = prepare_test_data(sample_submission, feats)
    pred_col = "pred" if ("pred" in sample_submission.columns) else sample_submission.columns[-1]

    print(f"Train shape: {X_train.shape}  Test shape: {X_test.shape}")

    # ---- step0: noise drop
    X_train0, dropped0 = drop_noise_features(X_train)
    X_test0 = X_test.drop(columns=dropped0, errors="ignore")
    print(f"\n[Noise drop] removed {len(dropped0)} cols")
    if len(dropped0) > 0:
        (out_root / "dropped_noise.txt").write_text("\n".join(dropped0), encoding="utf-8")

    # ---- pass1 train
    print("\n[4/5] Train base LGBM (pass1)...")
    models1, oof1, auc1 = train_lgb_cv(X_train0, y_train)
    print(f"OOF AUC (pass1): {auc1:.6f}")

    # shap1
    print("\nSHAP importance (pass1)...")
    imp1 = shap_importance(models1, X_train0, out_root / "pass1", sample_size=1500)

    drop1 = select_drop_by_shap(imp1)
    print(f"[SHAP drop] will drop {len(drop1)} cols (mode={SHAP_DROP_MODE})")
    (out_root / "dropped_shap_pass1.txt").write_text("\n".join(drop1), encoding="utf-8")

    # ---- pass2 train (after shap drop)
    X_train2 = X_train0.drop(columns=drop1, errors="ignore")
    X_test2 = X_test0.drop(columns=drop1, errors="ignore")

    print("\n[5/5] Train pruned LGBM (pass2)...")
    models2, oof2, auc2 = train_lgb_cv(X_train2, y_train)
    print(f"OOF AUC (pass2): {auc2:.6f}  (delta={auc2-auc1:+.6f})")

    # shap2 (final)
    print("\nSHAP importance (pass2)...")
    _ = shap_importance(models2, X_train2, out_root / "pass2", sample_size=1500)

    # predict test (avg folds)
    preds_test = np.zeros(len(X_test2), dtype=float)
    for m in models2:
        preds_test += m.predict(X_test2, num_iteration=m.best_iteration)
    preds_test /= len(models2)

    sub = sample_submission.copy()
    sub[pred_col] = preds_test
    sub_path = sub_dir / "sub_shap_pruned_lgb.csv"
    sub.to_csv(sub_path, index=False)
    print(f"\nSubmission created: {sub_path}")
    print("Done.")


if __name__ == "__main__":
    main()