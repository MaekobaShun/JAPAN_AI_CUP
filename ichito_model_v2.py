# run_pipeline_dual_gate_v3.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple, Optional

_mpl_dir = Path(__file__).resolve().parent / ".matplotlib"
_mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(_mpl_dir)

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

# =========================
# CONFIG
# =========================
REF_DATE = pd.Timestamp("2025-02-03")

# model1: 現状
WINDOWS_M1 = [7, 14, 28, 56, 84]
# model2: 別視点（ここが肝）
WINDOWS_M2 = [7, 21, 45, 90]

# gating（モデル間の揉め具合でhard判定）
GATE_Q = 0.30     # hard上位30%（abs(p1-p2)が大きい）
W_HARD = 0.70     # hardはmodel2寄せ
W_EASY = 0.05     # easyはほぼmodel1

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
    raise FileNotFoundError("CSVが見つからない。\n" + tried)

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = get_project_root()
    data_path, train_flag_path, sub_path = _find_input_files(root)
    return pd.read_csv(data_path), pd.read_csv(train_flag_path), pd.read_csv(sub_path)

# =========================
# preprocessing
# =========================
def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["date"] = pd.to_datetime(data["date"].astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
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
# features (WINDOWS可変)
# =========================
def _window_agg(hist: pd.DataFrame, days: int, ref_date: pd.Timestamp) -> pd.DataFrame:
    start = ref_date - pd.Timedelta(days=days)
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
        agg_dict.update({
            "amount_sum": ("amount", "sum"),
            "amount_mean": ("amount", "mean"),
            "amount_max": ("amount", "max"),
        })
    g = w.groupby("user_id").agg(**agg_dict).reset_index()
    prefix = f"w{days}_"
    g.columns = ["user_id"] + [prefix + c for c in g.columns if c != "user_id"]
    return g

def _prev_window_agg(hist: pd.DataFrame, days: int, ref_date: pd.Timestamp) -> pd.DataFrame:
    end = ref_date - pd.Timedelta(days=days)
    start = ref_date - pd.Timedelta(days=2 * days)
    w = hist[(hist["date"] >= start) & (hist["date"] < end)]
    g = w.groupby("user_id").agg(
        prev_spend_sum=("total_price", "sum"),
        prev_visit_days=("date", "nunique"),
        prev_txn_count=("total_price", "size"),
    ).reset_index()
    g.columns = ["user_id", f"prev{days}_spend_sum", f"prev{days}_visit_days", f"prev{days}_txn_count"]
    return g

def _recency_features(hist: pd.DataFrame, ref_date: pd.Timestamp) -> pd.DataFrame:
    last_visit = hist.groupby("user_id")["date"].max().reset_index()
    last_visit.columns = ["user_id", "last_visit_date"]
    last_visit["days_since_last_purchase"] = (ref_date - last_visit["last_visit_date"]).dt.days
    return last_visit[["user_id", "days_since_last_purchase"]]

def _interval_features(hist: pd.DataFrame) -> pd.DataFrame:
    def per_user(d: pd.DataFrame) -> pd.Series:
        dates = np.sort(d["date"].dropna().unique())
        if len(dates) <= 1:
            return pd.Series({
                "avg_interval_days": np.nan,
                "std_interval_days": np.nan,
                "min_interval_days": np.nan,
                "max_interval_days": np.nan,
                "last_interval_days": np.nan,
                "last3_interval_mean": np.nan,
            })
        intervals = np.diff(dates) / np.timedelta64(1, "D")
        last_interval = float(intervals[-1])
        last3 = intervals[-3:] if len(intervals) >= 3 else intervals
        return pd.Series({
            "avg_interval_days": float(np.mean(intervals)),
            "std_interval_days": float(np.std(intervals)) if len(intervals) >= 2 else np.nan,
            "min_interval_days": float(np.min(intervals)),
            "max_interval_days": float(np.max(intervals)),
            "last_interval_days": last_interval,
            "last3_interval_mean": float(np.mean(last3)) if len(last3) > 0 else np.nan,
        })
    return hist.groupby("user_id").apply(per_user).reset_index()

def create_features(data: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
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
    for d in windows:
        feats = pd.merge(feats, _window_agg(hist, d, REF_DATE), on="user_id", how="left")

    # 28がある場合だけ差分（model2が28無いなら差分無し、別視点にする）
    if 28 in windows:
        prev28 = _prev_window_agg(hist, 28, REF_DATE)
        feats = pd.merge(feats, prev28, on="user_id", how="left")
        feats["diff28_visit_days"] = feats["w28_visit_days"] - feats["prev28_visit_days"]
        feats["ratio28_visit_days"] = (feats["w28_visit_days"] + 1.0) / (feats["prev28_visit_days"] + 1.0)

    feats = pd.merge(feats, _recency_features(hist, REF_DATE), on="user_id", how="left")
    feats = pd.merge(feats, _interval_features(hist), on="user_id", how="left")

    for c in [col for col in feats.columns if col.startswith("w") or col.startswith("prev")]:
        feats[c] = feats[c].fillna(0)

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
# CV with OOF + test pred
# =========================
def train_lgb_cv_with_oof(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    out_dir: Path,
    tag: str,
) -> Tuple[np.ndarray, np.ndarray, float]:
    ensure_dir(out_dir)
    skf = StratifiedKFold(
        n_splits=CV_CONFIG["n_splits"],
        shuffle=CV_CONFIG["shuffle"],
        random_state=CV_CONFIG["random_state"],
    )

    X = X_train.copy()
    Xt = X_test.copy()

    for c in X.columns:
        if X[c].dtype == "object":
            X[c] = X[c].astype("category")
            if c in Xt.columns:
                Xt[c] = Xt[c].astype("category")

    oof = np.zeros(len(X), dtype=float)
    test_pred = np.zeros(len(Xt), dtype=float)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y_train), 1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

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
        test_pred += model.predict(Xt, num_iteration=model.best_iteration) / skf.n_splits

        auc = roc_auc_score(y_va, pred_va)
        print(f"[{tag} fold {fold}] valid AUC: {auc:.6f} best_iter={model.best_iteration}")

    auc_all = roc_auc_score(y_train, oof)
    print(f"[{tag}] OOF AUC: {auc_all:.6f}")

    pd.DataFrame({"row_id": np.arange(len(oof)), "y": y_train.values, "oof_pred": oof}).to_csv(out_dir / f"{tag}_oof.csv", index=False)
    pd.DataFrame({"row_id": np.arange(len(test_pred)), "test_pred": test_pred}).to_csv(out_dir / f"{tag}_test_pred.csv", index=False)

    return oof, test_pred, auc_all

# =========================
# gating: abs(p1-p2)
# =========================
def gated_blend_by_disagreement(p1: np.ndarray, p2: np.ndarray, gate_q: float, w_hard: float, w_easy: float):
    gate = np.abs(p1 - p2)  # 大きいほど揉めてる
    thr = float(np.quantile(gate, 1.0 - gate_q))  # 上位gate_qがhard
    is_hard = gate >= thr
    w = np.where(is_hard, w_hard, w_easy).astype(float)
    p = (1.0 - w) * p1 + w * p2
    return p, float(is_hard.mean()), thr

# =========================
# main
# =========================
def main():
    out_root = ensure_dir(get_output_path("dual_gate_v3"))
    sub_dir = ensure_dir(get_output_path("submissions"))

    print(f"REF_DATE: {REF_DATE.date()}")
    print(f"WINDOWS_M1: {WINDOWS_M1}")
    print(f"WINDOWS_M2: {WINDOWS_M2}")
    print(f"GATE_Q: {GATE_Q}  W_HARD: {W_HARD}  W_EASY: {W_EASY}")

    print("\n[1/4] Loading data...")
    data, train_flag, sample_submission = load_data()

    print("[2/4] Preprocessing...")
    data = clean_data(data)

    print("[3/4] Features (model1/model2)...")
    feats1 = create_features(data, WINDOWS_M1)
    X1_train, y_train = prepare_train_data(train_flag, feats1)
    X1_test = prepare_test_data(sample_submission, feats1)

    feats2 = create_features(data, WINDOWS_M2)
    X2_train, _ = prepare_train_data(train_flag, feats2)
    X2_test = prepare_test_data(sample_submission, feats2)

    pred_col = "pred" if ("pred" in sample_submission.columns) else sample_submission.columns[-1]

    print("\n[4/4] Train model1...")
    oof1, test1, auc1 = train_lgb_cv_with_oof(X1_train, y_train, X1_test, out_root, "model1")

    print("\nTrain model2...")
    oof2, test2, auc2 = train_lgb_cv_with_oof(X2_train, y_train, X2_test, out_root, "model2")

    # OOF blend
    oof_blend, hard_rate, thr = gated_blend_by_disagreement(oof1, oof2, GATE_Q, W_HARD, W_EASY)
    auc_blend = roc_auc_score(y_train, oof_blend)

    print("\n[OOF gated blend by disagreement]")
    print(f"  hard_rate: {hard_rate*100:.2f}%  (abs(p1-p2) >= thr)")
    print(f"  model1 AUC: {auc1:.6f}")
    print(f"  model2 AUC: {auc2:.6f}")
    print(f"  blend  AUC: {auc_blend:.6f}")

    pd.DataFrame({
        "row_id": np.arange(len(oof_blend)),
        "y": y_train.values,
        "oof1": oof1,
        "oof2": oof2,
        "gate_abs_p1_p2": np.abs(oof1 - oof2),
        "oof_blend": oof_blend,
    }).to_csv(out_root / "oof_blend.csv", index=False)

    # test blend（同じgate_q方式でOK）
    test_blend, hard_rate_test, _ = gated_blend_by_disagreement(test1, test2, GATE_Q, W_HARD, W_EASY)
    print("\n[Test gated blend]")
    print(f"  hard_rate_test: {hard_rate_test*100:.2f}%")

    sub = sample_submission.copy()
    sub[pred_col] = test_blend
    sub_path = sub_dir / "sub_dual_gate_v3.csv"
    sub.to_csv(sub_path, index=False)

    print(f"\nSubmission created: {sub_path}")
    print(f"Artifacts saved under: {out_root}")
    print("Done.")

if __name__ == "__main__":
    main()