# run_pipeline_dynamic_uncertainty_ensemble_v1.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Optional

_mpl_dir = Path(__file__).resolve().parent / ".matplotlib"
_mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(_mpl_dir)

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# =========================
# CONFIG
# =========================
REF_DATE = pd.Timestamp("2025-02-03")

# train_flag の目的変数列名
TARGET_COL = "churn"  # <- ここが違うなら変える（例: "target"）

LGB_BASE_PARAMS = {
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

OUT_DIR_NAME = "dynamic_uncertainty_ensemble_v1"


# =========================
# MODEL SPECS（増やすならここ）
# =========================
@dataclass
class ModelSpec:
    tag: str
    windows: List[int]
    feature_set: str
    lgb_overrides: Dict[str, object]

# [0] が「強いモデル（メイン）」扱いになる（後で重みの基準にもなる）
MODEL_SPECS: List[ModelSpec] = [
    ModelSpec("m0_R_only", [7, 14, 28, 30, 60, 90], "R_ONLY", {}),
    ModelSpec("m1_volume_trend_noR", [7, 14, 28, 30, 60, 90], "VOL_TREND", {"min_data_in_leaf": 120}),
    ModelSpec("m2_interval_unitprice_noR", [7, 14, 28, 30, 60, 90], "INT_PRICE", {"num_leaves": 63, "lambda_l2": 8.0}),
    # 追加例：
    # ModelSpec("m3_volume_trend_alt", [7, 14, 28, 30, 60, 90], "VOL_TREND", {"feature_fraction": 0.7}),
    # ModelSpec("m4_interval_price_alt", [7, 14, 28, 30, 60, 90], "INT_PRICE", {"num_leaves": 15, "min_data_in_leaf": 400}),
]

MODEL_FEATURE_COLUMNS: Dict[str, List[str]] = {
    # Model 1: R-only (pure recency)
    "R_ONLY": [
        "days_since_last_purchase",
        "recency_bin",
        "last_purchase_interval",
    ],
    # Model 2: Volume + Trend (without recency)
    "VOL_TREND": [
        "txn_count_all",
        "spend_sum_all",
        "w7_txn_count",
        "w14_txn_count",
        "w28_txn_count",
        "w7_spend_sum",
        "w14_spend_sum",
        "w28_spend_sum",
        "spend_trend_30d",
        "txn_ratio_30d_prev30d",
        "activity_decay_30d_over_90d",
        "diff28_visit_days",
    ],
    # Model 3: Interval + UnitPrice (without recency)
    "INT_PRICE": [
        "mean_purchase_interval",
        "std_purchase_interval",
        "last_purchase_interval",
        "avg_unit_price_all",
        "avg_unit_price_7d",
        "unit_price_std",
    ],
}


# =========================
# utils
# =========================
def get_project_root() -> Path:
    return Path(__file__).resolve().parent

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

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
    data["date"] = pd.to_datetime(
        data["date"].astype("Int64").astype(str),
        format="%Y%m%d",
        errors="coerce",
    )
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
# features (windows可変)
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

    # 28がある時だけ追加特徴（視点差を出す）
    if 28 in windows:
        prev28 = _prev_window_agg(hist, 28, REF_DATE)
        feats = pd.merge(feats, prev28, on="user_id", how="left")
        feats["diff28_visit_days"] = feats["w28_visit_days"] - feats["prev28_visit_days"]
        feats["ratio28_visit_days"] = (feats["w28_visit_days"] + 1.0) / (feats["prev28_visit_days"] + 1.0)

    feats = pd.merge(feats, _recency_features(hist, REF_DATE), on="user_id", how="left")
    feats = pd.merge(feats, _interval_features(hist), on="user_id", how="left")

    # Explicit aliases for requested feature names
    feats["mean_purchase_interval"] = feats["avg_interval_days"]
    feats["std_purchase_interval"] = feats["std_interval_days"]
    feats["last_purchase_interval"] = feats["last_interval_days"]
    feats["avg_unit_price_all"] = feats["unit_price_mean_all"]
    feats["avg_unit_price_7d"] = feats["w7_unit_price_mean"] if "w7_unit_price_mean" in feats.columns else np.nan
    feats["unit_price_std"] = feats["unit_price_std_all"]

    # Recency bin (0..4): lower is more recent
    recency_bins = pd.cut(
        feats["days_since_last_purchase"],
        bins=[-np.inf, 7, 14, 30, 60, np.inf],
        labels=[0, 1, 2, 3, 4],
    )
    feats["recency_bin"] = recency_bins.astype(float)

    # 30-day trend features
    cur30_start = REF_DATE - pd.Timedelta(days=30)
    prev30_start = REF_DATE - pd.Timedelta(days=60)
    cur90_start = REF_DATE - pd.Timedelta(days=90)

    cur30 = hist[hist["date"] >= cur30_start]
    prev30 = hist[(hist["date"] >= prev30_start) & (hist["date"] < cur30_start)]
    cur90 = hist[hist["date"] >= cur90_start]

    cur30_agg = cur30.groupby("user_id").agg(
        cur_spend_30d=("total_price", "sum"),
        cur_txn_30d=("total_price", "size"),
    ).reset_index()
    prev30_agg = prev30.groupby("user_id").agg(
        spend_prev30d=("total_price", "sum"),
        txn_prev30d=("total_price", "size"),
    ).reset_index()
    cur90_agg = cur90.groupby("user_id").agg(cur_txn_90d=("total_price", "size")).reset_index()

    feats = pd.merge(feats, cur30_agg, on="user_id", how="left")
    feats = pd.merge(feats, prev30_agg, on="user_id", how="left")
    feats = pd.merge(feats, cur90_agg, on="user_id", how="left")

    for c in ["cur_spend_30d", "cur_txn_30d", "spend_prev30d", "txn_prev30d", "cur_txn_90d"]:
        if c in feats.columns:
            feats[c] = feats[c].fillna(0.0)

    feats["spend_trend_30d"] = feats["cur_spend_30d"] - feats["spend_prev30d"]
    feats["txn_trend_30d"] = feats["cur_txn_30d"] - feats["txn_prev30d"]
    feats["spend_ratio_30d_prev30d"] = (feats["cur_spend_30d"] + 1.0) / (feats["spend_prev30d"] + 1.0)
    feats["txn_ratio_30d_prev30d"] = (feats["cur_txn_30d"] + 1.0) / (feats["txn_prev30d"] + 1.0)
    feats["activity_decay_30d_over_90d"] = (feats["cur_txn_30d"] + 1.0) / (feats["cur_txn_90d"] + 1.0)

    # 窓系/prev系は0埋め
    for c in [col for col in feats.columns if col.startswith("w") or col.startswith("prev")]:
        feats[c] = feats[c].fillna(0)

    feats = feats.replace([np.inf, -np.inf], np.nan)
    return feats

def prepare_train_data(train_flag: pd.DataFrame, features: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    if TARGET_COL not in train_flag.columns:
        raise KeyError(f"train_flag に {TARGET_COL} 列が無い。TARGET_COL を確認して。 columns={list(train_flag.columns)}")

    df = pd.merge(train_flag[["user_id", TARGET_COL]], features, on="user_id", how="left", sort=False)
    df = df.set_index("user_id").loc[train_flag["user_id"]].reset_index()
    y = df[TARGET_COL].astype(int)
    X = df.drop(["user_id", TARGET_COL], axis=1)
    return X, y

def prepare_test_data(sample_submission: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    df = pd.merge(sample_submission[["user_id"]], features, on="user_id", how="left", sort=False)
    df = df.set_index("user_id").loc[sample_submission["user_id"]].reset_index()
    X = df.drop(["user_id"], axis=1)
    return X


def select_model_feature_columns(all_columns: List[str], feature_set: str) -> List[str]:
    if feature_set not in MODEL_FEATURE_COLUMNS:
        raise ValueError(f"Unknown feature_set={feature_set}. Expected one of {list(MODEL_FEATURE_COLUMNS.keys())}.")

    wanted = MODEL_FEATURE_COLUMNS[feature_set]
    use_cols = [c for c in wanted if c in all_columns]
    missing = [c for c in wanted if c not in all_columns]

    if missing:
        print(f"  Warning: {feature_set} missing columns ({len(missing)}): {missing}")
    if len(use_cols) == 0:
        raise ValueError(f"feature_set={feature_set} has zero usable columns.")

    return use_cols


# =========================
# CV with OOF + test pred
# =========================
def train_lgb_cv_with_oof(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    params: Dict[str, object],
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
            params=params,
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

    return oof, test_pred, float(auc_all)


# =========================
# Dynamic uncertainty-weighted ensemble
#   - fixed strength weights from OOF AUC (w0)
#   - per-row flattening by std(preds) (u)
#   - u big => alpha big => weights become uniform => "強いモデル比率が下がる"
# =========================
def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)

def base_weights_from_auc(aucs: List[float], temp: float = 0.02) -> np.ndarray:
    """
    OOF AUC -> 強さ重みw0（K,）
    temp小: 強いモデルに寄せる
    """
    a = np.array(aucs, dtype=float)
    z = (a - a.mean()) / max(float(temp), 1e-9)
    return softmax(z)

@dataclass(frozen=True)
class DynWeightParams:
    u_center_q: float   # u_center quantile
    k: float            # sigmoid sharpness
    alpha_max: float    # flatten strength max (0..1)
    temp: float         # temp for w0

def dynamic_weights(P: np.ndarray, w0: np.ndarray, u_center: float, k: float, alpha_max: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    P: (N,K)
    w0: (K,)
    returns: w_row (N,K), u (N,), alpha (N,)
    """
    N, K = P.shape
    u = P.std(axis=1)

    # u big => alpha close to alpha_max
    alpha = alpha_max * sigmoid(k * (u - u_center))
    alpha = np.clip(alpha, 0.0, alpha_max)

    uni = np.full(K, 1.0 / K, dtype=float)
    w_row = (1.0 - alpha)[:, None] * w0[None, :] + alpha[:, None] * uni[None, :]

    w_row = np.clip(w_row, 1e-12, None)
    w_row = w_row / w_row.sum(axis=1, keepdims=True)
    return w_row, u, alpha

def predict_dynamic_weighted(P: np.ndarray, w_row: np.ndarray) -> np.ndarray:
    return np.sum(P * w_row, axis=1)

def fit_dynamic_ensemble_oof(
    y: np.ndarray,
    oof_list: List[np.ndarray],
    aucs: List[float],
    grid: List[DynWeightParams],
) -> Tuple[DynWeightParams, float, Dict[str, float]]:
    P = np.stack(oof_list, axis=1)  # (N,K)
    u = P.std(axis=1)

    best_auc = -1.0
    best_params: Optional[DynWeightParams] = None
    best_info: Dict[str, float] = {}

    for g in grid:
        w0 = base_weights_from_auc(aucs, temp=g.temp)
        u_center = float(np.quantile(u, g.u_center_q))

        w_row, u2, alpha = dynamic_weights(P, w0, u_center=u_center, k=g.k, alpha_max=g.alpha_max)
        pred = predict_dynamic_weighted(P, w_row)
        auc = float(roc_auc_score(y, pred))

        if auc > best_auc:
            best_auc = auc
            best_params = g
            best_info = {
                "u_center": u_center,
                "mean_u": float(u2.mean()),
                "std_u": float(u2.std()),
                "mean_alpha": float(alpha.mean()),
                "w0_max": float(w0.max()),
                "w0_min": float(w0.min()),
            }

    assert best_params is not None
    return best_params, best_auc, best_info


# =========================
# main
# =========================
def main():
    root = get_project_root()
    out_root = ensure_dir(root / "output" / OUT_DIR_NAME)
    model_out = ensure_dir(out_root / "models")
    sub_dir = ensure_dir(root / "output" / "submissions")

    print(f"REF_DATE: {REF_DATE.date()}")
    print(f"TARGET_COL: {TARGET_COL}")
    print(f"Num models: {len(MODEL_SPECS)}")
    for i, ms in enumerate(MODEL_SPECS):
        print(
            f"  [{i}] {ms.tag} windows={ms.windows} "
            f"feature_set={ms.feature_set} overrides={ms.lgb_overrides}"
        )

    print("\n[1/4] Loading data...")
    data, train_flag, sample_submission = load_data()

    print("[2/4] Preprocessing...")
    data = clean_data(data)

    pred_col = "pred" if ("pred" in sample_submission.columns) else sample_submission.columns[-1]

    oof_list: List[np.ndarray] = []
    test_list: List[np.ndarray] = []
    aucs: List[float] = []

    print("\n[3/4] Train multiple models...")
    y_train: Optional[pd.Series] = None

    for i, spec in enumerate(MODEL_SPECS):
        print(f"\n--- Training {spec.tag} ({i+1}/{len(MODEL_SPECS)}) ---")
        feats = create_features(data, spec.windows)
        X_train, y = prepare_train_data(train_flag, feats)
        X_test = prepare_test_data(sample_submission, feats)
        use_cols = select_model_feature_columns(list(X_train.columns), spec.feature_set)
        print(f"  feature_set={spec.feature_set} num_features={len(use_cols)}")
        print(f"  columns={use_cols}")

        X_train = X_train[use_cols].copy()
        X_test = X_test[use_cols].copy()

        if y_train is None:
            y_train = y
        else:
            # 同一順序・同一yであることを前提（念のためチェック）
            if not np.array_equal(y_train.values, y.values):
                raise RuntimeError("y_train がモデル間で一致していない。train_flagの順序などを確認して。")

        params = dict(LGB_BASE_PARAMS)
        params.update(spec.lgb_overrides)

        oof, test_pred, auc = train_lgb_cv_with_oof(
            X_train=X_train,
            y_train=y,
            X_test=X_test,
            params=params,
            out_dir=model_out,
            tag=spec.tag,
        )
        oof_list.append(oof)
        test_list.append(test_pred)
        aucs.append(auc)

    assert y_train is not None

    print("\n[Model OOF AUC summary]")
    for i, (spec, auc) in enumerate(zip(MODEL_SPECS, aucs)):
        print(f"  [{i}] {spec.tag}: {auc:.6f}")

    # 参考：固定の強さ重み
    w0_ref = base_weights_from_auc(aucs, temp=0.02)
    print("\n[Reference strength weights w0 from OOF AUC (temp=0.02)]")
    for i, w in enumerate(w0_ref):
        print(f"  [{i}] {MODEL_SPECS[i].tag}: w0={w:.4f}")

    print("\n[4/4] Fit dynamic uncertainty-weighted ensemble on OOF...")

    y_np = y_train.values.astype(int)

    # 探索グリッド：多すぎると過学習するのでほどほど
    grid: List[DynWeightParams] = []
    for q in [0.60, 0.70, 0.80]:
        for k in [5.0, 10.0, 20.0, 40.0]:
            for alpha_max in [0.20, 0.35, 0.50, 0.70, 0.90]:
                for temp in [0.01, 0.02, 0.05]:
                    grid.append(DynWeightParams(u_center_q=q, k=k, alpha_max=alpha_max, temp=temp))

    best_params, best_auc, info = fit_dynamic_ensemble_oof(
        y=y_np,
        oof_list=oof_list,
        aucs=aucs,
        grid=grid,
    )

    # OOFで決めた u_center を固定して test に適用
    P_oof = np.stack(oof_list, axis=1)
    u_oof = P_oof.std(axis=1)
    u_center = float(np.quantile(u_oof, best_params.u_center_q))

    w0 = base_weights_from_auc(aucs, temp=best_params.temp)
    w_row_oof, u_oof2, alpha_oof = dynamic_weights(P_oof, w0, u_center=u_center, k=best_params.k, alpha_max=best_params.alpha_max)
    oof_blend = predict_dynamic_weighted(P_oof, w_row_oof)
    auc_final = float(roc_auc_score(y_np, oof_blend))

    print("\n[Best dynamic ensemble params]")
    print(f"  best_grid_oof_auc: {best_auc:.6f}")
    print(f"  final_oof_auc (recalc): {auc_final:.6f}")
    print(f"  params: {best_params}")
    print(f"  info: {info}")

    print("\n[Final w0 used]")
    for i, w in enumerate(w0):
        print(f"  [{i}] {MODEL_SPECS[i].tag}: w0={w:.4f}")

    # OOFログ保存
    oof_log = pd.DataFrame({
        "row_id": np.arange(len(oof_blend)),
        "y": y_np,
        "u_std": u_oof2,
        "alpha_flatten": alpha_oof,
        "oof_blend": oof_blend,
    })
    oof_log.to_csv(out_root / "oof_blend.csv", index=False)

    # test へ適用
    P_test = np.stack(test_list, axis=1)
    w_row_test, u_test, alpha_test = dynamic_weights(P_test, w0, u_center=u_center, k=best_params.k, alpha_max=best_params.alpha_max)
    test_blend = predict_dynamic_weighted(P_test, w_row_test)

    # 提出
    sub = sample_submission.copy()
    sub[pred_col] = test_blend
    sub_path = sub_dir / "sub_dynamic_uncertainty_weighted_v1.csv"
    sub.to_csv(sub_path, index=False)

    # 比較用：固定重み（w0）だけのアンサンブル
    test_fixed = np.sum(P_test * w0[None, :], axis=1)
    sub2 = sample_submission.copy()
    sub2[pred_col] = test_fixed
    sub2_path = sub_dir / "sub_fixed_strength_w0.csv"
    sub2.to_csv(sub2_path, index=False)

    # 比較用：単純平均
    test_mean = P_test.mean(axis=1)
    sub3 = sample_submission.copy()
    sub3[pred_col] = test_mean
    sub3_path = sub_dir / "sub_simple_mean_models.csv"
    sub3.to_csv(sub3_path, index=False)

    # testの不確実性ログ（どのくらい平坦化されたか）
    pd.DataFrame({
        "row_id": np.arange(len(test_blend)),
        "u_std": u_test,
        "alpha_flatten": alpha_test,
        "pred_blend": test_blend,
        "pred_fixed_w0": test_fixed,
        "pred_mean": test_mean,
    }).to_csv(out_root / "test_blend_debug.csv", index=False)

    print("\n[Outputs]")
    print(f"  submission (dynamic): {sub_path}")
    print(f"  submission (fixed w0): {sub2_path}")
    print(f"  submission (mean): {sub3_path}")
    print(f"  artifacts: {out_root}")
    print("Done.")

if __name__ == "__main__":
    main()
