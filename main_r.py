"""
Recency特徴量のみでモデルを学習するアブレーションテスト

テスト内容:
  ① days_since_last_purchase を消す（ratioが上位なので単体recencyは冗長の可能性）
  ② period_30_60_days_visit_count を消す（recent_30が強いなら過去窓は弱い可能性）
  ③ 金額系を一旦全部消す（SHAPで弱い → Rだけで0.90維持できるか？）
"""
import yaml
from pathlib import Path
import pandas as pd
import numpy as np

from src.preprocessing import load_data, clean_data
from src.features import create_features, prepare_train_data, prepare_test_data, filter_features_r_only
from src.models import train_lightgbm_cv, evaluate_cv, predict_with_models
from src.utils import get_project_root, get_output_path, ensure_dir


def load_config(config_path: Path) -> dict:
    """設定ファイルを読み込む"""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def run_ablation(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    params: dict,
    cv_config: dict,
    experiment_name: str,
    remove_days_since_last_purchase: bool = False,
    remove_period_30_60_visit_count: bool = False,
    remove_all_price_features: bool = False,
) -> tuple[float, np.ndarray]:
    """
    特徴量を絞ったアブレーションテストを実行

    Args:
        X_train: 学習データの特徴量
        y_train: 学習データのターゲット
        X_test: テストデータの特徴量
        params: LightGBMパラメータ
        cv_config: CV設定
        experiment_name: 実験名
        remove_days_since_last_purchase: ① days_since_last_purchase を削除
        remove_period_30_60_visit_count: ② period_30_60_days_visit_count を削除
        remove_all_price_features: ③ 金額系を全て削除

    Returns:
        (cv_score, y_sub) のタプル
    """
    # 特徴量フィルタリング
    X_train_f = filter_features_r_only(
        X_train,
        remove_days_since_last_purchase=remove_days_since_last_purchase,
        remove_period_30_60_visit_count=remove_period_30_60_visit_count,
        remove_all_price_features=remove_all_price_features,
    )
    X_test_f = filter_features_r_only(
        X_test,
        remove_days_since_last_purchase=remove_days_since_last_purchase,
        remove_period_30_60_visit_count=remove_period_30_60_visit_count,
        remove_all_price_features=remove_all_price_features,
    )

    print(f"\n{'='*60}")
    print(f"  {experiment_name}")
    print(f"{'='*60}")
    print(f"  Features ({X_train_f.shape[1]}): {list(X_train_f.columns)}")

    models, oof_train, y_preds_cv, y_test_preds = train_lightgbm_cv(
        X_train=X_train_f,
        y_train=y_train,
        params=params,
        cv_config=cv_config,
        X_test=X_test_f,
        categorical_features=[],
    )

    cv_score = evaluate_cv(y_train, oof_train)
    print(f"  => CV AUC: {cv_score:.6f}")

    # テスト予測の平均
    y_sub = sum(y_test_preds) / len(y_test_preds)
    return cv_score, y_sub


def main():
    """アブレーションテスト メイン処理"""
    project_root = get_project_root()
    config_path = project_root / "conf" / "config.yaml"
    config = load_config(config_path)

    model_name = config.get("defaults", [{}])[0].get("model", "lightgbm")
    model_config_path = project_root / "conf" / "model" / f"{model_name.capitalize()}.yaml"
    model_config = load_config(model_config_path)

    exp_name = config.get("exp_name", "default")

    # === データ準備 ===
    print("[1/3] Loading & preprocessing data...")
    data, train_flag, sample_submission = load_data("raw")
    data = clean_data(data)

    print("[2/3] Creating features...")
    features = create_features(data)
    X_train, y_train = prepare_train_data(train_flag, features)
    X_test = prepare_test_data(sample_submission, features)

    print(f"Full features ({X_train.shape[1]}): {list(X_train.columns)}")

    params = model_config["params"]
    cv_config = {
        "n_splits": model_config["cv"]["n_splits"],
        "shuffle": model_config["cv"]["shuffle"],
        "random_state": model_config["cv"]["random_state"],
        "num_boost_round": model_config["train"]["num_boost_round"],
        "early_stopping_rounds": model_config["train"]["early_stopping_rounds"],
        "log_evaluation_period": model_config["train"]["log_evaluation_period"],
    }

    # === アブレーションテスト ===
    print("\n[3/3] Running ablation tests...")
    results = {}

    # ベースライン（全特徴量）
    score_base, y_sub_base = run_ablation(
        X_train, y_train, X_test, params, cv_config,
        experiment_name="Baseline (all features)",
    )
    results["baseline"] = score_base

    # ① days_since_last_purchase を消す
    score_1, _ = run_ablation(
        X_train, y_train, X_test, params, cv_config,
        experiment_name="① Remove days_since_last_purchase",
        remove_days_since_last_purchase=True,
    )
    results["remove_days_since"] = score_1

    # ② period_30_60_days_visit_count を消す
    score_2, _ = run_ablation(
        X_train, y_train, X_test, params, cv_config,
        experiment_name="② Remove period_30_60_days_visit_count",
        remove_period_30_60_visit_count=True,
    )
    results["remove_period_visit"] = score_2

    # ③ 金額系を全部消す（Recencyだけ）
    score_3, y_sub_r_only = run_ablation(
        X_train, y_train, X_test, params, cv_config,
        experiment_name="③ Remove all price features (R only)",
        remove_all_price_features=True,
    )
    results["remove_all_price"] = score_3

    # ①+②+③ 全部消す（最小構成: Recencyのみ）
    score_all, y_sub_minimal = run_ablation(
        X_train, y_train, X_test, params, cv_config,
        experiment_name="①+②+③ Minimal (R only, no redundancy)",
        remove_days_since_last_purchase=True,
        remove_period_30_60_visit_count=True,
        remove_all_price_features=True,
    )
    results["minimal_r_only"] = score_all

    # === 結果サマリ ===
    print(f"\n{'='*60}")
    print("  ABLATION TEST RESULTS")
    print(f"{'='*60}")
    for name, score in results.items():
        diff = score - score_base
        marker = "★" if abs(diff) < 0.001 else ("↑" if diff > 0 else "↓")
        print(f"  {marker} {name:30s}  AUC={score:.6f}  (diff={diff:+.6f})")

    # === 提出ファイル保存 ===
    submission_output_dir = ensure_dir(get_output_path("submissions"))

    # ①+②+③ の最小構成で提出ファイルを作成
    submission = sample_submission.copy()
    submission["pred"] = y_sub_minimal
    sub_path = submission_output_dir / f"sub_{exp_name}_r_only.csv"
    submission.to_csv(sub_path, index=False)
    print(f"\nSubmission (minimal R only): {sub_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
