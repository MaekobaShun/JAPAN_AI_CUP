"""
Optunaチューニングとseedアンサンブルを使ったエントリーポイント（6特徴量版）
"""
import yaml
from pathlib import Path
import pandas as pd
import pickle

from src.preprocessing import load_data, clean_data
from src.features import create_features_rf, prepare_train_data, prepare_test_data
from src.models import (
    train_lightgbm_cv_with_optuna,
    train_with_seed_ensemble,
    save_optuna_results,
    save_seed_ensemble_models
)
from src.utils import get_project_root, get_output_path, ensure_dir


def load_config(config_path: Path) -> dict:
    """設定ファイルを読み込む"""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def main():
    """メイン処理（Optuna + seedアンサンブル、RF特徴量版）"""
    project_root = get_project_root()
    config_path = project_root / "conf" / "config.yaml"
    config = load_config(config_path)
    
    # モデル設定を読み込む
    model_name = config.get("defaults", [{}])[0].get("model", "lightgbm")
    model_config_path = project_root / "conf" / "model" / f"{model_name.capitalize()}.yaml"
    model_config = load_config(model_config_path)
    
    exp_name = config.get("exp_name", "default")
    print(f"Experiment: {exp_name} (Optuna + Seed Ensemble, RF enhanced)")
    print(f"Model: {model_name}")
    
    # データ読み込み
    print("\n[1/6] Loading data...")
    data, train_flag, sample_submission = load_data("raw")
    
    # データ前処理
    print("[2/6] Preprocessing data...")
    data = clean_data(data)
    
    # 特徴量生成（RFMのF特徴量強化版）
    print("[3/6] Creating features (RF enhanced)...")
    features = create_features_rf(data)
    
    # 学習・テストデータの準備
    X_train, y_train = prepare_train_data(train_flag, features)
    X_test = prepare_test_data(sample_submission, features)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Features: {list(X_train.columns)}")
    
    # CV設定
    cv_config = {
        "n_splits": model_config["cv"]["n_splits"],
        "shuffle": model_config["cv"]["shuffle"],
        "random_state": model_config["cv"]["random_state"],
        "num_boost_round": model_config["train"]["num_boost_round"],
        "early_stopping_rounds": model_config["train"]["early_stopping_rounds"],
        "log_evaluation_period": model_config["train"]["log_evaluation_period"]
    }
    
    # Optuna設定
    optuna_config = model_config.get("optuna", {})
    ensemble_config = model_config.get("ensemble", {})
    
    # Optunaで最適パラメータを探索
    print("\n[4/6] Optimizing hyperparameters with Optuna...")
    best_params, fixed_folds, study = train_lightgbm_cv_with_optuna(
        X_train=X_train,
        y_train=y_train,
        cv_config=cv_config,
        optuna_config=optuna_config,
        categorical_features=[],
        fixed_folds=None  # 最初は新規作成
    )
    
    # Optunaの結果を保存
    optuna_output_dir = ensure_dir(get_output_path("optuna"))
    # RF版用に別名で保存
    best_params_path = optuna_output_dir / "best_params_rf.yaml"
    with open(best_params_path, "w", encoding="utf-8") as f:
        yaml.dump(best_params, f, default_flow_style=False, allow_unicode=True)
    
    study_path = optuna_output_dir / "study_rf.pkl"
    with open(study_path, "wb") as f:
        pickle.dump(study, f)
    
    print(f"Optuna results saved to: {optuna_output_dir}")
    
    # Seedアンサンブルで予測を安定化
    print("\n[5/6] Training with seed ensemble...")
    final_pred, seed_results, models_by_seed = train_with_seed_ensemble(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        best_params=best_params,
        cv_config=cv_config,
        ensemble_config=ensemble_config,
        fixed_folds=None,  # seedごとに異なるfoldを使用
        categorical_features=[],
        save_models_flag=True
    )
    
    # 各seedの結果を表示
    print("\nSeed ensemble results:")
    for seed, results in seed_results.items():
        print(f"  Seed {seed}: CV Score = {results['mean_cv_score']:.6f} ± {results['std_cv_score']:.6f}")
    
    # モデルの保存
    model_output_dir = ensure_dir(get_output_path("models"))
    save_seed_ensemble_models(models_by_seed, model_output_dir)
    print(f"Models saved to: {model_output_dir}")
    
    # 提出ファイルの作成
    print("\n[6/6] Creating submission file...")
    submission = sample_submission.copy()
    submission["pred"] = final_pred
    
    submission_output_dir = ensure_dir(get_output_path("submissions"))
    submission_path = submission_output_dir / f"sub_{exp_name}_optuna_rf.csv"
    submission.to_csv(submission_path, index=False)
    
    print(f"Submission file created: {submission_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
