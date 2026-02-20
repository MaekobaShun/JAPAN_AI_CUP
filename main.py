"""
エントリーポイント
"""
import yaml
from pathlib import Path
import pandas as pd

from src.preprocessing import load_data, clean_data
from src.features import create_features, prepare_train_data, prepare_test_data
from src.models import train_lightgbm_cv, evaluate_cv, predict_with_models, save_models, calculate_and_visualize_shap
from src.utils import get_project_root, get_output_path, ensure_dir


def load_config(config_path: Path) -> dict:
    """設定ファイルを読み込む"""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def main():
    """メイン処理"""
    project_root = get_project_root()
    config_path = project_root / "conf" / "config.yaml"
    config = load_config(config_path)
    
    # モデル設定を読み込む
    model_name = config.get("defaults", [{}])[0].get("model", "lightgbm")
    model_config_path = project_root / "conf" / "model" / f"{model_name.capitalize()}.yaml"
    model_config = load_config(model_config_path)
    
    print(f"Experiment: {config.get('exp_name', 'default')}")
    print(f"Model: {model_name}")
    
    # データ読み込み
    print("\n[1/5] Loading data...")
    data, train_flag, sample_submission = load_data("raw")
    
    # データ前処理
    print("[2/5] Preprocessing data...")
    data = clean_data(data)
    
    # 特徴量生成
    print("[3/5] Creating features...")
    features = create_features(data)
    
    # 学習・テストデータの準備
    X_train, y_train = prepare_train_data(train_flag, features)
    X_test = prepare_test_data(sample_submission, features)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Features: {list(X_train.columns)}")
    
    # モデル学習
    print("\n[4/5] Training model...")
    params = model_config["params"]
    cv_config = {
        "n_splits": model_config["cv"]["n_splits"],
        "shuffle": model_config["cv"]["shuffle"],
        "random_state": model_config["cv"]["random_state"],
        "num_boost_round": model_config["train"]["num_boost_round"],
        "early_stopping_rounds": model_config["train"]["early_stopping_rounds"],
        "log_evaluation_period": model_config["train"]["log_evaluation_period"]
    }
    
    models, oof_train, y_preds_cv, y_test_preds = train_lightgbm_cv(
        X_train=X_train,
        y_train=y_train,
        params=params,
        cv_config=cv_config,
        X_test=X_test,
        categorical_features=[]
    )
    
    # CVスコアの評価
    cv_score = evaluate_cv(y_train, oof_train)
    print(f"\nCV AUC Score: {cv_score:.6f}")
    
    # モデルの保存
    model_output_dir = ensure_dir(get_output_path("models"))
    save_models(models, model_output_dir)
    print(f"Models saved to: {model_output_dir}")
    
    # SHAP値の計算と可視化
    print("\n[6/6] Calculating SHAP values...")
    try:
        shap_importance = calculate_and_visualize_shap(
            models=models,
            X_train=X_train,
            feature_names=list(X_train.columns),
            output_dir=ensure_dir(get_output_path("shap")),
            sample_size=1000
        )
        print("\nTop 10 features by SHAP importance:")
        print(shap_importance.head(10).to_string(index=False))
    except Exception as e:
        print(f"Warning: SHAP calculation failed: {e}")
        print("Continuing without SHAP values...")
    
    # 提出ファイルの作成
    print("\n[5/6] Creating submission file...")
    if y_test_preds:
        # CV中に予測済みの場合
        y_sub = sum(y_test_preds) / len(y_test_preds)
    else:
        # 後から予測する場合
        y_sub = predict_with_models(models, X_test)
    
    submission = sample_submission.copy()
    submission["pred"] = y_sub
    
    submission_output_dir = ensure_dir(get_output_path("submissions"))
    submission_path = submission_output_dir / f"sub_{config.get('exp_name', 'default')}.csv"
    submission.to_csv(submission_path, index=False)
    
    print(f"Submission file created: {submission_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
