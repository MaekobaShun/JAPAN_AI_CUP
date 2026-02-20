"""
学習・推論ロジック
"""
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from typing import Optional
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import yaml
import pickle

from .utils import get_output_path, ensure_dir


def train_lightgbm_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict,
    cv_config: dict,
    X_test: Optional[pd.DataFrame] = None,
    categorical_features: Optional[list[str]] = None
) -> tuple[list[lgb.Booster], np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """
    LightGBMでクロスバリデーションを実行
    
    Args:
        X_train: 学習データの特徴量
        y_train: 学習データのターゲット
        params: LightGBMのパラメータ
        cv_config: クロスバリデーションの設定
        X_test: テストデータの特徴量（オプション）
        categorical_features: カテゴリカル特徴量のリスト
    
    Returns:
        (models, oof_train, y_preds_cv, y_test_preds) のタプル
        - models: 各foldの学習済みモデル
        - oof_train: Out-of-fold予測値
        - y_preds_cv: 各foldのバリデーション予測値
        - y_test_preds: 各foldのテストデータ予測値（X_testが提供された場合）
    """
    if categorical_features is None:
        categorical_features = []
    
    y_preds_cv = []
    y_test_preds = []
    models = []
    oof_train = np.zeros(len(X_train))
    
    cv = StratifiedKFold(
        n_splits=cv_config["n_splits"],
        shuffle=cv_config["shuffle"],
        random_state=cv_config["random_state"]
    )
    
    for fold_id, (train_index, valid_index) in enumerate(cv.split(X_train, y_train)):
        # クロスバリデーション用分割
        X_tr = X_train.iloc[train_index, :]
        X_val = X_train.iloc[valid_index, :]
        y_tr = y_train.iloc[train_index]
        y_val = y_train.iloc[valid_index]
        
        # referenceを使うと、参照先のデータ構造をそのまま使って学習できる
        lgb_train = lgb.Dataset(X_tr, y_tr, categorical_feature=categorical_features)
        lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train, categorical_feature=categorical_features)
        
        # モデル学習
        model = lgb.train(
            params,
            lgb_train,
            valid_sets=[lgb_train, lgb_eval],
            num_boost_round=cv_config.get("num_boost_round", 10000),
            callbacks=[
                lgb.early_stopping(cv_config.get("early_stopping_rounds", 100)),
                lgb.log_evaluation(period=cv_config.get("log_evaluation_period", 100))
            ]
        )
        
        # oof_trainをバリデーションデータの予測で埋める
        oof_train[valid_index] = model.predict(X_val, num_iteration=model.best_iteration)
        models.append(model)
        
        # 各valごとの予測を保存
        y_pred = model.predict(X_val, num_iteration=model.best_iteration)
        y_preds_cv.append(y_pred)
        
        # テストデータの予測（X_testが提供された場合）
        if X_test is not None:
            y_test_pred = model.predict(X_test, num_iteration=model.best_iteration)
            y_test_preds.append(y_test_pred)
    
    return models, oof_train, y_preds_cv, y_test_preds


def predict_with_models(
    models: list[lgb.Booster],
    X_test: pd.DataFrame
) -> np.ndarray:
    """
    複数のモデルで予測を行い、平均を取る
    
    Args:
        models: 学習済みモデルのリスト
        X_test: テストデータの特徴量
    
    Returns:
        予測値の平均
    """
    y_test_preds = []
    for model in models:
        y_test_pred = model.predict(X_test, num_iteration=model.best_iteration)
        y_test_preds.append(y_test_pred)
    
    # 複数モデルの予測を平均
    y_sub = np.mean(y_test_preds, axis=0)
    return y_sub


def evaluate_cv(y_train: pd.Series, oof_train: np.ndarray) -> float:
    """
    クロスバリデーションスコアを計算
    
    Args:
        y_train: 学習データのターゲット
        oof_train: Out-of-fold予測値
    
    Returns:
        CV AUCスコア
    """
    return roc_auc_score(y_train.values, oof_train)


def save_models(models: list[lgb.Booster], output_dir: Optional[Path] = None) -> Path:
    """
    学習済みモデルを保存
    
    Args:
        models: 学習済みモデルのリスト
        output_dir: 出力ディレクトリ（Noneの場合はデフォルト）
    
    Returns:
        保存先ディレクトリのPath
    """
    if output_dir is None:
        output_dir = ensure_dir(get_output_path("models"))
    else:
        output_dir = ensure_dir(output_dir)
    
    for fold_id, model in enumerate(models):
        model_path = output_dir / f"model_fold_{fold_id}.txt"
        model.save_model(str(model_path))
    
    return output_dir


def load_models(model_dir: Path) -> list[lgb.Booster]:
    """
    保存されたモデルを読み込む
    
    Args:
        model_dir: モデルが保存されているディレクトリ
    
    Returns:
        読み込んだモデルのリスト
    """
    models = []
    model_files = sorted(model_dir.glob("model_fold_*.txt"))
    
    for model_file in model_files:
        model = lgb.Booster(model_file_path=str(model_file))
        models.append(model)
    
    return models


def calculate_and_visualize_shap(
    models: list[lgb.Booster],
    X_train: pd.DataFrame,
    feature_names: list[str],
    output_dir: Optional[Path] = None,
    sample_size: int = 1000
) -> pd.DataFrame:
    """
    SHAP値を計算し、可視化する
    
    Args:
        models: 学習済みモデルのリスト
        X_train: 学習データの特徴量
        feature_names: 特徴量名のリスト
        output_dir: 出力ディレクトリ（Noneの場合はデフォルト）
        sample_size: SHAP計算に使用するサンプル数
    
    Returns:
        SHAP値のDataFrame（特徴量ごとの平均絶対SHAP値）
    """
    try:
        import shap
    except ImportError:
        raise ImportError("shapライブラリがインストールされていません。pip install shapでインストールしてください。")
    
    if output_dir is None:
        output_dir = ensure_dir(get_output_path("shap"))
    else:
        output_dir = ensure_dir(output_dir)
    
    # サンプリング（計算時間を短縮するため）
    if len(X_train) > sample_size:
        sample_indices = np.random.choice(len(X_train), sample_size, replace=False)
        X_sample = X_train.iloc[sample_indices].copy()
    else:
        X_sample = X_train.copy()
    
    # 各foldのSHAP値を計算して平均
    all_shap_values = []
    
    for fold_id, model in enumerate(models):
        print(f"  Calculating SHAP values for fold {fold_id + 1}/{len(models)}...")
        
        # TreeExplainerを使用（LightGBM用）
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        
        # 二値分類の場合、shap_valuesはリストになることがある
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # クラス1のSHAP値を使用
        
        all_shap_values.append(shap_values)
    
    # 全foldの平均を計算
    mean_shap_values = np.mean(all_shap_values, axis=0)
    
    # 特徴量ごとの平均絶対SHAP値を計算
    mean_abs_shap = np.abs(mean_shap_values).mean(axis=0)
    shap_importance = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=False)
    
    # SHAP値をCSVで保存
    shap_df = pd.DataFrame(mean_shap_values, columns=feature_names, index=X_sample.index)
    shap_df.to_csv(output_dir / "shap_values.csv", index=False)
    shap_importance.to_csv(output_dir / "shap_importance.csv", index=False)
    
    # 可視化
    print("  Creating SHAP visualizations...")
    
    # 1. Summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        mean_shap_values,
        X_sample,
        feature_names=feature_names,
        show=False,
        max_display=min(20, len(feature_names))
    )
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary_plot.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    # 2. Bar plot（特徴量重要度）
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        mean_shap_values,
        X_sample,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
        max_display=min(20, len(feature_names))
    )
    plt.tight_layout()
    plt.savefig(output_dir / "shap_bar_plot.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    # 3. 特徴量重要度の棒グラフ（カスタム）
    plt.figure(figsize=(10, max(6, len(feature_names) * 0.3)))
    top_features = shap_importance.head(20)
    plt.barh(range(len(top_features)), top_features["mean_abs_shap"].values)
    plt.yticks(range(len(top_features)), top_features["feature"].values)
    plt.xlabel("Mean |SHAP value|")
    plt.title("Feature Importance (SHAP)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_dir / "shap_importance_bar.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"  SHAP values and visualizations saved to: {output_dir}")
    
    return shap_importance


def train_lightgbm_cv_with_optuna(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_config: dict,
    optuna_config: dict,
    categorical_features: Optional[list[str]] = None,
    fixed_folds: Optional[list] = None
) -> tuple[dict, list]:
    """
    Optunaを使ってLightGBMのハイパーパラメータを最適化
    
    Args:
        X_train: 学習データの特徴量
        y_train: 学習データのターゲット
        cv_config: クロスバリデーションの設定
        optuna_config: Optunaの設定（search_space, fixed_params, alpha, n_trials等）
        categorical_features: カテゴリカル特徴量のリスト
        fixed_folds: 固定されたfold分割（Noneの場合は新規作成）
    
    Returns:
        (best_params, fixed_folds) のタプル
        - best_params: 最適化されたパラメータ
        - fixed_folds: 固定されたfold分割（再利用用）
    """
    try:
        import optuna
    except ImportError:
        raise ImportError("optunaライブラリがインストールされていません。pip install optunaでインストールしてください。")
    
    if categorical_features is None:
        categorical_features = []
    
    # 固定split（fold）を作成（毎回同じsplitを使用）
    if fixed_folds is None:
        cv = StratifiedKFold(
            n_splits=cv_config["n_splits"],
            shuffle=cv_config["shuffle"],
            random_state=cv_config["random_state"]
        )
        fixed_folds = list(cv.split(X_train, y_train))
    
    search_space = optuna_config.get("search_space", {})
    fixed_params = optuna_config.get("fixed_params", {})
    alpha = optuna_config.get("alpha", 0.5)
    n_trials = optuna_config.get("n_trials", 50)
    
    def objective(trial):
        # パラメータの提案
        params = fixed_params.copy()
        
        if "num_leaves" in search_space:
            params["num_leaves"] = trial.suggest_int(
                "num_leaves",
                search_space["num_leaves"]["low"],
                search_space["num_leaves"]["high"]
            )
        
        if "min_data_in_leaf" in search_space:
            params["min_data_in_leaf"] = trial.suggest_int(
                "min_data_in_leaf",
                search_space["min_data_in_leaf"]["low"],
                search_space["min_data_in_leaf"]["high"]
            )
        
        if "feature_fraction" in search_space:
            params["feature_fraction"] = trial.suggest_float(
                "feature_fraction",
                search_space["feature_fraction"]["low"],
                search_space["feature_fraction"]["high"]
            )
        
        if "bagging_fraction" in search_space:
            params["bagging_fraction"] = trial.suggest_float(
                "bagging_fraction",
                search_space["bagging_fraction"]["low"],
                search_space["bagging_fraction"]["high"]
            )
        
        if "bagging_freq" in search_space:
            params["bagging_freq"] = trial.suggest_int(
                "bagging_freq",
                search_space["bagging_freq"]["low"],
                search_space["bagging_freq"]["high"]
            )
        
        if "lambda_l1" in search_space:
            params["lambda_l1"] = trial.suggest_float(
                "lambda_l1",
                search_space["lambda_l1"]["low"],
                search_space["lambda_l1"]["high"]
            )
        
        if "lambda_l2" in search_space:
            params["lambda_l2"] = trial.suggest_float(
                "lambda_l2",
                search_space["lambda_l2"]["low"],
                search_space["lambda_l2"]["high"]
            )
        
        # CVで評価
        fold_scores = []
        for fold_id, (train_index, valid_index) in enumerate(fixed_folds):
            X_tr = X_train.iloc[train_index, :]
            X_val = X_train.iloc[valid_index, :]
            y_tr = y_train.iloc[train_index]
            y_val = y_train.iloc[valid_index]
            
            lgb_train = lgb.Dataset(X_tr, y_tr, categorical_feature=categorical_features)
            lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train, categorical_feature=categorical_features)
            
            model = lgb.train(
                params,
                lgb_train,
                valid_sets=[lgb_eval],
                num_boost_round=cv_config.get("num_boost_round", 10000),
                callbacks=[
                    lgb.early_stopping(cv_config.get("early_stopping_rounds", 100), verbose=False),
                    lgb.log_evaluation(period=0)  # ログを出力しない
                ]
            )
            
            y_pred = model.predict(X_val, num_iteration=model.best_iteration)
            score = roc_auc_score(y_val.values, y_pred)
            fold_scores.append(score)
        
        # 目的関数：mean_auc - alpha * std_auc
        mean_auc = np.mean(fold_scores)
        std_auc = np.std(fold_scores)
        objective_score = mean_auc - alpha * std_auc
        
        return objective_score
    
    # Optuna studyを作成
    study_name = optuna_config.get("study_name", "lightgbm_optuna")
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name
    )
    
    print(f"Starting Optuna optimization ({n_trials} trials)...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # 最適パラメータを取得
    best_params = fixed_params.copy()
    best_params.update(study.best_params)
    
    print(f"\nBest objective score: {study.best_value:.6f}")
    print(f"Best params: {best_params}")
    
    return best_params, fixed_folds, study


def train_with_seed_ensemble(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    best_params: dict,
    cv_config: dict,
    ensemble_config: dict,
    fixed_folds: Optional[list] = None,
    categorical_features: Optional[list[str]] = None,
    save_models_flag: bool = True
) -> tuple[np.ndarray, dict, dict]:
    """
    seedアンサンブルで予測を安定化
    
    Args:
        X_train: 学習データの特徴量
        y_train: 学習データのターゲット
        X_test: テストデータの特徴量
        best_params: 最適化されたパラメータ
        cv_config: クロスバリデーションの設定
        ensemble_config: アンサンブルの設定（seeds等）
        fixed_folds: 固定されたfold分割（Noneの場合は新規作成）
        categorical_features: カテゴリカル特徴量のリスト
        save_models_flag: モデルを保存するかどうか
    
    Returns:
        (final_pred, seed_results, models_by_seed) のタプル
        - final_pred: 最終予測（fold×seed平均）
        - seed_results: 各seedの結果（デバッグ用）
        - models_by_seed: {seed: [model_fold_0, model_fold_1, ...]} の辞書
    """
    if categorical_features is None:
        categorical_features = []
    
    seeds = ensemble_config.get("seeds", [23, 42, 123, 456, 789])
    
    # 固定split（fold）を作成（毎回同じsplitを使用）
    if fixed_folds is None:
        cv = StratifiedKFold(
            n_splits=cv_config["n_splits"],
            shuffle=cv_config["shuffle"],
            random_state=cv_config["random_state"]
        )
        fixed_folds = list(cv.split(X_train, y_train))
    
    all_seed_test_preds = []
    seed_results = {}
    models_by_seed = {}
    
    for seed_idx, seed in enumerate(seeds):
        print(f"\nTraining with seed {seed} ({seed_idx + 1}/{len(seeds)})...")
        
        # seedを変えてfold学習
        seed_cv = StratifiedKFold(
            n_splits=cv_config["n_splits"],
            shuffle=cv_config["shuffle"],
            random_state=seed  # seedを変更
        )
        seed_folds = list(seed_cv.split(X_train, y_train))
        
        fold_test_preds = []
        fold_scores = []
        seed_models = []
        
        for fold_id, (train_index, valid_index) in enumerate(seed_folds):
            X_tr = X_train.iloc[train_index, :]
            X_val = X_train.iloc[valid_index, :]
            y_tr = y_train.iloc[train_index]
            y_val = y_train.iloc[valid_index]
            
            lgb_train = lgb.Dataset(X_tr, y_tr, categorical_feature=categorical_features)
            lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train, categorical_feature=categorical_features)
            
            model = lgb.train(
                best_params,
                lgb_train,
                valid_sets=[lgb_eval],
                num_boost_round=cv_config.get("num_boost_round", 10000),
                callbacks=[
                    lgb.early_stopping(cv_config.get("early_stopping_rounds", 100), verbose=False),
                    lgb.log_evaluation(period=0)
                ]
            )
            
            # バリデーションスコア
            y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)
            score = roc_auc_score(y_val.values, y_pred_val)
            fold_scores.append(score)
            
            # テストデータの予測
            y_pred_test = model.predict(X_test, num_iteration=model.best_iteration)
            fold_test_preds.append(y_pred_test)
            
            if save_models_flag:
                seed_models.append(model)
        
        # 各seedで：test_pred_seed = mean_over_folds(test_pred_fold)
        seed_test_pred = np.mean(fold_test_preds, axis=0)
        all_seed_test_preds.append(seed_test_pred)
        
        seed_results[seed] = {
            "mean_cv_score": np.mean(fold_scores),
            "std_cv_score": np.std(fold_scores),
            "fold_scores": fold_scores
        }
        
        if save_models_flag:
            models_by_seed[seed] = seed_models
        
        print(f"  Seed {seed} - CV Score: {np.mean(fold_scores):.6f} ± {np.std(fold_scores):.6f}")
    
    # 最終予測：final_pred = mean_over_seeds(test_pred_seed)
    final_pred = np.mean(all_seed_test_preds, axis=0)
    
    print(f"\nEnsemble completed. Final prediction shape: {final_pred.shape}")
    
    return final_pred, seed_results, models_by_seed


def save_optuna_results(
    best_params: dict,
    study,
    output_dir: Optional[Path] = None
) -> Path:
    """
    Optunaの結果を保存
    
    Args:
        best_params: 最適化されたパラメータ
        study: Optunaのstudyオブジェクト
        output_dir: 出力ディレクトリ（Noneの場合はデフォルト）
    
    Returns:
        保存先ディレクトリのPath
    """
    if output_dir is None:
        output_dir = ensure_dir(get_output_path("optuna"))
    else:
        output_dir = ensure_dir(output_dir)
    
    # best_paramsをYAMLで保存
    best_params_path = output_dir / "best_params.yaml"
    with open(best_params_path, "w", encoding="utf-8") as f:
        yaml.dump(best_params, f, default_flow_style=False, allow_unicode=True)
    
    # studyオブジェクトをpickleで保存
    study_path = output_dir / "study.pkl"
    with open(study_path, "wb") as f:
        pickle.dump(study, f)
    
    print(f"Optuna results saved to: {output_dir}")
    
    return output_dir


def save_seed_ensemble_models(
    models_by_seed: dict,
    output_dir: Optional[Path] = None
) -> Path:
    """
    seedアンサンブル用のモデルを保存
    
    Args:
        models_by_seed: {seed: [model_fold_0, model_fold_1, ...]} の辞書
        output_dir: 出力ディレクトリ（Noneの場合はデフォルト）
    
    Returns:
        保存先ディレクトリのPath
    """
    if output_dir is None:
        output_dir = ensure_dir(get_output_path("models"))
    else:
        output_dir = ensure_dir(output_dir)
    
    for seed, models in models_by_seed.items():
        seed_dir = ensure_dir(output_dir / f"seed_{seed}")
        for fold_id, model in enumerate(models):
            model_path = seed_dir / f"model_fold_{fold_id}.txt"
            model.save_model(str(model_path))
    
    print(f"Seed ensemble models saved to: {output_dir}")
    
    return output_dir
