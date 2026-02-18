import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from pathlib import Path

# プロジェクトルートのパスを取得
project_root = Path(__file__).parent.parent

# データの読み込み
data = pd.read_csv(project_root / "inputs" / "data.csv")
train_flag = pd.read_csv(project_root / "inputs" / "train_flag.csv")
init_sub = pd.read_csv(project_root / "inputs" / "sample_submission.csv")

# dateカラムを日付型に変換（YYYYMMDD形式の整数を日付に変換）
data["date"] = pd.to_datetime(data["date"], format="%Y%m%d")

# 各ユーザーの最後の来店日を取得
last_visit = data.groupby("user_id")["date"].max().reset_index()
last_visit.columns = ["user_id", "last_visit_date"]

# データ全体の最新日付を取得（基準日として使用）
latest_date = data["date"].max()

# 最後の来店日からの経過日数を計算
last_visit["days_since_last_visit"] = (latest_date - last_visit["last_visit_date"]).dt.days

# 既存の特徴量を作成
features = pd.DataFrame(data.groupby("user_id").agg({
    "date": ["count"],
    "average_unit_price": ["sum"]
}).reset_index().to_numpy())

features.columns = ["user_id", "date_count", "average_unit_price_sum"]

features["average_unit_price_sum"] = pd.to_numeric(features["average_unit_price_sum"], errors="coerce")
features["date_count"] = pd.to_numeric(features["date_count"], errors="coerce")

# 最後の来店日からの経過日数をマージ
features = pd.merge(features, last_visit[["user_id", "days_since_last_visit"]], on="user_id", how="left")

# 学習データの準備
X_train = pd.merge(train_flag, features, on="user_id", how="left")
X_train = X_train.drop(["user_id", "churn"], axis=1)
y_train = train_flag["churn"].to_numpy()

# テストデータの特徴量を作成
X_test = pd.merge(init_sub[["user_id"]], features, on="user_id", how="left")
X_test = X_test.drop(["user_id"], axis=1)  # 予測に不要なID列を削除

# クロスバリデーションの設定
y_preds_cv = []
y_test_preds = []
models = []
oof_train = np.zeros(len(X_train))
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)

categorical_features = []

params = {
    "objective": "binary",
    "max_bin": 300,
    "learning_rate": 0.1,
    "num_leaves": 40,
    "metric": "auc",
    "verbose": -1
}

# 学習するときは毎回違うバリデーションデータ
for fold_id, (train_index, valid_index) in enumerate(cv.split(X_train, y_train)):
    # クロスバリデーション用分割
    X_tr = X_train.iloc[train_index, :]
    X_val = X_train.iloc[valid_index, :]
    y_tr = y_train[train_index]
    y_val = y_train[valid_index]

    # referenceを使うと、参照先のデータ構造をそのまま使って学習できる
    lgb_train = lgb.Dataset(X_tr, y_tr, categorical_feature=categorical_features)
    lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train, categorical_feature=categorical_features)

    # モデル
    model = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_train, lgb_eval],
        num_boost_round=10000,
        callbacks=[
            lgb.early_stopping(100),
            lgb.log_evaluation(period=100)
        ]
    )

    # oof_trainをバリデーションデータの予測でどんどん埋めていく
    oof_train[valid_index] = model.predict(X_val, num_iteration=model.best_iteration)
    models.append(model)

    # 各valごとに振り替えれるように保存し解く（使ってない）
    y_pred = model.predict(X_val, num_iteration=model.best_iteration)
    y_preds_cv.append(y_pred)
    
    # 5分割毎回分のテストデータの予測をアンサンブル用に追加
    y_test_pred = model.predict(X_test, num_iteration=model.best_iteration)
    y_test_preds.append(y_test_pred)

# CVスコアの表示
cv_score = roc_auc_score(y_train, oof_train)
print(f"CV AUC Score: {cv_score}")

# 提出ファイルの作成
y_sub = sum(y_test_preds) / len(y_test_preds)
init_sub["pred"] = y_sub
output_path = project_root / "output" / "sub_2.csv"
init_sub.to_csv(output_path, index=False)

print(f"Submission file created: {output_path}")
print(f"Features used: {list(X_train.columns)}")
