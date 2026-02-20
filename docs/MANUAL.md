# JAPAN AI CUP プロジェクト マニュアル

このマニュアルは、プロジェクトの構造と各モジュールの使い方を説明します。コード作成時や実行時に参照してください。

## 📁 プロジェクト構成

```
JAPAN_AI_CUP/
├── data/                    # データディレクトリ
│   ├── raw/                 # オリジナルデータ（変更しない）
│   │   ├── data.csv
│   │   ├── train_flag.csv
│   │   └── sample_submission.csv
│   ├── interim/             # 中間データ（必要に応じて使用）
│   └── processed/           # 最終的な特徴量データ（必要に応じて使用）
│
├── src/                     # ソースコード（メインの実装）
│   ├── __init__.py          # パッケージ初期化
│   ├── utils.py             # 共通ユーティリティ（パス管理など）
│   ├── preprocessing.py     # データ読み込み・前処理
│   ├── features.py          # 特徴量生成ロジック
│   └── models.py            # モデル学習・推論ロジック
│
├── conf/                    # 設定ファイル
│   ├── config.yaml          # メイン設定（実験名など）
│   └── model/               # モデル設定
│       └── LightGBM.yaml    # LightGBMのハイパーパラメータ
│
├── outputs/                 # 出力ディレクトリ
│   ├── models/              # 学習済みモデル保存先
│   └── submissions/         # 提出ファイル保存先
│
├── notebooks/               # Jupyter Notebook（探索的分析用）
│   └── init.ipynb
│
├── tests/                   # テストコード（将来の拡張用）
├── docs/                    # ドキュメント
├── main.py                  # エントリーポイント（実行用）
└── README.md
```

---

## 🚀 基本的な実行方法

### 1. 学習から提出ファイル作成まで一括実行

```bash
python main.py
```

このコマンドで以下が自動実行されます：
1. データ読み込み
2. データ前処理
3. 特徴量生成
4. モデル学習（クロスバリデーション）
5. 提出ファイル作成

### 2. 実行結果の確認

- **モデル**: `outputs/models/` に保存
- **提出ファイル**: `outputs/submissions/sub_{実験名}.csv` に保存
- **CVスコア**: コンソールに表示

---

## 📝 各モジュールの詳細

### `src/utils.py` - 共通ユーティリティ

**役割**: パス管理などの共通関数

**主な関数**:

```python
from src.utils import get_project_root, get_data_path, get_output_path, ensure_dir

# プロジェクトルートのパス取得
project_root = get_project_root()

# データディレクトリのパス取得
data_path = get_data_path("raw")      # data/raw/
data_path = get_data_path("interim")  # data/interim/
data_path = get_data_path("processed") # data/processed/

# 出力ディレクトリのパス取得
output_path = get_output_path("models")      # outputs/models/
output_path = get_output_path("submissions") # outputs/submissions/

# ディレクトリが存在しない場合は作成
ensure_dir(output_path)
```

**使用例**:
- 新しいデータファイルを読み込む際
- 出力先を指定する際
- パスを動的に生成する際

---

### `src/preprocessing.py` - データ前処理

**役割**: データの読み込みとクレンジング

**主な関数**:

#### `load_data(data_type: str = "raw")`

データを読み込む関数。

```python
from src.preprocessing import load_data

# 生データを読み込む
data, train_flag, sample_submission = load_data("raw")

# 中間データを読み込む（特徴量を保存した場合など）
data, train_flag, sample_submission = load_data("interim")
```

**戻り値**:
- `data`: メインデータ（pd.DataFrame）
- `train_flag`: 学習フラグデータ（user_id, churnを含む）
- `sample_submission`: サブミッションテンプレート

**使用例**:
- データを最初に読み込む際
- 保存した中間データを読み込む際

#### `clean_data(data: pd.DataFrame)`

データをクレンジングする関数。

```python
from src.preprocessing import clean_data

# データをクレンジング（日付型への変換など）
data_cleaned = clean_data(data)
```

**処理内容**:
- `date`カラムを日付型（datetime）に変換
- その他の前処理（必要に応じて追加）

**使用例**:
- データ読み込み後、特徴量生成前に実行
- 日付型への変換が必要な場合

---

### `src/features.py` - 特徴量生成

**役割**: 機械学習用の特徴量を生成

**主な関数**:

#### `create_features(data: pd.DataFrame)`

生データから特徴量を生成する関数。

```python
from src.features import create_features

# 特徴量を生成
features = create_features(data_cleaned)
```

**生成される特徴量**:
- `date_count`: 各ユーザーの来店回数
- `average_unit_price_sum`: 各ユーザーの平均単価の合計
- `days_since_last_visit`: 最後の来店日からの経過日数

**戻り値**: `pd.DataFrame`（`user_id`を含む）

**使用例**:
- 新しい特徴量を追加したい場合、この関数を編集
- 特徴量エンジニアリングのメインロジック

#### `prepare_train_data(train_flag, features)`

学習データを準備する関数。

```python
from src.features import prepare_train_data

# 学習データの準備
X_train, y_train = prepare_train_data(train_flag, features)
```

**戻り値**:
- `X_train`: 特徴量データ（user_id, churnを除く）
- `y_train`: ターゲット（churn）

**使用例**:
- モデル学習前に実行
- 特徴量とターゲットを分離する際

#### `prepare_test_data(sample_submission, features)`

テストデータを準備する関数。

```python
from src.features import prepare_test_data

# テストデータの準備
X_test = prepare_test_data(sample_submission, features)
```

**戻り値**: `X_test`（user_idを除く特徴量データ）

**使用例**:
- 予測前に実行
- テストデータの特徴量を準備する際

---

### `src/models.py` - モデル学習・推論

**役割**: 機械学習モデルの学習、評価、予測

**主な関数**:

#### `train_lightgbm_cv(X_train, y_train, params, cv_config, X_test=None, categorical_features=None)`

LightGBMでクロスバリデーションを実行する関数。

```python
from src.models import train_lightgbm_cv

# クロスバリデーション実行
models, oof_train, y_preds_cv, y_test_preds = train_lightgbm_cv(
    X_train=X_train,
    y_train=y_train,
    params=params,              # LightGBMのパラメータ（dict）
    cv_config=cv_config,       # CV設定（dict）
    X_test=X_test,              # オプション：テストデータ
    categorical_features=[]     # オプション：カテゴリカル特徴量
)
```

**パラメータ**:
- `params`: LightGBMのパラメータ（例：`{"objective": "binary", "learning_rate": 0.1}`）
- `cv_config`: CV設定（例：`{"n_splits": 5, "shuffle": True, "random_state": 23}`）

**戻り値**:
- `models`: 各foldの学習済みモデル（list）
- `oof_train`: Out-of-fold予測値（numpy配列）
- `y_preds_cv`: 各foldのバリデーション予測値（list）
- `y_test_preds`: 各foldのテストデータ予測値（X_testが提供された場合）

**使用例**:
- モデル学習のメイン処理
- クロスバリデーションでモデルを評価する際

#### `evaluate_cv(y_train, oof_train)`

クロスバリデーションスコアを計算する関数。

```python
from src.models import evaluate_cv

# CVスコアを計算
cv_score = evaluate_cv(y_train, oof_train)
print(f"CV AUC Score: {cv_score:.6f}")
```

**戻り値**: CV AUCスコア（float）

**使用例**:
- モデル性能を評価する際
- 実験結果を記録する際

#### `predict_with_models(models, X_test)`

複数のモデルで予測を行い、平均を取る関数。

```python
from src.models import predict_with_models

# 複数モデルで予測（アンサンブル）
y_sub = predict_with_models(models, X_test)
```

**戻り値**: 予測値の平均（numpy配列）

**使用例**:
- CV中に予測しなかった場合の予測
- 保存したモデルで予測する際

#### `save_models(models, output_dir=None)`

学習済みモデルを保存する関数。

```python
from src.models import save_models
from src.utils import get_output_path, ensure_dir

# モデルを保存
output_dir = ensure_dir(get_output_path("models"))
save_models(models, output_dir)
```

**使用例**:
- 学習済みモデルを保存する際
- 後で予測に使用する場合

#### `load_models(model_dir)`

保存されたモデルを読み込む関数。

```python
from src.models import load_models
from pathlib import Path

# モデルを読み込み
model_dir = Path("outputs/models")
models = load_models(model_dir)
```

**使用例**:
- 保存したモデルで予測する際
- モデルの再評価を行う際

---

## ⚙️ 設定ファイル

### `conf/config.yaml`

メイン設定ファイル。

```yaml
defaults:
  - model: lightgbm  # 使用するモデル（conf/model/内のファイル名）

exp_name: "exp001_first_try"  # 実験名（提出ファイル名に使用）
```

**変更方法**:
- `exp_name`を変更すると、提出ファイル名が変わる
- `model`を変更すると、別のモデル設定を使用できる

### `conf/model/LightGBM.yaml`

LightGBMのハイパーパラメータ設定。

```yaml
name: LightGBM

params:                    # LightGBMのパラメータ
  objective: binary
  max_bin: 300
  learning_rate: 0.1
  num_leaves: 40
  metric: auc
  verbose: -1

train:                     # 学習設定
  num_boost_round: 10000
  early_stopping_rounds: 100
  log_evaluation_period: 100

cv:                        # クロスバリデーション設定
  n_splits: 5
  shuffle: true
  random_state: 23
```

**変更方法**:
- `params`セクション: LightGBMのパラメータを変更
- `train`セクション: 学習回数やearly stoppingを調整
- `cv`セクション: CVの分割数やrandom_stateを変更

---

## 🔄 よくある作業フロー

### 1. 新しい特徴量を追加する

1. `src/features.py`の`create_features()`関数を編集
2. 新しい特徴量の計算ロジックを追加
3. `python main.py`で実行して確認

**例**:
```python
def create_features(data: pd.DataFrame) -> pd.DataFrame:
    # 既存の特徴量生成...
    
    # 新しい特徴量を追加
    new_feature = data.groupby("user_id")["some_column"].mean()
    features = pd.merge(features, new_feature, on="user_id", how="left")
    
    return features
```

### 2. ハイパーパラメータを調整する

1. `conf/model/LightGBM.yaml`を編集
2. パラメータを変更（例：`learning_rate: 0.05`）
3. `conf/config.yaml`の`exp_name`を変更（例：`exp002_lr005`）
4. `python main.py`で実行

### 3. 異なるモデルを試す

1. `conf/model/`に新しいモデル設定ファイルを作成（例：`XGBoost.yaml`）
2. `conf/config.yaml`の`model`を変更（例：`model: xgboost`）
3. `main.py`でモデル読み込み部分を対応するモデルに変更
4. `python main.py`で実行

### 4. ノートブックで探索的分析を行う

1. `notebooks/init.ipynb`を開く
2. データのパスを`../data/raw/`に変更
3. 分析を実行
4. 良い特徴量が見つかったら`src/features.py`に反映

### 5. 保存したモデルで予測する

```python
from src.models import load_models, predict_with_models
from src.features import prepare_test_data
from pathlib import Path

# モデルを読み込み
models = load_models(Path("outputs/models"))

# テストデータの準備（既に特徴量が生成済みと仮定）
X_test = prepare_test_data(sample_submission, features)

# 予測
y_sub = predict_with_models(models, X_test)
```

---

## 📊 データフロー

```
1. データ読み込み (preprocessing.py)
   data.csv, train_flag.csv, sample_submission.csv
   ↓
2. データ前処理 (preprocessing.py)
   日付型への変換など
   ↓
3. 特徴量生成 (features.py)
   来店回数、平均単価、経過日数など
   ↓
4. データ準備 (features.py)
   X_train, y_train, X_test
   ↓
5. モデル学習 (models.py)
   クロスバリデーション実行
   ↓
6. 評価・予測 (models.py)
   CVスコア計算、テストデータ予測
   ↓
7. 出力 (main.py)
   モデル保存、提出ファイル作成
```

---

## ⚠️ 注意事項

1. **データのパス**: 
   - 生データは`data/raw/`に配置
   - ノートブックからは`../data/raw/`で参照

2. **出力先**:
   - モデル: `outputs/models/`
   - 提出ファイル: `outputs/submissions/`

3. **設定ファイル**:
   - `exp_name`を変更すると、提出ファイル名が変わる
   - 実験ごとに`exp_name`を変更することを推奨

4. **特徴量の追加**:
   - `create_features()`関数内で特徴量を追加
   - 必ず`user_id`を含むDataFrameを返すこと

5. **モデルの保存**:
   - デフォルトで`outputs/models/`に保存
   - 各foldのモデルが個別に保存される

---

## 🐛 トラブルシューティング

### エラー: "dateカラムが存在しません"
- `data.csv`に`date`カラムが含まれているか確認
- `preprocessing.py`の`clean_data()`関数を確認

### エラー: "FileNotFoundError"
- データファイルが`data/raw/`に存在するか確認
- パスが正しいか確認

### CVスコアが表示されない
- `train_lightgbm_cv()`の戻り値を確認
- `evaluate_cv()`が正しく呼ばれているか確認

---

## 📚 参考

- 各モジュールの詳細は、各ファイルのdocstringを参照
- ノートブック（`notebooks/init.ipynb`）で探索的分析の例を確認
