"""
データ前処理（データクレンジング）
"""
import pandas as pd
from pathlib import Path

from .utils import get_data_path


def load_data(data_type: str = "raw") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    データを読み込む

    Args:
        data_type: データタイプ（"raw", "interim", "processed"）
    Returns:
        (data, train_flag, sample_submission) のタプル
    """
    data_path = get_data_path(data_type)

    data = pd.read_csv(data_path / "data.csv")
    train_flag = pd.read_csv(data_path / "train_flag.csv")
    sample_submission = pd.read_csv(data_path / "sample_submission.csv")

    return data, train_flag, sample_submission


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    データをクレンジングする

    Args:
        data: 生データ
    Returns:
        クレンジング済みデータ
    """
    data = data.copy()

    # dateカラムを日付型に変換（YYYYMMDD形式の整数を日付に変換）
    if "date" in data.columns:
        data["date"] = pd.to_datetime(data["date"], format="%Y%m%d")
    else:
        raise ValueError("dateカラムが存在しません")
    return data
