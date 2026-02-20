"""
共通ユーティリティ関数
"""
from pathlib import Path
from typing import Optional

def get_project_root() -> Path:
    """プロジェクトルートのパスを取得"""
    return Path(__file__).parent.parent


def get_data_path(data_type: str = "raw") -> Path:
    """
    データディレクトリのパスを取得

    Args:
        data_type: "raw", "interim", "processed" のいずれか
    Returns:
        データディレクトリのPathオブジェクト
    """
    project_root = get_project_root()
    return project_root / "data" / data_type


def get_output_path(subdir: str = "submissions") -> Path:
    """
    出力ディレクトリのパスを取得

    Args:
        subdir: "models", "submissions" などのサブディレクトリ名
    Returns:
        出力ディレクトリのPathオブジェクト
    """
    project_root = get_project_root()
    return project_root / "outputs" / subdir


def ensure_dir(path: Path) -> Path:
    """
    ディレクトリが存在しない場合は作成

    Args:
        path: ディレクトリのPathオブジェクト
    Returns:
        作成された（または既存の）ディレクトリのPathオブジェクト
    """
    path.mkdir(parents=True, exist_ok=True)
    return path
