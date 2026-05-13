
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


DEFAULT_USER_ID_COL = "user_id"
DEFAULT_ITEM_ID_COL = "item_id"
DEFAULT_VALUE_COL = "value"
DEFAULT_REQUIRED_TRAIN_COLUMNS = (
    DEFAULT_USER_ID_COL,
    DEFAULT_ITEM_ID_COL,
    DEFAULT_VALUE_COL,
)


@dataclass
class CSVColumnGroups:
    """
    Role grouping for input CSV columns.

    The hybrid recommender requires only ``user_id``, ``item_id`` and ``value``.
    Optional user and item features can be passed with ``user_`` and ``item_``
    prefixes. Any other optional columns are preserved in the dataframe but are
    not used by the feature pipeline by default.
    """

    required_columns: list[str]
    user_feature_cols: list[str]
    item_feature_cols: list[str]
    other_optional_cols: list[str]


def ensure_columns_present(df: pd.DataFrame, required_columns: list[str] | tuple[str, ...]) -> None:
    """
    Validate that all required columns exist in the dataframe.
    """
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")


def ensure_file_exists(path: str | Path) -> Path:
    """
    Validate that a file exists and return a normalized Path object.
    """
    normalized_path = Path(path)
    if not normalized_path.exists():
        raise FileNotFoundError(f"File not found: {normalized_path}")
    return normalized_path


def ensure_csv_file(path: str | Path) -> Path:
    """
    Validate that the input path exists and points to a CSV file.
    """
    normalized_path = ensure_file_exists(path)
    if normalized_path.suffix.lower() != ".csv":
        raise ValueError(f"Expected a .csv file, got: {normalized_path}")
    return normalized_path


def infer_csv_column_groups(
    df: pd.DataFrame,
    required_columns: list[str] | tuple[str, ...] = DEFAULT_REQUIRED_TRAIN_COLUMNS,
    user_prefix: str = "user_",
    item_prefix: str = "item_",
) -> CSVColumnGroups:
    """
    Split CSV columns into required fields and optional feature groups.
    """
    ensure_columns_present(df, required_columns)
    required_set = set(required_columns)

    user_feature_cols = [
        column for column in df.columns if column.startswith(user_prefix) and column not in required_set
    ]
    item_feature_cols = [
        column for column in df.columns if column.startswith(item_prefix) and column not in required_set
    ]
    optional_known_cols = set(user_feature_cols + item_feature_cols)
    other_optional_cols = [
        column
        for column in df.columns
        if column not in required_set and column not in optional_known_cols
    ]

    return CSVColumnGroups(
        required_columns=list(required_columns),
        user_feature_cols=user_feature_cols,
        item_feature_cols=item_feature_cols,
        other_optional_cols=other_optional_cols,
    )


def load_csv_data(
    csv_path: str | Path,
    required_columns: list[str] | tuple[str, ...] | None = None,
    sep: str = ",",
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """
    Load a CSV file into a pandas dataframe.

    Args:
        csv_path: Path to input CSV.
        required_columns: Optional list of columns that must exist in the CSV.
        sep: CSV separator.
        encoding: CSV encoding.

    Returns:
        Input data as a dataframe.
    """
    normalized_path = ensure_csv_file(csv_path)

    try:
        dataframe = pd.read_csv(normalized_path, sep=sep, encoding=encoding)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"CSV file is empty: {normalized_path}") from exc

    if dataframe.empty and len(dataframe.columns) == 0:
        raise ValueError(f"CSV file has no columns: {normalized_path}")

    if required_columns is not None:
        ensure_columns_present(dataframe, required_columns)
    return dataframe


def load_training_csv_data(
    csv_path: str | Path,
    user_id_col: str = DEFAULT_USER_ID_COL,
    item_id_col: str = DEFAULT_ITEM_ID_COL,
    value_col: str = DEFAULT_VALUE_COL,
    user_prefix: str = "user_",
    item_prefix: str = "item_",
    sep: str = ",",
    encoding: str = "utf-8",
) -> tuple[pd.DataFrame, CSVColumnGroups]:
    """
    Load a training CSV for the hybrid recommender.

    Required columns:
    - ``user_id``
    - ``item_id``
    - ``value``

    All other columns are optional. Columns starting with ``user_`` and
    ``item_`` are treated as candidate features for the ranker.
    """
    required_columns = [user_id_col, item_id_col, value_col]
    dataframe = load_csv_data(
        csv_path=csv_path,
        required_columns=required_columns,
        sep=sep,
        encoding=encoding,
    )
    column_groups = infer_csv_column_groups(
        dataframe,
        required_columns=required_columns,
        user_prefix=user_prefix,
        item_prefix=item_prefix,
    )
    return dataframe, column_groups
