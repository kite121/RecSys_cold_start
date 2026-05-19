from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from pandas.api.types import is_numeric_dtype
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class FeatureRoleSplit:
    """
    Role assignment for optional CSV feature columns.
    """

    user_feature_cols: list[str]
    item_feature_cols: list[str]
    ignored_feature_cols: list[str]


@dataclass
class PreprocessingArtifacts:
    """
    Result of preprocessing fit on the training dataframe.
    """

    interactions_df: pd.DataFrame
    user_features_df: pd.DataFrame
    item_features_df: pd.DataFrame
    feature_roles: FeatureRoleSplit
    numeric_cols: list[str]
    categorical_cols: list[str]
    preprocessor: ColumnTransformer | None


def ensure_columns_present(df: pd.DataFrame, required_columns: list[str] | tuple[str, ...]) -> None:
    """
    Validate that all required columns exist in the dataframe.
    """
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")


def standardize_guid(value: object) -> object:
    """
    Normalize string ids the same way as in the reference ``recommender 2`` project.

    Args:
        value: User or item identifier.

    Returns:
        Lowercased string without braces if the value is a string.
    """
    if not isinstance(value, str):
        return value
    return value.strip("{}").lower()


def normalize_identifier_series(series: pd.Series) -> pd.Series:
    """
    Normalize identifier values from a raw CSV column.

    Missing values and blank strings are preserved as ``pd.NA`` so they can be
    safely removed during preprocessing.
    """
    normalized = series.map(
        lambda value: pd.NA
        if pd.isna(value)
        else standardize_guid(str(value).strip())
    )
    normalized = normalized.replace({"": pd.NA, "nan": pd.NA, "none": pd.NA})
    return normalized


def coerce_feature_dtypes(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """
    Try to cast numeric-looking optional features to numeric dtype.

    This keeps columns such as ``item_price`` numeric even if they came from the
    CSV as strings, while leaving genuinely categorical features untouched.
    """
    coerced_df = df.copy()
    for column in feature_columns:
        converted = pd.to_numeric(coerced_df[column], errors="coerce")
        original_non_null = coerced_df[column].notna().sum()
        converted_non_null = converted.notna().sum()
        if original_non_null > 0 and converted_non_null == original_non_null:
            coerced_df[column] = converted
    return coerced_df


@dataclass
class RecommendationDataPreprocessor:
    """
    Reusable data preparation component for recommendation CSV files.

    Expected train CSV format:
    - required columns: ``user_id``, ``item_id``, ``value``
    - optional user features: columns prefixed with ``user_``
    - optional item features: columns prefixed with ``item_``
    - other columns are ignored by the pair-feature encoder
    """

    user_id_col: str = "user_id"
    item_id_col: str = "item_id"
    value_col: str = "value"
    user_prefix: str = "user_"
    item_prefix: str = "item_"

    feature_roles: FeatureRoleSplit | None = None
    numeric_cols: list[str] | None = None
    categorical_cols: list[str] | None = None
    feature_preprocessor: ColumnTransformer | None = None

    def prepare_training_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and normalize a training dataframe.

        This step:
        - validates mandatory columns
        - standardizes ids
        - replaces missing values in ``value`` with 0
        - converts value to float
        - drops duplicate rows
        - drops rows with missing required identifiers
        """
        ensure_columns_present(df, [self.user_id_col, self.item_id_col, self.value_col])

        prepared_df = df.copy()
        prepared_df[self.user_id_col] = normalize_identifier_series(prepared_df[self.user_id_col])
        prepared_df[self.item_id_col] = normalize_identifier_series(prepared_df[self.item_id_col])
        prepared_df[self.value_col] = prepared_df[self.value_col].fillna(0)
        prepared_df[self.value_col] = pd.to_numeric(prepared_df[self.value_col], errors="coerce").fillna(0.0)
        prepared_df = prepared_df.dropna(subset=[self.user_id_col, self.item_id_col])
        prepared_df = prepared_df.drop_duplicates(keep="last").reset_index(drop=True)

        feature_roles = self.infer_feature_roles(prepared_df)
        feature_columns = feature_roles.user_feature_cols + feature_roles.item_feature_cols
        if feature_columns:
            prepared_df = coerce_feature_dtypes(prepared_df, feature_columns)
        return prepared_df

    def prepare_pair_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and normalize a candidate-pair dataframe for inference.

        Only ``user_id`` and ``item_id`` are mandatory at inference time.
        """
        ensure_columns_present(df, [self.user_id_col, self.item_id_col])

        prepared_df = df.copy()
        prepared_df[self.user_id_col] = normalize_identifier_series(prepared_df[self.user_id_col])
        prepared_df[self.item_id_col] = normalize_identifier_series(prepared_df[self.item_id_col])
        prepared_df = prepared_df.dropna(subset=[self.user_id_col, self.item_id_col])
        prepared_df = prepared_df.drop_duplicates(keep="last").reset_index(drop=True)
        return prepared_df

    def infer_feature_roles(self, df: pd.DataFrame) -> FeatureRoleSplit:
        """
        Split raw feature columns into user-level and item-level groups.
        """
        ignored_base_cols = {self.user_id_col, self.item_id_col, self.value_col}
        user_feature_cols = [
            column
            for column in df.columns
            if column.startswith(self.user_prefix) and column not in ignored_base_cols
        ]
        item_feature_cols = [
            column
            for column in df.columns
            if column.startswith(self.item_prefix) and column not in ignored_base_cols
        ]
        known_feature_cols = set(user_feature_cols + item_feature_cols)
        ignored_feature_cols = [
            column for column in df.columns if column not in ignored_base_cols and column not in known_feature_cols
        ]
        return FeatureRoleSplit(
            user_feature_cols=user_feature_cols,
            item_feature_cols=item_feature_cols,
            ignored_feature_cols=ignored_feature_cols,
        )

    def build_entity_feature_tables(
        self,
        interactions_df: pd.DataFrame,
        feature_roles: FeatureRoleSplit,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Build one row per user and one row per item feature tables.
        """
        user_columns = [self.user_id_col] + feature_roles.user_feature_cols
        item_columns = [self.item_id_col] + feature_roles.item_feature_cols

        user_features_df = interactions_df[user_columns].drop_duplicates(subset=[self.user_id_col], keep="last")
        item_features_df = interactions_df[item_columns].drop_duplicates(subset=[self.item_id_col], keep="last")
        return user_features_df.reset_index(drop=True), item_features_df.reset_index(drop=True)

    def build_pair_feature_frame(
        self,
        pairs_df: pd.DataFrame,
        user_features_df: pd.DataFrame,
        item_features_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Merge candidate pairs with stored user/item feature tables.
        """
        merged_df = pairs_df[[self.user_id_col, self.item_id_col]].copy()
        merged_df = merged_df.merge(user_features_df, on=self.user_id_col, how="left")
        merged_df = merged_df.merge(item_features_df, on=self.item_id_col, how="left")
        return merged_df

    def detect_feature_types(self, pair_feature_df: pd.DataFrame) -> tuple[list[str], list[str]]:
        """
        Detect numeric and categorical columns for the shared feature encoder.
        """
        feature_columns = [column for column in pair_feature_df.columns if column not in [self.user_id_col, self.item_id_col]]
        numeric_cols = [column for column in feature_columns if is_numeric_dtype(pair_feature_df[column])]
        categorical_cols = [column for column in feature_columns if column not in numeric_cols]
        return numeric_cols, categorical_cols

    def build_feature_preprocessor(
        self,
        numeric_cols: list[str],
        categorical_cols: list[str],
    ) -> ColumnTransformer | None:
        """
        Build a preprocessing transformer:
        - numeric columns: median imputation + standard scaling
        - categorical columns: constant imputation + one-hot encoding
        """
        transformers = []

        if numeric_cols:
            numeric_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )
            transformers.append(("num", numeric_pipeline, numeric_cols))

        if categorical_cols:
            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="constant", fill_value="__missing__")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
                ]
            )
            transformers.append(("cat", categorical_pipeline, categorical_cols))

        if not transformers:
            return None

        return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=1.0)

    def fit(self, raw_df: pd.DataFrame) -> PreprocessingArtifacts:
        """
        Fit preprocessing artifacts on a training dataframe.
        """
        interactions_df = self.prepare_training_dataframe(raw_df)
        feature_roles = self.infer_feature_roles(interactions_df)
        user_features_df, item_features_df = self.build_entity_feature_tables(interactions_df, feature_roles)
        pair_feature_df = self.build_pair_feature_frame(
            interactions_df[[self.user_id_col, self.item_id_col]],
            user_features_df=user_features_df,
            item_features_df=item_features_df,
        )
        numeric_cols, categorical_cols = self.detect_feature_types(pair_feature_df)
        feature_preprocessor = self.build_feature_preprocessor(numeric_cols, categorical_cols)

        if feature_preprocessor is not None:
            feature_preprocessor.fit(pair_feature_df[numeric_cols + categorical_cols])

        self.feature_roles = feature_roles
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.feature_preprocessor = feature_preprocessor

        return PreprocessingArtifacts(
            interactions_df=interactions_df,
            user_features_df=user_features_df,
            item_features_df=item_features_df,
            feature_roles=feature_roles,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            preprocessor=feature_preprocessor,
        )

    def transform_pairs(
        self,
        pairs_df: pd.DataFrame,
        user_features_df: pd.DataFrame,
        item_features_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, sparse.spmatrix]:
        """
        Transform user-item pairs into the encoded feature matrix used by the ranker.
        """
        if self.feature_roles is None or self.numeric_cols is None or self.categorical_cols is None:
            raise RuntimeError("Preprocessor must be fitted before calling transform_pairs.")

        prepared_pairs_df = self.prepare_pair_dataframe(pairs_df)
        pair_feature_df = self.build_pair_feature_frame(
            prepared_pairs_df,
            user_features_df=user_features_df,
            item_features_df=item_features_df,
        )

        if self.feature_preprocessor is None:
            empty_matrix = sparse.csr_matrix((len(pair_feature_df), 0))
            return pair_feature_df, empty_matrix

        feature_columns = self.numeric_cols + self.categorical_cols
        feature_matrix = self.feature_preprocessor.transform(pair_feature_df[feature_columns])
        return pair_feature_df, feature_matrix
