from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse

from src.preprocessing import RecommendationDataPreprocessor


RETRIEVAL_NUMERIC_COLUMNS = ["retrieval_score", "retrieval_rank", "is_cold_item"]
RETRIEVAL_CATEGORICAL_COLUMNS = ["retrieval_source"]


@dataclass(slots=True)
class RankerDataset:
    """
    Feature matrix and metadata for CatBoostRanker train or inference.
    """

    pairs_df: pd.DataFrame
    feature_matrix: sparse.csr_matrix
    feature_names: list[str]
    group_ids: np.ndarray
    group_keys: np.ndarray
    labels: np.ndarray | None = None


def ensure_columns_present(df: pd.DataFrame, required_columns: list[str] | tuple[str, ...]) -> None:
    """
    Validate that the dataframe contains all required columns.
    """
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")


def build_label_frame(
    interactions_df: pd.DataFrame,
    user_id_col: str,
    item_id_col: str,
    value_col: str,
    use_interaction_value_as_label: bool,
) -> pd.DataFrame:
    """
    Aggregate observed interactions into one label row per user-item pair.
    """
    ensure_columns_present(interactions_df, [user_id_col, item_id_col, value_col])

    if use_interaction_value_as_label:
        label_df = (
            interactions_df[[user_id_col, item_id_col, value_col]]
            .groupby([user_id_col, item_id_col], as_index=False)[value_col]
            .max()
            .rename(columns={value_col: "label"})
        )
        label_df["label"] = pd.to_numeric(label_df["label"], errors="coerce").fillna(0.0).astype(np.float32)
        return label_df

    label_df = interactions_df[[user_id_col, item_id_col]].drop_duplicates().copy()
    label_df["label"] = np.float32(1.0)
    return label_df


def attach_labels_to_pairs(
    pairs_df: pd.DataFrame,
    interactions_df: pd.DataFrame,
    user_id_col: str,
    item_id_col: str,
    value_col: str,
    use_interaction_value_as_label: bool,
) -> pd.DataFrame:
    """
    Left-join observed labels onto candidate pairs.
    """
    label_df = build_label_frame(
        interactions_df=interactions_df,
        user_id_col=user_id_col,
        item_id_col=item_id_col,
        value_col=value_col,
        use_interaction_value_as_label=use_interaction_value_as_label,
    )
    labeled_pairs_df = pairs_df.merge(label_df, on=[user_id_col, item_id_col], how="left")
    labeled_pairs_df["label"] = labeled_pairs_df["label"].fillna(0.0).astype(np.float32)
    return labeled_pairs_df


def build_retrieval_feature_frame(pairs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a dense retrieval-feature frame with stable defaults.
    """
    retrieval_df = pd.DataFrame(index=pairs_df.index)
    retrieval_score = (
        pairs_df["retrieval_score"]
        if "retrieval_score" in pairs_df.columns
        else pd.Series(0.0, index=pairs_df.index)
    )
    retrieval_rank = (
        pairs_df["retrieval_rank"]
        if "retrieval_rank" in pairs_df.columns
        else pd.Series(0.0, index=pairs_df.index)
    )
    is_cold_item = (
        pairs_df["is_cold_item"]
        if "is_cold_item" in pairs_df.columns
        else pd.Series(False, index=pairs_df.index)
    )
    retrieval_source = (
        pairs_df["retrieval_source"]
        if "retrieval_source" in pairs_df.columns
        else pd.Series("__missing__", index=pairs_df.index)
    )

    retrieval_df["retrieval_score"] = pd.to_numeric(retrieval_score, errors="coerce").fillna(0.0)
    retrieval_df["retrieval_rank"] = pd.to_numeric(retrieval_rank, errors="coerce").fillna(0.0)
    retrieval_df["is_cold_item"] = (
        is_cold_item
        .fillna(False)
        .astype(bool)
        .astype(np.float32)
    )
    retrieval_df["retrieval_source"] = (
        retrieval_source
        .fillna("__missing__")
        .astype(str)
    )
    return retrieval_df


def encode_retrieval_features(retrieval_feature_df: pd.DataFrame) -> tuple[sparse.csr_matrix, list[str]]:
    """
    Encode retrieval features into a sparse matrix.
    """
    numeric_matrix = sparse.csr_matrix(
        retrieval_feature_df[RETRIEVAL_NUMERIC_COLUMNS].to_numpy(dtype=np.float32, copy=False)
    )
    retrieval_feature_names = list(RETRIEVAL_NUMERIC_COLUMNS)

    source_dummies = pd.get_dummies(
        retrieval_feature_df["retrieval_source"],
        prefix="retrieval_source",
        dtype=np.float32,
    )
    if not source_dummies.empty:
        source_matrix = sparse.csr_matrix(source_dummies.to_numpy(dtype=np.float32, copy=False))
        numeric_matrix = sparse.hstack([numeric_matrix, source_matrix], format="csr")
        retrieval_feature_names.extend(source_dummies.columns.tolist())

    return numeric_matrix, retrieval_feature_names


def build_group_ids(user_ids: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert user ids to ranker group ids and preserve the original keys.
    """
    group_keys = user_ids.astype(str).to_numpy()
    group_ids, _ = pd.factorize(group_keys, sort=False)
    return group_ids.astype(np.int64, copy=False), group_keys


def extract_entity_feature_names(
    preprocessor: RecommendationDataPreprocessor,
    pair_feature_df: pd.DataFrame,
) -> list[str]:
    """
    Return encoded entity-feature names if the fitted transformer exposes them.
    """
    if preprocessor.feature_preprocessor is not None:
        return [str(name) for name in preprocessor.feature_preprocessor.get_feature_names_out()]

    return [
        column for column in pair_feature_df.columns if column not in [preprocessor.user_id_col, preprocessor.item_id_col]
    ]


def combine_feature_blocks(
    entity_feature_matrix: sparse.spmatrix,
    retrieval_feature_matrix: sparse.spmatrix,
) -> sparse.csr_matrix:
    """
    Concatenate entity and retrieval feature blocks.
    """
    return sparse.hstack([entity_feature_matrix, retrieval_feature_matrix], format="csr")


def build_ranker_dataset(
    candidate_pairs_df: pd.DataFrame,
    preprocessor: RecommendationDataPreprocessor,
    user_features_df: pd.DataFrame,
    item_features_df: pd.DataFrame,
    interactions_df: pd.DataFrame | None = None,
    use_interaction_value_as_label: bool = False,
) -> RankerDataset:
    """
    Build a ranker-ready dataset from candidate pairs and feature tables.
    """
    ensure_columns_present(candidate_pairs_df, [preprocessor.user_id_col, preprocessor.item_id_col])

    prepared_pairs_df = preprocessor.prepare_pair_dataframe(candidate_pairs_df)
    pair_feature_df, entity_feature_matrix = preprocessor.transform_pairs(
        prepared_pairs_df,
        user_features_df=user_features_df,
        item_features_df=item_features_df,
    )

    retrieval_feature_df = build_retrieval_feature_frame(prepared_pairs_df)
    retrieval_feature_matrix, retrieval_feature_names = encode_retrieval_features(retrieval_feature_df)

    combined_pairs_df = pair_feature_df.join(retrieval_feature_df)
    feature_matrix = combine_feature_blocks(entity_feature_matrix, retrieval_feature_matrix)

    entity_feature_names = extract_entity_feature_names(preprocessor, pair_feature_df)
    feature_names = entity_feature_names + retrieval_feature_names
    group_ids, group_keys = build_group_ids(combined_pairs_df[preprocessor.user_id_col])

    labels: np.ndarray | None = None
    if interactions_df is not None:
        labeled_pairs_df = attach_labels_to_pairs(
            pairs_df=combined_pairs_df,
            interactions_df=interactions_df,
            user_id_col=preprocessor.user_id_col,
            item_id_col=preprocessor.item_id_col,
            value_col=preprocessor.value_col,
            use_interaction_value_as_label=use_interaction_value_as_label,
        )
        combined_pairs_df = labeled_pairs_df
        labels = labeled_pairs_df["label"].to_numpy(dtype=np.float32, copy=False)

    return RankerDataset(
        pairs_df=combined_pairs_df,
        feature_matrix=feature_matrix,
        feature_names=feature_names,
        group_ids=group_ids,
        group_keys=group_keys,
        labels=labels,
    )


@dataclass(slots=True)
class RankerFeatureBuilder:
    """
    Build train and inference feature matrices for the ranking stage.
    """

    preprocessor: RecommendationDataPreprocessor
    use_interaction_value_as_label: bool = False

    def build_training_dataset(
        self,
        candidate_pairs_df: pd.DataFrame,
        interactions_df: pd.DataFrame,
        user_features_df: pd.DataFrame,
        item_features_df: pd.DataFrame,
    ) -> RankerDataset:
        """
        Build a supervised dataset for ranker training.
        """
        return build_ranker_dataset(
            candidate_pairs_df=candidate_pairs_df,
            preprocessor=self.preprocessor,
            user_features_df=user_features_df,
            item_features_df=item_features_df,
            interactions_df=interactions_df,
            use_interaction_value_as_label=self.use_interaction_value_as_label,
        )

    def build_inference_dataset(
        self,
        candidate_pairs_df: pd.DataFrame,
        user_features_df: pd.DataFrame,
        item_features_df: pd.DataFrame,
    ) -> RankerDataset:
        """
        Build an inference-time dataset without labels.
        """
        return build_ranker_dataset(
            candidate_pairs_df=candidate_pairs_df,
            preprocessor=self.preprocessor,
            user_features_df=user_features_df,
            item_features_df=item_features_df,
            interactions_df=None,
            use_interaction_value_as_label=self.use_interaction_value_as_label,
        )
