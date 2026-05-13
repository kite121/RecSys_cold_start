from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from scipy import sparse

from src.als_model import ALSRecommender
from src.candidate_generator import CandidateGenerationResult, CandidateGenerator
from src.cold_vector_builder import ColdItemVectorBuilder, ColdVectorBuildResult
from src.maxvol_selector import DiverseSelectionResult, MaxVolSelector
from src.popular_selector import PopularItemsSelector, PopularSelectionResult
from src.similarity_index import ItemSimilarityIndex, NeighborSearchResult


@dataclass(slots=True)
class ItemFeatureSpace:
    """
    Shared encoded item-feature space used by support and cold items.
    """

    item_catalog_df: pd.DataFrame
    feature_matrix: sparse.csr_matrix
    feature_names: list[str]
    item_to_row: dict[str, int]


@dataclass(slots=True)
class RetrievalArtifacts:
    """
    Fitted retrieval-stage artifacts for cold-item handling.
    """

    item_feature_space: ItemFeatureSpace
    popular_result: PopularSelectionResult
    support_selection_result: DiverseSelectionResult
    neighbors_result: NeighborSearchResult
    cold_vector_result: ColdVectorBuildResult
    warm_candidate_item_ids: list[str]
    cold_candidate_item_ids: list[str]


def ensure_columns_present(df: pd.DataFrame, required_columns: list[str] | tuple[str, ...]) -> None:
    """
    Validate that required columns are present in the dataframe.
    """
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")


def normalize_item_ids(item_ids: set[str] | list[str] | tuple[str, ...]) -> list[str]:
    """
    Normalize item ids to strings while preserving input order.
    """
    normalized_ids: list[str] = []
    seen_ids: set[str] = set()
    for item_id in item_ids:
        normalized_item_id = str(item_id)
        if normalized_item_id not in seen_ids:
            normalized_ids.append(normalized_item_id)
            seen_ids.add(normalized_item_id)
    return normalized_ids


def build_item_catalog(
    item_features_df: pd.DataFrame,
    all_item_ids: list[str],
    item_id_col: str,
) -> pd.DataFrame:
    """
    Build one-row-per-item catalog, keeping ids even when features are missing.
    """
    ensure_columns_present(item_features_df, [item_id_col])

    base_catalog_df = pd.DataFrame({item_id_col: normalize_item_ids(all_item_ids)})
    deduplicated_features_df = item_features_df.copy()
    deduplicated_features_df[item_id_col] = deduplicated_features_df[item_id_col].astype(str)
    deduplicated_features_df = deduplicated_features_df.drop_duplicates(subset=[item_id_col], keep="last")

    item_catalog_df = base_catalog_df.merge(deduplicated_features_df, on=item_id_col, how="left")
    return item_catalog_df.reset_index(drop=True)


def encode_item_feature_space(
    item_catalog_df: pd.DataFrame,
    item_id_col: str,
) -> tuple[sparse.csr_matrix, list[str]]:
    """
    Encode item features once for the entire catalog.
    """
    feature_columns = [column for column in item_catalog_df.columns if column != item_id_col]
    if not feature_columns:
        return sparse.csr_matrix((len(item_catalog_df), 0), dtype=np.float32), []

    numeric_cols = [column for column in feature_columns if is_numeric_dtype(item_catalog_df[column])]
    categorical_cols = [column for column in feature_columns if column not in numeric_cols]

    matrices: list[sparse.csr_matrix] = []
    feature_names: list[str] = []

    if numeric_cols:
        numeric_df = item_catalog_df[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        matrices.append(sparse.csr_matrix(numeric_df.to_numpy(dtype=np.float32, copy=False)))
        feature_names.extend(numeric_cols)

    if categorical_cols:
        categorical_df = item_catalog_df[categorical_cols].copy()
        for column in categorical_cols:
            categorical_df[column] = categorical_df[column].fillna("__missing__").astype(str)
        dummies_df = pd.get_dummies(categorical_df, prefix=categorical_cols, dtype=np.float32)
        if not dummies_df.empty:
            matrices.append(sparse.csr_matrix(dummies_df.to_numpy(dtype=np.float32, copy=False)))
            feature_names.extend(dummies_df.columns.astype(str).tolist())

    if not matrices:
        return sparse.csr_matrix((len(item_catalog_df), 0), dtype=np.float32), []

    return sparse.hstack(matrices, format="csr"), feature_names


def build_item_feature_space(
    item_features_df: pd.DataFrame,
    all_item_ids: list[str],
    item_id_col: str = "item_id",
) -> ItemFeatureSpace:
    """
    Build the shared item catalog and its encoded feature space.
    """
    item_catalog_df = build_item_catalog(
        item_features_df=item_features_df,
        all_item_ids=all_item_ids,
        item_id_col=item_id_col,
    )
    feature_matrix, feature_names = encode_item_feature_space(item_catalog_df=item_catalog_df, item_id_col=item_id_col)
    item_to_row = {str(item_id): row_index for row_index, item_id in enumerate(item_catalog_df[item_id_col].astype(str))}
    return ItemFeatureSpace(
        item_catalog_df=item_catalog_df,
        feature_matrix=feature_matrix,
        feature_names=feature_names,
        item_to_row=item_to_row,
    )


def subset_item_feature_space(
    item_feature_space: ItemFeatureSpace,
    item_ids: list[str],
    item_id_col: str,
) -> tuple[pd.DataFrame, sparse.csr_matrix]:
    """
    Slice the shared item feature space in the requested item order.
    """
    row_indices = [
        item_feature_space.item_to_row[item_id]
        for item_id in normalize_item_ids(item_ids)
        if item_id in item_feature_space.item_to_row
    ]

    if not row_indices:
        empty_catalog_df = pd.DataFrame(columns=item_feature_space.item_catalog_df.columns.tolist())
        empty_matrix = sparse.csr_matrix((0, item_feature_space.feature_matrix.shape[1]), dtype=np.float32)
        return empty_catalog_df, empty_matrix

    subset_items_df = item_feature_space.item_catalog_df.iloc[row_indices].copy().reset_index(drop=True)
    subset_matrix = item_feature_space.feature_matrix[row_indices]
    if item_id_col in subset_items_df.columns:
        subset_items_df[item_id_col] = subset_items_df[item_id_col].astype(str)
    return subset_items_df, subset_matrix


def merge_item_metadata(
    base_items_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    item_id_col: str,
) -> pd.DataFrame:
    """
    Merge auxiliary metadata into an item table without changing row order.
    """
    if metadata_df.empty:
        return base_items_df.reset_index(drop=True)
    merged_df = base_items_df.merge(metadata_df, on=item_id_col, how="left")
    return merged_df.reset_index(drop=True)


def fit_retrieval_artifacts(
    interactions_df: pd.DataFrame,
    item_features_df: pd.DataFrame,
    als_model: ALSRecommender,
    warm_item_ids: set[str] | list[str] | tuple[str, ...],
    cold_item_ids: set[str] | list[str] | tuple[str, ...],
    item_id_col: str = "item_id",
    user_id_col: str = "user_id",
    value_col: str = "value",
    event_type_col: str | None = None,
    timestamp_col: str | None = None,
    top_n_popular: int = 5000,
    top_k_diverse: int = 500,
    top_m_neighbors: int = 20,
    similarity_metric: str = "cosine",
    max_projection_dim: int = 64,
    random_state: int = 42,
    decay_rate: float = 0.05,
    user_cap: float = 5.0,
    unique_user_bonus: float = 0.2,
    neighbor_weighting_strategy: str = "similarity",
) -> RetrievalArtifacts:
    """
    Fit all retrieval-stage artifacts required for cold-item recommendation.
    """
    ensure_columns_present(interactions_df, [user_id_col, item_id_col, value_col])
    ensure_columns_present(item_features_df, [item_id_col])

    warm_ids = normalize_item_ids(warm_item_ids)
    cold_ids = normalize_item_ids(cold_item_ids)

    all_item_ids = normalize_item_ids(warm_ids + cold_ids)
    item_feature_space = build_item_feature_space(
        item_features_df=item_features_df,
        all_item_ids=all_item_ids,
        item_id_col=item_id_col,
    )

    warm_interactions_df = interactions_df[interactions_df[item_id_col].astype(str).isin(warm_ids)].copy()
    popular_selector = PopularItemsSelector(
        item_col=item_id_col,
        user_col=user_id_col,
        value_col=value_col,
        event_type_col=event_type_col,
        timestamp_col=timestamp_col,
        decay_rate=decay_rate,
        user_cap=user_cap,
        unique_user_bonus=unique_user_bonus,
    )
    popular_result = popular_selector.select(warm_interactions_df, top_n=top_n_popular)

    popular_items_df, popular_feature_matrix = subset_item_feature_space(
        item_feature_space=item_feature_space,
        item_ids=popular_result.top_item_ids,
        item_id_col=item_id_col,
    )
    popular_items_df = merge_item_metadata(
        base_items_df=popular_items_df,
        metadata_df=popular_result.top_items_df,
        item_id_col=item_id_col,
    )

    maxvol_selector = MaxVolSelector(
        item_id_col=item_id_col,
        max_projection_dim=max_projection_dim,
        random_state=random_state,
    )
    support_selection_result = maxvol_selector.select(
        items_df=popular_items_df,
        feature_matrix=popular_feature_matrix,
        top_k=top_k_diverse,
    )

    support_feature_matrix = (
        popular_feature_matrix[support_selection_result.selected_indices]
        if support_selection_result.selected_indices
        else sparse.csr_matrix((0, popular_feature_matrix.shape[1]), dtype=np.float32)
    )

    cold_items_df, cold_feature_matrix = subset_item_feature_space(
        item_feature_space=item_feature_space,
        item_ids=cold_ids,
        item_id_col=item_id_col,
    )

    similarity_index = ItemSimilarityIndex(
        cold_item_col=item_id_col,
        support_item_col=item_id_col,
        similarity_metric=similarity_metric,
        max_projection_dim=max_projection_dim,
        random_state=random_state,
    )
    neighbors_result = similarity_index.find_neighbors(
        cold_items_df=cold_items_df,
        cold_feature_matrix=cold_feature_matrix,
        support_items_df=support_selection_result.selected_items_df,
        support_feature_matrix=support_feature_matrix,
        top_m=top_m_neighbors,
    )

    cold_vector_builder = ColdItemVectorBuilder(
        cold_item_col=item_id_col,
        neighbor_item_col="neighbor_item_id",
        similarity_col="similarity",
        weighting_strategy=neighbor_weighting_strategy,
    )
    cold_vector_result = cold_vector_builder.build(
        neighbors_df=neighbors_result.neighbors_df,
        als_model=als_model,
    )

    return RetrievalArtifacts(
        item_feature_space=item_feature_space,
        popular_result=popular_result,
        support_selection_result=support_selection_result,
        neighbors_result=neighbors_result,
        cold_vector_result=cold_vector_result,
        warm_candidate_item_ids=warm_ids,
        cold_candidate_item_ids=sorted(cold_vector_result.cold_vector_map.keys()),
    )


def generate_retrieval_candidates_for_user(
    als_model: ALSRecommender,
    user_id: str,
    retrieval_artifacts: RetrievalArtifacts,
    warm_candidates_per_user: int,
    cold_candidates_per_user: int,
    final_candidate_pool_size: int,
    exclude_seen: bool = True,
) -> CandidateGenerationResult:
    """
    Generate the retrieval candidate pool for one user from fitted artifacts.
    """
    candidate_generator = CandidateGenerator(
        warm_candidates_per_user=warm_candidates_per_user,
        cold_candidates_per_user=cold_candidates_per_user,
        final_candidate_pool_size=final_candidate_pool_size,
        exclude_seen=exclude_seen,
    )
    return candidate_generator.generate_for_user(
        als_model=als_model,
        user_id=str(user_id),
        cold_vector_map=retrieval_artifacts.cold_vector_result.cold_vector_map,
        warm_candidate_item_ids=retrieval_artifacts.warm_candidate_item_ids,
        cold_candidate_item_ids=retrieval_artifacts.cold_candidate_item_ids,
    )


def generate_retrieval_candidates_for_users(
    als_model: ALSRecommender,
    user_ids: list[str] | tuple[str, ...] | set[str],
    retrieval_artifacts: RetrievalArtifacts,
    warm_candidates_per_user: int,
    cold_candidates_per_user: int,
    final_candidate_pool_size: int,
    exclude_seen: bool = True,
) -> pd.DataFrame:
    """
    Generate retrieval candidate pools for many users and concatenate them.
    """
    candidate_frames: list[pd.DataFrame] = []
    for user_id in user_ids:
        generation_result = generate_retrieval_candidates_for_user(
            als_model=als_model,
            user_id=str(user_id),
            retrieval_artifacts=retrieval_artifacts,
            warm_candidates_per_user=warm_candidates_per_user,
            cold_candidates_per_user=cold_candidates_per_user,
            final_candidate_pool_size=final_candidate_pool_size,
            exclude_seen=exclude_seen,
        )
        if not generation_result.candidates_df.empty:
            candidate_frames.append(generation_result.candidates_df)

    if not candidate_frames:
        return pd.DataFrame(
            columns=["user_id", "item_id", "retrieval_score", "retrieval_source", "is_cold_item", "retrieval_rank"]
        )
    return pd.concat(candidate_frames, ignore_index=True)


@dataclass(slots=True)
class ColdItemRetrievalModel:
    """
    Retrieval-stage orchestration for support-set building and candidate generation.
    """

    item_id_col: str = "item_id"
    user_id_col: str = "user_id"
    value_col: str = "value"
    event_type_col: str | None = None
    timestamp_col: str | None = None

    top_n_popular: int = 5000
    top_k_diverse: int = 500
    top_m_neighbors: int = 20
    similarity_metric: str = "cosine"
    max_projection_dim: int = 64
    random_state: int = 42

    decay_rate: float = 0.05
    user_cap: float = 5.0
    unique_user_bonus: float = 0.2
    neighbor_weighting_strategy: str = "similarity"

    warm_candidates_per_user: int = 200
    cold_candidates_per_user: int = 200
    final_candidate_pool_size: int = 400
    exclude_seen: bool = True

    retrieval_artifacts: RetrievalArtifacts | None = field(default=None, init=False)

    def fit(
        self,
        interactions_df: pd.DataFrame,
        item_features_df: pd.DataFrame,
        als_model: ALSRecommender,
        warm_item_ids: set[str] | list[str] | tuple[str, ...],
        cold_item_ids: set[str] | list[str] | tuple[str, ...],
    ) -> RetrievalArtifacts:
        """
        Fit retrieval artifacts for the new cold-item pipeline.
        """
        self.retrieval_artifacts = fit_retrieval_artifacts(
            interactions_df=interactions_df,
            item_features_df=item_features_df,
            als_model=als_model,
            warm_item_ids=warm_item_ids,
            cold_item_ids=cold_item_ids,
            item_id_col=self.item_id_col,
            user_id_col=self.user_id_col,
            value_col=self.value_col,
            event_type_col=self.event_type_col,
            timestamp_col=self.timestamp_col,
            top_n_popular=self.top_n_popular,
            top_k_diverse=self.top_k_diverse,
            top_m_neighbors=self.top_m_neighbors,
            similarity_metric=self.similarity_metric,
            max_projection_dim=self.max_projection_dim,
            random_state=self.random_state,
            decay_rate=self.decay_rate,
            user_cap=self.user_cap,
            unique_user_bonus=self.unique_user_bonus,
            neighbor_weighting_strategy=self.neighbor_weighting_strategy,
        )
        return self.retrieval_artifacts

    def generate_for_user(self, als_model: ALSRecommender, user_id: str) -> CandidateGenerationResult:
        """
        Generate the retrieval candidate pool for one user.
        """
        if self.retrieval_artifacts is None:
            raise RuntimeError("Retrieval model must be fitted before candidate generation.")
        return generate_retrieval_candidates_for_user(
            als_model=als_model,
            user_id=str(user_id),
            retrieval_artifacts=self.retrieval_artifacts,
            warm_candidates_per_user=self.warm_candidates_per_user,
            cold_candidates_per_user=self.cold_candidates_per_user,
            final_candidate_pool_size=self.final_candidate_pool_size,
            exclude_seen=self.exclude_seen,
        )

    def generate_for_users(
        self,
        als_model: ALSRecommender,
        user_ids: list[str] | tuple[str, ...] | set[str],
    ) -> pd.DataFrame:
        """
        Generate retrieval candidate pools for multiple users.
        """
        if self.retrieval_artifacts is None:
            raise RuntimeError("Retrieval model must be fitted before candidate generation.")
        return generate_retrieval_candidates_for_users(
            als_model=als_model,
            user_ids=user_ids,
            retrieval_artifacts=self.retrieval_artifacts,
            warm_candidates_per_user=self.warm_candidates_per_user,
            cold_candidates_per_user=self.cold_candidates_per_user,
            final_candidate_pool_size=self.final_candidate_pool_size,
            exclude_seen=self.exclude_seen,
        )
