from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

from src.maxvol_selector import project_feature_matrix, row_normalize_dense


@dataclass(slots=True)
class NeighborSearchResult:
    """
    Result of nearest-neighbor search for cold items.
    """

    neighbors_df: pd.DataFrame
    num_cold_items: int
    num_support_items: int
    neighbors_per_item: int
    similarity_metric: str


def ensure_matching_rows(items_df: pd.DataFrame, feature_matrix: sparse.spmatrix | np.ndarray) -> None:
    """
    Validate that the item table and its feature matrix have the same row count.
    """
    if len(items_df) != feature_matrix.shape[0]:
        raise ValueError(
            "items_df and feature_matrix must contain the same number of rows. "
            f"Got len(items_df)={len(items_df)} and feature_matrix.shape[0]={feature_matrix.shape[0]}."
        )


def project_joint_feature_matrices(
    cold_feature_matrix: sparse.spmatrix | np.ndarray,
    support_feature_matrix: sparse.spmatrix | np.ndarray,
    max_projection_dim: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Project cold and support matrices into the same compact feature space.

    Similarity search must compare vectors expressed in one shared basis. For
    that reason both matrices are vertically stacked, jointly projected, and
    then split back into cold/support blocks.
    """
    if cold_feature_matrix.shape[1] != support_feature_matrix.shape[1]:
        raise ValueError(
            "cold_feature_matrix and support_feature_matrix must have the same number of columns. "
            f"Got {cold_feature_matrix.shape[1]} and {support_feature_matrix.shape[1]}."
        )

    cold_num_rows = cold_feature_matrix.shape[0]
    if sparse.issparse(cold_feature_matrix) or sparse.issparse(support_feature_matrix):
        joint_matrix = sparse.vstack([cold_feature_matrix, support_feature_matrix], format="csr")
    else:
        joint_matrix = np.vstack(
            [
                np.asarray(cold_feature_matrix, dtype=np.float64),
                np.asarray(support_feature_matrix, dtype=np.float64),
            ]
        )

    projected_joint_matrix = project_feature_matrix(
        feature_matrix=joint_matrix,
        max_projection_dim=max_projection_dim,
        random_state=random_state,
    )
    normalized_joint_matrix = row_normalize_dense(projected_joint_matrix)

    cold_matrix_projected = normalized_joint_matrix[:cold_num_rows]
    support_matrix_projected = normalized_joint_matrix[cold_num_rows:]
    return cold_matrix_projected, support_matrix_projected


def compute_similarity_matrix(
    cold_matrix: np.ndarray,
    support_matrix: np.ndarray,
    similarity_metric: str,
) -> np.ndarray:
    """
    Compute pairwise similarity between cold items and support items.
    """
    if similarity_metric == "cosine":
        return cosine_similarity(cold_matrix, support_matrix)
    if similarity_metric == "dot":
        return cold_matrix @ support_matrix.T
    if similarity_metric == "euclidean":
        distances = euclidean_distances(cold_matrix, support_matrix)
        return -distances
    raise ValueError("similarity_metric must be one of: 'cosine', 'dot', 'euclidean'.")


def top_neighbor_indices(similarity_scores: np.ndarray, top_m: int) -> np.ndarray:
    """
    Select top-M support indices for every cold item row.
    """
    if similarity_scores.shape[1] == 0 or top_m <= 0:
        return np.empty((similarity_scores.shape[0], 0), dtype=int)

    effective_top_m = min(top_m, similarity_scores.shape[1])
    partition_indices = np.argpartition(-similarity_scores, kth=effective_top_m - 1, axis=1)[:, :effective_top_m]

    sorted_order = np.take_along_axis(similarity_scores, partition_indices, axis=1).argsort(axis=1)[:, ::-1]
    return np.take_along_axis(partition_indices, sorted_order, axis=1)


def build_neighbors_dataframe(
    cold_items_df: pd.DataFrame,
    support_items_df: pd.DataFrame,
    similarity_scores: np.ndarray,
    neighbor_indices: np.ndarray,
    cold_item_col: str,
    support_item_col: str,
) -> pd.DataFrame:
    """
    Convert nearest-neighbor indices into a flat dataframe.
    """
    rows: list[dict[str, float | int | str]] = []

    cold_item_ids = cold_items_df[cold_item_col].astype(str).tolist()
    support_item_ids = support_items_df[support_item_col].astype(str).tolist()

    for cold_row_index, cold_item_id in enumerate(cold_item_ids):
        for rank_position, support_row_index in enumerate(neighbor_indices[cold_row_index], start=1):
            similarity = float(similarity_scores[cold_row_index, support_row_index])
            rows.append(
                {
                    cold_item_col: cold_item_id,
                    "neighbor_item_id": support_item_ids[int(support_row_index)],
                    "neighbor_rank": rank_position,
                    "similarity": similarity,
                    "cold_row_index": cold_row_index,
                    "support_row_index": int(support_row_index),
                }
            )

    return pd.DataFrame(rows)


def find_item_neighbors(
    cold_items_df: pd.DataFrame,
    cold_feature_matrix: sparse.spmatrix | np.ndarray,
    support_items_df: pd.DataFrame,
    support_feature_matrix: sparse.spmatrix | np.ndarray,
    top_m: int,
    cold_item_col: str = "item_id",
    support_item_col: str = "item_id",
    similarity_metric: str = "cosine",
    max_projection_dim: int = 64,
    random_state: int = 42,
) -> NeighborSearchResult:
    """
    Find top-M nearest support items for every cold item.
    """
    if cold_item_col not in cold_items_df.columns:
        raise KeyError(f"Missing cold item id column: {cold_item_col}")
    if support_item_col not in support_items_df.columns:
        raise KeyError(f"Missing support item id column: {support_item_col}")

    ensure_matching_rows(cold_items_df, cold_feature_matrix)
    ensure_matching_rows(support_items_df, support_feature_matrix)

    if len(cold_items_df) == 0 or len(support_items_df) == 0 or top_m <= 0:
        empty_neighbors_df = pd.DataFrame(
            columns=[
                cold_item_col,
                "neighbor_item_id",
                "neighbor_rank",
                "similarity",
                "cold_row_index",
                "support_row_index",
            ]
        )
        return NeighborSearchResult(
            neighbors_df=empty_neighbors_df,
            num_cold_items=int(len(cold_items_df)),
            num_support_items=int(len(support_items_df)),
            neighbors_per_item=0,
            similarity_metric=similarity_metric,
        )

    cold_matrix_projected, support_matrix_projected = project_joint_feature_matrices(
        cold_feature_matrix=cold_feature_matrix,
        support_feature_matrix=support_feature_matrix,
        max_projection_dim=max_projection_dim,
        random_state=random_state,
    )

    similarity_scores = compute_similarity_matrix(
        cold_matrix=cold_matrix_projected,
        support_matrix=support_matrix_projected,
        similarity_metric=similarity_metric,
    )
    neighbor_indices = top_neighbor_indices(similarity_scores=similarity_scores, top_m=top_m)

    neighbors_df = build_neighbors_dataframe(
        cold_items_df=cold_items_df,
        support_items_df=support_items_df,
        similarity_scores=similarity_scores,
        neighbor_indices=neighbor_indices,
        cold_item_col=cold_item_col,
        support_item_col=support_item_col,
    )

    return NeighborSearchResult(
        neighbors_df=neighbors_df,
        num_cold_items=int(len(cold_items_df)),
        num_support_items=int(len(support_items_df)),
        neighbors_per_item=int(neighbor_indices.shape[1]) if neighbor_indices.ndim == 2 else 0,
        similarity_metric=similarity_metric,
    )


@dataclass(slots=True)
class ItemSimilarityIndex:
    """
    Reusable nearest-neighbor search component for cold-item support mapping.
    """

    cold_item_col: str = "item_id"
    support_item_col: str = "item_id"
    similarity_metric: str = "cosine"
    max_projection_dim: int = 64
    random_state: int = 42

    def find_neighbors(
        self,
        cold_items_df: pd.DataFrame,
        cold_feature_matrix: sparse.spmatrix | np.ndarray,
        support_items_df: pd.DataFrame,
        support_feature_matrix: sparse.spmatrix | np.ndarray,
        top_m: int,
    ) -> NeighborSearchResult:
        """
        Find nearest support neighbors for each cold item.
        """
        return find_item_neighbors(
            cold_items_df=cold_items_df,
            cold_feature_matrix=cold_feature_matrix,
            support_items_df=support_items_df,
            support_feature_matrix=support_feature_matrix,
            top_m=top_m,
            cold_item_col=self.cold_item_col,
            support_item_col=self.support_item_col,
            similarity_metric=self.similarity_metric,
            max_projection_dim=self.max_projection_dim,
            random_state=self.random_state,
        )
