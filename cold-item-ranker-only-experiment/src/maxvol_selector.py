from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.linalg import qr
from sklearn.decomposition import TruncatedSVD


@dataclass(slots=True)
class DiverseSelectionResult:
    """
    Result of support-set diversification.
    """

    selected_indices: list[int]
    selected_item_ids: list[str]
    selected_items_df: pd.DataFrame
    input_size: int
    output_size: int
    strategy: str


def ensure_matching_rows(items_df: pd.DataFrame, feature_matrix: sparse.spmatrix | np.ndarray) -> None:
    """
    Validate that the item table and its feature matrix have the same row count.
    """
    if len(items_df) != feature_matrix.shape[0]:
        raise ValueError(
            "items_df and feature_matrix must contain the same number of rows. "
            f"Got len(items_df)={len(items_df)} and feature_matrix.shape[0]={feature_matrix.shape[0]}."
        )


def row_normalize_dense(matrix: np.ndarray) -> np.ndarray:
    """
    L2-normalize dense row vectors, keeping zero rows unchanged.
    """
    normalized = np.asarray(matrix, dtype=np.float64).copy()
    norms = np.linalg.norm(normalized, axis=1, keepdims=True)
    non_zero_rows = norms.squeeze(axis=1) > 0
    normalized[non_zero_rows] /= norms[non_zero_rows]
    return normalized


def project_feature_matrix(
    feature_matrix: sparse.spmatrix | np.ndarray,
    max_projection_dim: int,
    random_state: int,
) -> np.ndarray:
    """
    Project an item-feature matrix to a compact dense representation.

    The selector operates on a dense matrix because QR pivoting is used as a
    practical approximation of max-volume row selection. Sparse high-dimensional
    matrices are first compressed with TruncatedSVD.
    """
    num_rows, num_cols = feature_matrix.shape
    if num_rows == 0:
        return np.empty((0, 0), dtype=np.float64)
    if num_cols == 0:
        return np.zeros((num_rows, 1), dtype=np.float64)

    if sparse.issparse(feature_matrix):
        target_dim = min(max_projection_dim, num_rows - 1, num_cols - 1)
        if target_dim >= 1:
            projector = TruncatedSVD(n_components=target_dim, random_state=random_state)
            return np.asarray(projector.fit_transform(feature_matrix), dtype=np.float64)
        return np.asarray(feature_matrix.toarray(), dtype=np.float64)

    dense_matrix = np.asarray(feature_matrix, dtype=np.float64)
    target_dim = min(max_projection_dim, dense_matrix.shape[0] - 1, dense_matrix.shape[1] - 1)
    if target_dim >= 1 and dense_matrix.shape[1] > max_projection_dim:
        projector = TruncatedSVD(n_components=target_dim, random_state=random_state)
        return np.asarray(projector.fit_transform(dense_matrix), dtype=np.float64)
    return dense_matrix


def greedy_diversity_fill(
    normalized_matrix: np.ndarray,
    selected_indices: list[int],
    target_size: int,
) -> list[int]:
    """
    Fill the remainder of the support set with a greedy farthest-point heuristic.

    This is used when QR pivoting returns fewer unique useful pivots than needed,
    which can happen with low-rank or degenerate feature matrices.
    """
    if len(selected_indices) >= target_size:
        return selected_indices[:target_size]

    num_rows = normalized_matrix.shape[0]
    selected_mask = np.zeros(num_rows, dtype=bool)
    if selected_indices:
        selected_mask[np.asarray(selected_indices, dtype=int)] = True

    if not selected_indices and num_rows > 0:
        row_norms = np.linalg.norm(normalized_matrix, axis=1)
        first_index = int(np.argmax(row_norms))
        selected_indices.append(first_index)
        selected_mask[first_index] = True

    while len(selected_indices) < min(target_size, num_rows):
        remaining = np.flatnonzero(~selected_mask)
        if len(remaining) == 0:
            break

        selected_vectors = normalized_matrix[np.asarray(selected_indices, dtype=int)]
        similarities = normalized_matrix[remaining] @ selected_vectors.T
        max_similarity = similarities.max(axis=1) if similarities.size else np.zeros(len(remaining), dtype=float)
        next_index = int(remaining[np.argmin(max_similarity)])
        selected_indices.append(next_index)
        selected_mask[next_index] = True

    return selected_indices[:target_size]


def select_maxvol_indices(
    feature_matrix: sparse.spmatrix | np.ndarray,
    top_k: int,
    max_projection_dim: int = 64,
    random_state: int = 42,
) -> tuple[list[int], str]:
    """
    Select diverse row indices with a practical maxvol-style approximation.

    Steps:
    1. Project the feature matrix to a compact dense representation.
    2. Normalize rows.
    3. Run QR pivoting on the transposed matrix to prioritize diverse rows.
    4. Greedily fill the remainder if needed.
    """
    num_rows = feature_matrix.shape[0]
    if top_k <= 0 or num_rows == 0:
        return [], "empty_selection"
    if num_rows <= top_k:
        return list(range(num_rows)), "passthrough_all_items"

    projected_matrix = project_feature_matrix(
        feature_matrix=feature_matrix,
        max_projection_dim=max_projection_dim,
        random_state=random_state,
    )
    normalized_matrix = row_normalize_dense(projected_matrix)

    if normalized_matrix.shape[1] == 0:
        return list(range(top_k)), "passthrough_zero_features"

    _, _, pivots = qr(normalized_matrix.T, pivoting=True, mode="economic")
    selected_indices = [int(index) for index in pivots[: min(top_k, len(pivots))]]

    if len(selected_indices) < top_k:
        selected_indices = greedy_diversity_fill(
            normalized_matrix=normalized_matrix,
            selected_indices=selected_indices,
            target_size=top_k,
        )
        return selected_indices, "qr_pivoting_plus_greedy_fill"

    return selected_indices[:top_k], "qr_pivoting_maxvol"


def select_diverse_items(
    items_df: pd.DataFrame,
    feature_matrix: sparse.spmatrix | np.ndarray,
    top_k: int,
    item_id_col: str = "item_id",
    max_projection_dim: int = 64,
    random_state: int = 42,
) -> DiverseSelectionResult:
    """
    Select a diverse subset of items from a support pool.

    Args:
        items_df: Candidate support items, usually top-N popular items.
        feature_matrix: Feature matrix aligned row-wise with ``items_df``.
        top_k: Target number of diverse items to keep.
        item_id_col: Item identifier column.
        max_projection_dim: Size of the low-dimensional dense projection used
            for maxvol-style selection.
        random_state: Random seed for deterministic projection behavior.
    """
    if item_id_col not in items_df.columns:
        raise KeyError(f"Missing required item id column: {item_id_col}")

    ensure_matching_rows(items_df, feature_matrix)
    selected_indices, strategy = select_maxvol_indices(
        feature_matrix=feature_matrix,
        top_k=top_k,
        max_projection_dim=max_projection_dim,
        random_state=random_state,
    )

    selected_items_df = items_df.iloc[selected_indices].copy().reset_index(drop=True)
    selected_items_df["diversity_rank"] = np.arange(1, len(selected_items_df) + 1)

    return DiverseSelectionResult(
        selected_indices=selected_indices,
        selected_item_ids=selected_items_df[item_id_col].astype(str).tolist(),
        selected_items_df=selected_items_df,
        input_size=int(len(items_df)),
        output_size=int(len(selected_items_df)),
        strategy=strategy,
    )


@dataclass(slots=True)
class MaxVolSelector:
    """
    Reusable selector for building a diverse support set for cold-item handling.
    """

    item_id_col: str = "item_id"
    max_projection_dim: int = 64
    random_state: int = 42

    def select(
        self,
        items_df: pd.DataFrame,
        feature_matrix: sparse.spmatrix | np.ndarray,
        top_k: int,
    ) -> DiverseSelectionResult:
        """
        Select top-K diverse items from the input pool.
        """
        return select_diverse_items(
            items_df=items_df,
            feature_matrix=feature_matrix,
            top_k=top_k,
            item_id_col=self.item_id_col,
            max_projection_dim=self.max_projection_dim,
            random_state=self.random_state,
        )
