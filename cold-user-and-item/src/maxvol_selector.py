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


def ensure_item_id_column(df: pd.DataFrame, item_id_col: str) -> None:
    """
    Validate that the item feature table contains the identifier column.
    """
    if item_id_col not in df.columns:
        raise KeyError(f"Missing required item id column: {item_id_col}")


def ensure_matching_rows(items_df: pd.DataFrame, feature_matrix: sparse.spmatrix | np.ndarray) -> None:
    """
    Validate that the item table and its feature matrix have the same row count.
    """
    if len(items_df) != feature_matrix.shape[0]:
        raise ValueError(
            "items_df and feature_matrix must contain the same number of rows. "
            f"Got len(items_df)={len(items_df)} and feature_matrix.shape[0]={feature_matrix.shape[0]}."
        )


def build_candidate_items_df(
    candidate_item_ids: list[str],
    item_features: pd.DataFrame,
    item_id_col: str = "item_id",
) -> pd.DataFrame:
    """
    Build an item feature table aligned with the requested candidate order.
    """
    ensure_item_id_column(item_features, item_id_col)
    if not candidate_item_ids:
        return pd.DataFrame(columns=item_features.columns)

    normalized_item_features = item_features.copy()
    normalized_item_features[item_id_col] = normalized_item_features[item_id_col].astype(str)
    normalized_item_features = normalized_item_features.drop_duplicates(subset=[item_id_col], keep="last")

    requested_df = pd.DataFrame({item_id_col: [str(item_id) for item_id in candidate_item_ids]})
    candidate_items_df = requested_df.merge(normalized_item_features, on=item_id_col, how="left")
    candidate_items_df = candidate_items_df.dropna(
        how="all",
        subset=[column for column in candidate_items_df.columns if column != item_id_col],
    )
    return candidate_items_df.reset_index(drop=True)


def build_item_feature_matrix(
    items_df: pd.DataFrame,
    item_id_col: str = "item_id",
) -> sparse.csr_matrix:
    """
    Convert an item feature table into a numeric matrix for maxvol selection.
    """
    ensure_item_id_column(items_df, item_id_col)
    if items_df.empty:
        return sparse.csr_matrix((0, 0), dtype=np.float64)

    feature_df = items_df.drop(columns=[item_id_col]).copy()
    if feature_df.empty:
        return sparse.csr_matrix((len(items_df), 0), dtype=np.float64)

    numeric_df = feature_df.select_dtypes(include=[np.number]).astype(float, copy=False)
    categorical_columns = [column for column in feature_df.columns if column not in numeric_df.columns]
    categorical_df = (
        pd.get_dummies(
            feature_df[categorical_columns].fillna("__missing__").astype(str),
            dummy_na=False,
        )
        if categorical_columns
        else pd.DataFrame(index=feature_df.index)
    )

    if numeric_df.empty and categorical_df.empty:
        return sparse.csr_matrix((len(items_df), 0), dtype=np.float64)

    combined_df = pd.concat([numeric_df, categorical_df], axis=1).fillna(0.0)
    return sparse.csr_matrix(combined_df.to_numpy(dtype=np.float64))


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
    """
    if len(selected_indices) >= target_size:
        return selected_indices[:target_size]

    num_rows = normalized_matrix.shape[0]
    selected_mask = np.zeros(num_rows, dtype=bool)
    if selected_indices:
        selected_mask[np.asarray(selected_indices, dtype=int)] = True

    if not selected_indices and num_rows > 0:
        first_index = int(np.argmax(np.linalg.norm(normalized_matrix, axis=1)))
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
        return greedy_diversity_fill(normalized_matrix, selected_indices, top_k), "qr_pivoting_plus_greedy_fill"
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
    Low-level maxvol selection on an already prepared item table and feature matrix.
    """
    ensure_item_id_column(items_df, item_id_col)
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
    Fitted maxvol selector for cold-user and global cold-start candidate diversification.
    """

    item_id_col: str = "item_id"
    max_projection_dim: int = 64
    random_state: int = 42

    selection_result: DiverseSelectionResult | None = None
    fitted_candidate_item_ids: list[str] | None = None
    fitted_k: int | None = None

    def fit(
        self,
        candidate_item_ids: list[str],
        item_features: pd.DataFrame,
        k: int,
    ) -> "MaxVolSelector":
        """
        Build and cache the diversified candidate pool.

        If item features are missing, unusable, or maxvol fails, the selector
        falls back to the first K candidate items instead of raising.
        """
        normalized_candidate_item_ids = [str(item_id) for item_id in candidate_item_ids]
        if k <= 0 or not normalized_candidate_item_ids:
            self.selection_result = DiverseSelectionResult(
                selected_indices=[],
                selected_item_ids=[],
                selected_items_df=pd.DataFrame(columns=[self.item_id_col]),
                input_size=int(len(normalized_candidate_item_ids)),
                output_size=0,
                strategy="empty_selection",
            )
            self.fitted_candidate_item_ids = list(normalized_candidate_item_ids)
            self.fitted_k = int(k)
            return self

        fallback_item_ids = normalized_candidate_item_ids[: min(k, len(normalized_candidate_item_ids))]
        fallback_df = pd.DataFrame({self.item_id_col: fallback_item_ids})

        try:
            candidate_items_df = build_candidate_items_df(
                candidate_item_ids=normalized_candidate_item_ids,
                item_features=item_features,
                item_id_col=self.item_id_col,
            )
            if candidate_items_df.empty:
                self.selection_result = DiverseSelectionResult(
                    selected_indices=list(range(len(fallback_item_ids))),
                    selected_item_ids=fallback_item_ids,
                    selected_items_df=fallback_df,
                    input_size=int(len(normalized_candidate_item_ids)),
                    output_size=int(len(fallback_item_ids)),
                    strategy="fallback_empty_item_features",
                )
                self.fitted_candidate_item_ids = list(normalized_candidate_item_ids)
                self.fitted_k = int(k)
                return self

            feature_matrix = build_item_feature_matrix(
                items_df=candidate_items_df,
                item_id_col=self.item_id_col,
            )
            if feature_matrix.shape[1] == 0:
                selected_items_df = candidate_items_df.head(k).copy().reset_index(drop=True)
                selected_items_df["diversity_rank"] = np.arange(1, len(selected_items_df) + 1)
                self.selection_result = DiverseSelectionResult(
                    selected_indices=list(range(len(selected_items_df))),
                    selected_item_ids=selected_items_df[self.item_id_col].astype(str).tolist(),
                    selected_items_df=selected_items_df,
                    input_size=int(len(candidate_items_df)),
                    output_size=int(len(selected_items_df)),
                    strategy="fallback_no_item_features",
                )
                self.fitted_candidate_item_ids = list(normalized_candidate_item_ids)
                self.fitted_k = int(k)
                return self

            self.selection_result = select_diverse_items(
                items_df=candidate_items_df,
                feature_matrix=feature_matrix,
                top_k=k,
                item_id_col=self.item_id_col,
                max_projection_dim=self.max_projection_dim,
                random_state=self.random_state,
            )
            self.fitted_candidate_item_ids = list(normalized_candidate_item_ids)
            self.fitted_k = int(k)
            return self
        except Exception:
            self.selection_result = DiverseSelectionResult(
                selected_indices=list(range(len(fallback_item_ids))),
                selected_item_ids=fallback_item_ids,
                selected_items_df=fallback_df,
                input_size=int(len(normalized_candidate_item_ids)),
                output_size=int(len(fallback_item_ids)),
                strategy="fallback_exception_passthrough",
            )
            self.fitted_candidate_item_ids = list(normalized_candidate_item_ids)
            self.fitted_k = int(k)
            return self

    def select(self, k: int | None = None) -> list[str]:
        """
        Return the cached diversified item ids.
        """
        if self.selection_result is None:
            raise RuntimeError("MaxVolSelector must be fitted before calling select.")
        if k is None:
            return list(self.selection_result.selected_item_ids)
        return list(self.selection_result.selected_item_ids[: max(k, 0)])

    def get_result(self) -> DiverseSelectionResult:
        """
        Return the full cached selection result.
        """
        if self.selection_result is None:
            raise RuntimeError("MaxVolSelector must be fitted before calling get_result.")
        return self.selection_result
