from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.als_model import ALSRecommender


@dataclass(slots=True)
class ColdVectorBuildResult:
    """
    Result of synthetic latent vector construction for cold items.
    """

    cold_vectors_df: pd.DataFrame
    cold_vector_map: dict[str, np.ndarray]
    num_cold_items: int
    num_built_vectors: int
    num_missing_vectors: int
    weighting_strategy: str


def ensure_neighbor_columns_present(neighbors_df: pd.DataFrame, required_columns: list[str]) -> None:
    """
    Validate that the neighbor table contains all required columns.
    """
    missing_columns = [column for column in required_columns if column not in neighbors_df.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns in neighbors_df: {missing_columns}")


def extract_item_factor_map(als_model: ALSRecommender) -> dict[str, np.ndarray]:
    """
    Extract warm item latent vectors from a fitted ALS model.
    """
    if als_model.model is None or als_model.artifacts is None:
        raise RuntimeError("ALS model must be fitted before extracting item factors.")

    item_factor_map: dict[str, np.ndarray] = {}
    for item_id, item_index in als_model.artifacts.item2idx.items():
        item_factor_map[str(item_id)] = np.asarray(als_model.model.item_factors[item_index], dtype=np.float32).copy()
    return item_factor_map


def compute_neighbor_weights(
    neighbor_group_df: pd.DataFrame,
    similarity_col: str,
    weighting_strategy: str,
) -> np.ndarray:
    """
    Compute aggregation weights for one cold item's support neighbors.
    """
    if weighting_strategy == "uniform":
        return np.full(len(neighbor_group_df), 1.0 / len(neighbor_group_df), dtype=np.float64)

    if weighting_strategy != "similarity":
        raise ValueError("weighting_strategy must be either 'uniform' or 'similarity'.")

    raw_weights = neighbor_group_df[similarity_col].astype(float).to_numpy(dtype=np.float64)
    min_weight = float(raw_weights.min()) if raw_weights.size else 0.0
    if min_weight < 0.0:
        raw_weights = raw_weights - min_weight

    weight_sum = float(raw_weights.sum())
    if weight_sum <= 0.0:
        return np.full(len(neighbor_group_df), 1.0 / len(neighbor_group_df), dtype=np.float64)

    return raw_weights / weight_sum


def aggregate_neighbor_vectors(
    neighbor_group_df: pd.DataFrame,
    item_factor_map: dict[str, np.ndarray],
    neighbor_item_col: str,
    similarity_col: str,
    weighting_strategy: str,
) -> tuple[np.ndarray | None, int]:
    """
    Aggregate warm neighbor vectors into one synthetic cold-item vector.
    """
    valid_vectors: list[np.ndarray] = []
    valid_rows: list[int] = []

    for row_index, neighbor_item_id in enumerate(neighbor_group_df[neighbor_item_col].astype(str).tolist()):
        item_vector = item_factor_map.get(neighbor_item_id)
        if item_vector is not None:
            valid_vectors.append(item_vector.astype(np.float64, copy=False))
            valid_rows.append(row_index)

    if not valid_vectors:
        return None, 0

    filtered_group_df = neighbor_group_df.iloc[valid_rows].reset_index(drop=True)
    weights = compute_neighbor_weights(
        neighbor_group_df=filtered_group_df,
        similarity_col=similarity_col,
        weighting_strategy=weighting_strategy,
    )

    stacked_vectors = np.vstack(valid_vectors)
    synthetic_vector = np.average(stacked_vectors, axis=0, weights=weights)
    return synthetic_vector.astype(np.float32), len(valid_vectors)


def build_cold_item_vectors(
    neighbors_df: pd.DataFrame,
    als_model: ALSRecommender,
    cold_item_col: str = "item_id",
    neighbor_item_col: str = "neighbor_item_id",
    similarity_col: str = "similarity",
    weighting_strategy: str = "similarity",
) -> ColdVectorBuildResult:
    """
    Build synthetic latent vectors for cold items from warm neighbor vectors.
    """
    ensure_neighbor_columns_present(neighbors_df, [cold_item_col, neighbor_item_col, similarity_col])
    item_factor_map = extract_item_factor_map(als_model)

    rows: list[dict[str, object]] = []
    cold_vector_map: dict[str, np.ndarray] = {}

    unique_cold_item_ids = neighbors_df[cold_item_col].astype(str).drop_duplicates().tolist()
    for cold_item_id, neighbor_group_df in neighbors_df.groupby(cold_item_col, sort=False):
        normalized_cold_item_id = str(cold_item_id)
        synthetic_vector, num_neighbors_used = aggregate_neighbor_vectors(
            neighbor_group_df=neighbor_group_df,
            item_factor_map=item_factor_map,
            neighbor_item_col=neighbor_item_col,
            similarity_col=similarity_col,
            weighting_strategy=weighting_strategy,
        )

        if synthetic_vector is None:
            rows.append(
                {
                    cold_item_col: normalized_cold_item_id,
                    "vector": None,
                    "num_neighbors_used": 0,
                    "vector_dim": 0,
                    "build_status": "missing_neighbors_in_als_space",
                }
            )
            continue

        cold_vector_map[normalized_cold_item_id] = synthetic_vector
        rows.append(
            {
                    cold_item_col: normalized_cold_item_id,
                    "vector": synthetic_vector,
                    "num_neighbors_used": int(num_neighbors_used),
                    "vector_dim": int(synthetic_vector.shape[0]),
                    "build_status": "ok",
                }
            )

    cold_vectors_df = pd.DataFrame(rows)
    if cold_vectors_df.empty:
        cold_vectors_df = pd.DataFrame(
            columns=[cold_item_col, "vector", "num_neighbors_used", "vector_dim", "build_status"]
        )

    num_built_vectors = int((cold_vectors_df["build_status"] == "ok").sum()) if not cold_vectors_df.empty else 0
    num_cold_items = len(unique_cold_item_ids)

    return ColdVectorBuildResult(
        cold_vectors_df=cold_vectors_df,
        cold_vector_map=cold_vector_map,
        num_cold_items=num_cold_items,
        num_built_vectors=num_built_vectors,
        num_missing_vectors=int(num_cold_items - num_built_vectors),
        weighting_strategy=weighting_strategy,
    )


@dataclass(slots=True)
class ColdItemVectorBuilder:
    """
    Reusable synthetic-vector builder for cold items.
    """

    cold_item_col: str = "item_id"
    neighbor_item_col: str = "neighbor_item_id"
    similarity_col: str = "similarity"
    weighting_strategy: str = "similarity"

    def build(self, neighbors_df: pd.DataFrame, als_model: ALSRecommender) -> ColdVectorBuildResult:
        """
        Build synthetic latent vectors for cold items.
        """
        return build_cold_item_vectors(
            neighbors_df=neighbors_df,
            als_model=als_model,
            cold_item_col=self.cold_item_col,
            neighbor_item_col=self.neighbor_item_col,
            similarity_col=self.similarity_col,
            weighting_strategy=self.weighting_strategy,
        )
