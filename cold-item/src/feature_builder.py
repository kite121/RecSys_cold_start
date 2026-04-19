from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse

from src.als_model import ALSRecommender
from src.preprocessing import RecommendationDataPreprocessor
from src.split_warm_cold import WarmColdSplitResult


@dataclass
class RegressorTrainingDataset:
    """
    Training dataset for CatBoostRegressor.

    The target is the ALS score computed on warm user-item pairs.
    """

    pairs_df: pd.DataFrame
    X: sparse.spmatrix
    y: pd.Series


def sample_warm_negative_items(
    user_id: str,
    warm_item_ids: list[str],
    seen_items_by_user: dict[str, set[str]],
    num_samples: int,
    rng: np.random.Generator,
) -> list[str]:
    """
    Sample warm items that the user has not interacted with.
    """
    seen_items = seen_items_by_user.get(user_id, set())
    available_items = [item_id for item_id in warm_item_ids if item_id not in seen_items]
    if not available_items or num_samples <= 0:
        return []

    sample_size = min(num_samples, len(available_items))
    return rng.choice(np.array(available_items), size=sample_size, replace=False).tolist()


def build_regressor_training_dataset(
    interactions_df: pd.DataFrame,
    user_features_df: pd.DataFrame,
    item_features_df: pd.DataFrame,
    warm_cold_split: WarmColdSplitResult,
    preprocessor: RecommendationDataPreprocessor,
    als_model: ALSRecommender,
    negative_samples_per_user: int = 3,
    random_state: int = 42,
    user_col: str = "user_id",
    item_col: str = "item_id",
) -> RegressorTrainingDataset:
    """
    Build the training dataset for CatBoostRegressor.

    Logic:
    - keep only warm items
    - use observed warm pairs as positive examples
    - optionally add unseen warm pairs as negative examples
    - compute ALS score for each pair as the regression target
    """
    rng = np.random.default_rng(random_state)

    warm_interactions_df = interactions_df[interactions_df[item_col].isin(warm_cold_split.warm_items)].copy()
    warm_interactions_df = warm_interactions_df[[user_col, item_col]].drop_duplicates()

    seen_items_by_user = (
        interactions_df.groupby(user_col)[item_col]
        .agg(lambda values: set(pd.unique(values.astype(str))))
        .to_dict()
    )
    warm_item_ids = warm_interactions_df[item_col].drop_duplicates().tolist()

    rows: list[dict[str, str]] = []
    for current_user_id, user_rows_df in warm_interactions_df.groupby(user_col):
        positive_items = user_rows_df[item_col].tolist()
        for current_item_id in positive_items:
            rows.append(
                {
                    user_col: current_user_id,
                    item_col: current_item_id,
                    "sample_type": "positive",
                }
            )

        negative_items = sample_warm_negative_items(
            user_id=str(current_user_id),
            warm_item_ids=warm_item_ids,
            seen_items_by_user=seen_items_by_user,
            num_samples=max(1, len(positive_items) * negative_samples_per_user),
            rng=rng,
        )
        for current_item_id in negative_items:
            rows.append(
                {
                    user_col: current_user_id,
                    item_col: current_item_id,
                    "sample_type": "negative",
                }
            )

    if not rows:
        raise ValueError("No warm training pairs were generated for the CatBoost regressor.")

    pairs_df = pd.DataFrame(rows)
    merged_pairs_df, X = preprocessor.transform_pairs(
        pairs_df=pairs_df[[user_col, item_col]],
        user_features_df=user_features_df,
        item_features_df=item_features_df,
    )
    merged_pairs_df["sample_type"] = pairs_df["sample_type"].to_numpy()
    merged_pairs_df["target"] = als_model.score_pairs(merged_pairs_df[[user_col, item_col]])

    return RegressorTrainingDataset(
        pairs_df=merged_pairs_df,
        X=X,
        y=merged_pairs_df["target"],
    )

