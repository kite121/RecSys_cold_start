from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse

from src.als_model import ALSRecommender
from src.preprocessing import RecommendationDataPreprocessor
from src.split_warm_cold import WarmColdSplitResult
from src.split_warm_cold_users import WarmColdUserSplitResult


@dataclass
class RegressorTrainingDataset:
    """
    Training dataset for CatBoostRegressor.

    The target is the ALS score computed on warm user-item pairs.
    """

    pairs_df: pd.DataFrame
    X: sparse.spmatrix
    y: pd.Series


def sample_negative_pairs(
    interactions: pd.DataFrame,
    users: list[str] | set[str],
    items: list[str] | set[str],
    user_col: str = "user_id",
    item_col: str = "item_id",
    negatives_per_positive: int = 3,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Sample negative user-item pairs inside the provided warm user/item space.
    """
    if negatives_per_positive <= 0:
        return pd.DataFrame(columns=[user_col, item_col, "sample_type"])

    rng = np.random.default_rng(random_state)
    normalized_users = [str(user_id) for user_id in users]
    normalized_items = [str(item_id) for item_id in items]
    if not normalized_users or not normalized_items:
        return pd.DataFrame(columns=[user_col, item_col, "sample_type"])

    observed_pairs = interactions[[user_col, item_col]].copy()
    observed_pairs[user_col] = observed_pairs[user_col].astype(str)
    observed_pairs[item_col] = observed_pairs[item_col].astype(str)
    observed_pairs = observed_pairs[
        observed_pairs[user_col].isin(normalized_users) & observed_pairs[item_col].isin(normalized_items)
    ]

    seen_items_by_user = (
        observed_pairs.groupby(user_col)[item_col]
        .agg(lambda values: set(pd.unique(values.astype(str))))
        .to_dict()
    )

    rows: list[dict[str, str]] = []
    for current_user_id, user_rows_df in observed_pairs.groupby(user_col):
        positive_count = int(len(user_rows_df.drop_duplicates(subset=[item_col])))
        num_samples = max(1, positive_count * negatives_per_positive)
        negative_items = sample_warm_negative_items(
            user_id=str(current_user_id),
            warm_item_ids=normalized_items,
            seen_items_by_user=seen_items_by_user,
            num_samples=num_samples,
            rng=rng,
        )
        rows.extend(
            {
                user_col: str(current_user_id),
                item_col: str(current_item_id),
                "sample_type": "negative",
            }
            for current_item_id in negative_items
        )

    return pd.DataFrame(rows)


def build_train_pairs(
    warm_interactions_df: pd.DataFrame,
    warm_user_ids: list[str] | set[str],
    warm_item_ids: list[str] | set[str],
    user_col: str = "user_id",
    item_col: str = "item_id",
    negatives_per_positive: int = 3,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Build positive and negative train pairs for the CatBoost regressor.
    """
    positive_pairs = (
        warm_interactions_df[[user_col, item_col]]
        .drop_duplicates()
        .assign(sample_type="positive")
        .reset_index(drop=True)
    )
    negative_pairs = sample_negative_pairs(
        interactions=warm_interactions_df,
        users=warm_user_ids,
        items=warm_item_ids,
        user_col=user_col,
        item_col=item_col,
        negatives_per_positive=negatives_per_positive,
        random_state=random_state,
    )
    if negative_pairs.empty:
        return positive_pairs
    return pd.concat([positive_pairs, negative_pairs], ignore_index=True)


def build_train_features(
    preprocessor: RecommendationDataPreprocessor,
    pairs: pd.DataFrame,
    user_features_df: pd.DataFrame,
    item_features_df: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "item_id",
) -> tuple[pd.DataFrame, sparse.spmatrix]:
    """
    Build train-time feature matrix for user-item pairs.
    """
    return preprocessor.transform_pairs(
        pairs_df=pairs[[user_col, item_col]],
        user_features_df=user_features_df,
        item_features_df=item_features_df,
    )


def build_target_user_feature_table(
    user_id: str,
    preprocessor: RecommendationDataPreprocessor,
    user_features_df: pd.DataFrame,
    user_features: dict[str, object] | pd.Series | None = None,
) -> pd.DataFrame:
    """
    Build a one-row user feature table for inference.
    """
    user_feature_cols = preprocessor.feature_roles.user_feature_cols if preprocessor.feature_roles is not None else []
    target_columns = [preprocessor.user_id_col] + user_feature_cols

    existing_rows = user_features_df[user_features_df[preprocessor.user_id_col].astype(str) == str(user_id)]
    if existing_rows.empty:
        target_row = {column: np.nan for column in target_columns}
        target_row[preprocessor.user_id_col] = str(user_id)
    else:
        target_row = existing_rows.iloc[-1][target_columns].to_dict()
        target_row[preprocessor.user_id_col] = str(user_id)

    if user_features is not None:
        user_features_dict = user_features.to_dict() if isinstance(user_features, pd.Series) else dict(user_features)
        for feature_name, feature_value in user_features_dict.items():
            if feature_name in target_columns and feature_name != preprocessor.user_id_col:
                target_row[feature_name] = feature_value

    target_user_df = pd.DataFrame([target_row], columns=target_columns)
    return target_user_df.replace({pd.NA: np.nan})


def build_inference_features(
    preprocessor: RecommendationDataPreprocessor,
    user_id: str,
    item_ids: list[str],
    user_features_df: pd.DataFrame,
    item_features_df: pd.DataFrame,
    user_features: dict[str, object] | pd.Series | None = None,
) -> tuple[pd.DataFrame, sparse.spmatrix]:
    """
    Build inference-time feature matrix for a single user and candidate items.
    """
    pairs_df = pd.DataFrame(
        {
            preprocessor.user_id_col: [str(user_id)] * len(item_ids),
            preprocessor.item_id_col: [str(item_id) for item_id in item_ids],
        }
    )
    target_user_features_df = build_target_user_feature_table(
        user_id=user_id,
        preprocessor=preprocessor,
        user_features_df=user_features_df,
        user_features=user_features,
    )
    return preprocessor.transform_pairs(
        pairs_df=pairs_df,
        user_features_df=target_user_features_df,
        item_features_df=item_features_df,
    )


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
    warm_cold_user_split: WarmColdUserSplitResult | None,
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
    - keep only warm users and warm items
    - use observed warm pairs as positive examples
    - optionally add unseen warm pairs as negative examples
    - compute ALS score for each pair as the regression target
    """
    warm_user_ids = (
        set(map(str, warm_cold_user_split.warm_users))
        if warm_cold_user_split is not None
        else set(pd.unique(interactions_df[user_col].astype(str)))
    )
    warm_item_ids = set(map(str, warm_cold_split.warm_items))

    warm_interactions_df = interactions_df.copy()
    warm_interactions_df[user_col] = warm_interactions_df[user_col].astype(str)
    warm_interactions_df[item_col] = warm_interactions_df[item_col].astype(str)
    warm_interactions_df = warm_interactions_df[
        warm_interactions_df[user_col].isin(warm_user_ids)
        & warm_interactions_df[item_col].isin(warm_item_ids)
    ].copy()

    train_pairs_df = build_train_pairs(
        warm_interactions_df=warm_interactions_df,
        warm_user_ids=warm_user_ids,
        warm_item_ids=warm_item_ids,
        user_col=user_col,
        item_col=item_col,
        negatives_per_positive=negative_samples_per_user,
        random_state=random_state,
    )
    if train_pairs_df.empty:
        raise ValueError("No warm training pairs were generated for the CatBoost regressor.")

    merged_pairs_df, X = build_train_features(
        preprocessor=preprocessor,
        pairs=train_pairs_df,
        user_features_df=user_features_df,
        item_features_df=item_features_df,
        user_col=user_col,
        item_col=item_col,
    )
    merged_pairs_df["sample_type"] = train_pairs_df["sample_type"].to_numpy()
    merged_pairs_df["target"] = als_model.score_pairs(merged_pairs_df[[user_col, item_col]])

    return RegressorTrainingDataset(
        pairs_df=merged_pairs_df,
        X=X,
        y=merged_pairs_df["target"],
    )
