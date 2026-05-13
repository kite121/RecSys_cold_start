from __future__ import annotations

from pathlib import Path

from src.als_model import ALSRecommender
from src.data_loader import load_training_csv_data
from src.hybrid_recommender import HybridRecommender
from src.preprocessing import RecommendationDataPreprocessor
from src.ranker_model import CatBoostRegressorModel
from src.split_warm_cold import WarmColdItemSplitter


def ensure_parent_dir(path: str | Path) -> Path:
    """
    Ensure that the parent directory for an output file exists.
    """
    normalized_path = Path(path)
    normalized_path.parent.mkdir(parents=True, exist_ok=True)
    return normalized_path


def train_hybrid_model(
    train_csv_path: str | Path,
    model_output_path: str | Path | None = None,
    user_id_col: str = "user_id",
    item_id_col: str = "item_id",
    value_col: str = "value",
    user_prefix: str = "user_",
    item_prefix: str = "item_",
    min_warm_interactions: int = 5,
    popularity_metric: str = "count",
    als_factors: int = 20,
    als_regularization: float = 0.01,
    als_iterations: int = 150,
    als_alpha: float = 20.0,
    als_random_state: int = 42,
    regressor_iterations: int = 300,
    regressor_learning_rate: float = 0.05,
    regressor_depth: int = 5,
    regressor_loss_function: str = "RMSE",
    regressor_random_seed: int = 42,
    negative_samples_per_user: int = 3,
) -> tuple[HybridRecommender, dict]:
    """
    Train the hybrid recommender from a training CSV.

    Pipeline steps:
    1. load and validate CSV
    2. fit preprocessing
    3. split items into warm/cold
    4. train ALS
    5. train CatBoostRegressor to approximate ALS scores
    6. optionally save the fitted HybridRecommender
    """
    train_df, column_groups = load_training_csv_data(
        csv_path=train_csv_path,
        user_id_col=user_id_col,
        item_id_col=item_id_col,
        value_col=value_col,
        user_prefix=user_prefix,
        item_prefix=item_prefix,
    )

    preprocessor = RecommendationDataPreprocessor(
        user_id_col=user_id_col,
        item_id_col=item_id_col,
        value_col=value_col,
        user_prefix=user_prefix,
        item_prefix=item_prefix,
    )
    warm_cold_splitter = WarmColdItemSplitter(
        min_warm_interactions=min_warm_interactions,
        popularity_metric=popularity_metric,
        item_col=item_id_col,
        value_col=value_col,
    )
    als_model = ALSRecommender(
        user_col=user_id_col,
        item_col=item_id_col,
        value_col=value_col,
        factors=als_factors,
        regularization=als_regularization,
        iterations=als_iterations,
        alpha=als_alpha,
        random_state=als_random_state,
    )
    regressor_model = CatBoostRegressorModel(
        iterations=regressor_iterations,
        learning_rate=regressor_learning_rate,
        depth=regressor_depth,
        loss_function=regressor_loss_function,
        random_seed=regressor_random_seed,
    )

    hybrid_model = HybridRecommender(
        preprocessor=preprocessor,
        warm_cold_splitter=warm_cold_splitter,
        als_model=als_model,
        regressor_model=regressor_model,
        negative_samples_per_user=negative_samples_per_user,
        random_state=als_random_state,
    )
    hybrid_model.fit(train_df)

    if model_output_path is not None:
        hybrid_model.save(ensure_parent_dir(model_output_path))

    interactions_df = hybrid_model.preprocessing_artifacts.interactions_df
    training_summary = {
        "train_csv_path": str(train_csv_path),
        "num_rows": int(len(interactions_df)),
        "num_users": int(interactions_df[user_id_col].nunique()),
        "num_items": int(interactions_df[item_id_col].nunique()),
        "num_warm_items": int(len(hybrid_model.warm_cold_result.warm_items)),
        "num_cold_items": int(len(hybrid_model.warm_cold_result.cold_items)),
        "user_feature_cols": column_groups.user_feature_cols,
        "item_feature_cols": column_groups.item_feature_cols,
        "other_optional_cols": column_groups.other_optional_cols,
        "model_output_path": str(model_output_path) if model_output_path is not None else None,
    }
    return hybrid_model, training_summary
