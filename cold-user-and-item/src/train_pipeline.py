from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.als_model import ALSRecommender
from src.data_loader import load_training_csv_data
from src.feature_builder import RegressorTrainingDataset, build_regressor_training_dataset
from src.hybrid_recommender import HybridRecommender
from src.maxvol_selector import MaxVolSelector
from src.popular_selector import PopularSelector, PopularSelectionResult
from src.preprocessing import RecommendationDataPreprocessor
from src.regressor_model import RegressorModel
from src.split_warm_cold import WarmColdItemSplitter, WarmColdSplitResult
from src.split_warm_cold_users import WarmColdUserSplitResult, WarmColdUserSplitter
from src.utils import ensure_parent_dir


def build_warm_interactions_df(
    interactions_df: pd.DataFrame,
    warm_user_ids: set[str],
    warm_item_ids: set[str],
    user_col: str = "user_id",
    item_col: str = "item_id",
) -> pd.DataFrame:
    """
    Build the observed warm-user + warm-item interaction subset.
    """
    normalized_df = interactions_df.copy()
    normalized_df[user_col] = normalized_df[user_col].astype(str)
    normalized_df[item_col] = normalized_df[item_col].astype(str)
    return normalized_df[
        normalized_df[user_col].isin(warm_user_ids)
        & normalized_df[item_col].isin(warm_item_ids)
    ].copy()


def detect_global_cold_start(
    warm_user_ids: set[str],
    warm_item_ids: set[str],
    warm_interactions_df: pd.DataFrame,
) -> bool:
    """
    Determine whether a usable warm collaborative zone exists.
    """
    return (
        len(warm_user_ids) == 0
        or len(warm_item_ids) == 0
        or warm_interactions_df.empty
    )


def fit_popular_and_maxvol(
    interactions_df: pd.DataFrame,
    item_features_df: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "item_id",
    value_col: str = "value",
    popularity_mode: str = "count",
    top_n_popular: int = 500,
    top_k_diverse: int = 100,
) -> tuple[PopularSelector, MaxVolSelector, PopularSelectionResult]:
    """
    Fit reusable fallback selectors for cold-user and global cold-start flows.
    """
    popular_selector = PopularSelector(
        item_col=item_col,
        user_col=user_col,
        value_col=value_col,
        popularity_mode=popularity_mode,
    ).fit(interactions_df)

    popular_item_ids = popular_selector.select_top_n(top_n_popular)
    popularity_df = (
        popular_selector.popularity_df.copy()
        if popular_selector.popularity_df is not None
        else pd.DataFrame(columns=[item_col, "popularity", "unique_users", "selection_rank"])
    )
    selected_items_df = popularity_df[popularity_df[item_col].astype(str).isin(popular_item_ids)].copy()
    selected_items_df = (
        selected_items_df.set_index(item_col).reindex(popular_item_ids).reset_index()
        if not selected_items_df.empty
        else selected_items_df
    )
    popular_result = PopularSelectionResult(
        top_items_df=selected_items_df.reset_index(drop=True),
        popularity_df=popularity_df.reset_index(drop=True),
        top_item_ids=popular_item_ids,
        strategy=str(popularity_df.attrs.get("strategy", "popularity_selection")),
        input_size=int(len(popularity_df)),
        output_size=int(len(popular_item_ids)),
    )

    maxvol_selector = MaxVolSelector(item_id_col=item_col).fit(
        candidate_item_ids=popular_item_ids,
        item_features=item_features_df,
        k=min(top_k_diverse, len(popular_item_ids)),
    )
    return popular_selector, maxvol_selector, popular_result


def train_hybrid_model(
    train_csv_path: str | Path,
    model_output_path: str | Path | None = None,
    user_id_col: str = "user_id",
    item_id_col: str = "item_id",
    value_col: str = "value",
    user_prefix: str = "user_",
    item_prefix: str = "item_",
    min_user_interactions: int = 5,
    min_item_interactions: int = 5,
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
    top_n_popular: int = 500,
    top_k_diverse: int = 100,
) -> tuple[HybridRecommender, dict]:
    """
    Train the unified hybrid recommender from a training CSV.

    Train order:
    1. load CSV
    2. fit preprocessing
    3. split users into warm/cold
    4. split items into warm/cold
    5. build warm_interactions_df
    6. detect global cold-start
    7. fit popular selector and maxvol selector
    8. if warm zone exists: train ALS and RegressorModel
    9. save HybridRecommender artifact
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
    preprocessing_artifacts = preprocessor.fit(train_df)
    interactions_df = preprocessing_artifacts.interactions_df

    warm_cold_splitter = WarmColdItemSplitter(
        min_item_interactions=min_item_interactions,
        popularity_metric=popularity_metric,
        item_col=item_id_col,
        value_col=value_col,
    )
    warm_cold_user_splitter = WarmColdUserSplitter(
        min_user_interactions=min_user_interactions,
        popularity_metric=popularity_metric,
        user_col=user_id_col,
        value_col=value_col,
    )

    warm_cold_result: WarmColdSplitResult = warm_cold_splitter.split(interactions_df)
    warm_cold_user_result: WarmColdUserSplitResult = warm_cold_user_splitter.split(interactions_df)

    warm_interactions_df = build_warm_interactions_df(
        interactions_df=interactions_df,
        warm_user_ids=set(map(str, warm_cold_user_result.warm_users)),
        warm_item_ids=set(map(str, warm_cold_result.warm_items)),
        user_col=user_id_col,
        item_col=item_id_col,
    )
    global_cold_start_mode = detect_global_cold_start(
        warm_user_ids=set(map(str, warm_cold_user_result.warm_users)),
        warm_item_ids=set(map(str, warm_cold_result.warm_items)),
        warm_interactions_df=warm_interactions_df,
    )

    popular_selector, maxvol_selector, popular_result = fit_popular_and_maxvol(
        interactions_df=interactions_df,
        item_features_df=preprocessing_artifacts.item_features_df,
        user_col=user_id_col,
        item_col=item_id_col,
        value_col=value_col,
        popularity_mode=popularity_metric,
        top_n_popular=top_n_popular,
        top_k_diverse=top_k_diverse,
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
    regressor_model = RegressorModel(
        iterations=regressor_iterations,
        learning_rate=regressor_learning_rate,
        depth=regressor_depth,
        loss_function=regressor_loss_function,
        random_seed=regressor_random_seed,
    )

    regressor_dataset: RegressorTrainingDataset | None = None
    mode = "global_cold_start" if global_cold_start_mode else "hybrid"
    if not global_cold_start_mode:
        als_model.fit(warm_interactions_df)
        regressor_dataset = build_regressor_training_dataset(
            interactions_df=interactions_df,
            user_features_df=preprocessing_artifacts.user_features_df,
            item_features_df=preprocessing_artifacts.item_features_df,
            warm_cold_split=warm_cold_result,
            warm_cold_user_split=warm_cold_user_result,
            preprocessor=preprocessor,
            als_model=als_model,
            negative_samples_per_user=negative_samples_per_user,
            random_state=als_random_state,
            user_col=user_id_col,
            item_col=item_id_col,
        )
        regressor_model.fit(
            X=regressor_dataset.X,
            y=regressor_dataset.y.to_numpy(),
        )

    hybrid_model = HybridRecommender(
        preprocessor=preprocessor,
        warm_cold_splitter=warm_cold_splitter,
        als_model=als_model,
        regressor_model=regressor_model,
        negative_samples_per_user=negative_samples_per_user,
        random_state=als_random_state,
        warm_cold_user_splitter=warm_cold_user_splitter,
    )
    hybrid_model.set_training_artifacts(
        preprocessing_artifacts=preprocessing_artifacts,
        warm_cold_result=warm_cold_result,
        warm_cold_user_result=warm_cold_user_result,
        popular_selector=popular_selector,
        maxvol_selector=maxvol_selector,
        popular_selection_result=popular_result,
        warm_interactions_df=warm_interactions_df,
        mode=mode,
    )

    if model_output_path is not None:
        hybrid_model.save(ensure_parent_dir(model_output_path))

    training_summary = {
        "train_csv_path": str(train_csv_path),
        "mode": mode,
        "num_rows": int(len(interactions_df)),
        "num_users": int(interactions_df[user_id_col].nunique()),
        "num_items": int(interactions_df[item_id_col].nunique()),
        "num_warm_users": int(len(warm_cold_user_result.warm_users)),
        "num_cold_users": int(len(warm_cold_user_result.cold_users)),
        "num_warm_items": int(len(warm_cold_result.warm_items)),
        "num_cold_items": int(len(warm_cold_result.cold_items)),
        "num_warm_interactions": int(len(warm_interactions_df)),
        "global_cold_start_mode": bool(global_cold_start_mode),
        "num_popular_candidates": int(len(popular_result.top_item_ids)),
        "num_diverse_candidates": int(len(maxvol_selector.select())),
        "num_regressor_train_pairs": int(len(regressor_dataset.pairs_df)) if regressor_dataset is not None else 0,
        "user_feature_cols": column_groups.user_feature_cols,
        "item_feature_cols": column_groups.item_feature_cols,
        "other_optional_cols": column_groups.other_optional_cols,
        "model_output_path": str(model_output_path) if model_output_path is not None else None,
    }
    return hybrid_model, training_summary
