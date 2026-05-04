from __future__ import annotations

from dataclasses import dataclass, field

import joblib
import pandas as pd

from config import ColdItemProjectConfig, DEFAULT_CONFIG
from src.als_model import ALSRecommender
from src.candidate_generator import CandidateGenerator
from src.cold_vector_builder import ColdItemVectorBuilder, ColdVectorBuildResult
from src.data_loader import CSVColumnGroups, load_training_csv_data
from src.feature_builder import RankerDataset, RankerFeatureBuilder, attach_labels_to_pairs
from src.maxvol_selector import DiverseSelectionResult, MaxVolSelector
from src.popular_selector import PopularItemsSelector, PopularSelectionResult
from src.preprocessing import PreprocessingArtifacts, RecommendationDataPreprocessor
from src.ranker_model import CatBoostItemRanker
from src.similarity_index import ItemSimilarityIndex, NeighborSearchResult
from src.split_warm_cold import WarmColdItemSplitter, WarmColdSplitResult


@dataclass(slots=True)
class TrainPipelineArtifacts:
    """
    Fitted training-stage objects and intermediate results.
    """

    column_groups: CSVColumnGroups
    preprocessing_artifacts: PreprocessingArtifacts
    warm_cold_result: WarmColdSplitResult
    als_model: ALSRecommender
    popular_selection: PopularSelectionResult
    support_selection: DiverseSelectionResult
    neighbor_search: NeighborSearchResult
    cold_vector_result: ColdVectorBuildResult
    candidate_pairs_df: pd.DataFrame
    ranker_dataset: RankerDataset
    ranker_model: CatBoostItemRanker


@dataclass(slots=True)
class TrainPipelineResult:
    """
    Final output of the cold-item training pipeline.
    """

    artifacts: TrainPipelineArtifacts
    training_summary: dict[str, object]


def filter_interactions_by_items(
    interactions_df: pd.DataFrame,
    item_ids: set[str] | list[str],
    item_col: str,
) -> pd.DataFrame:
    """
    Keep only rows whose item ids belong to the provided set.
    """
    if not item_ids:
        return interactions_df.iloc[0:0].copy()

    normalized_item_ids = {str(item_id) for item_id in item_ids}
    filtered_df = interactions_df[interactions_df[item_col].astype(str).isin(normalized_item_ids)].copy()
    return filtered_df.reset_index(drop=True)


def build_item_feature_matrix(
    preprocessor: RecommendationDataPreprocessor,
    items_df: pd.DataFrame,
    user_features_df: pd.DataFrame,
    item_features_df: pd.DataFrame,
) -> tuple[pd.DataFrame, object]:
    """
    Encode item rows into the pair-feature space with one anchor user.

    The user-side features are held constant across all rows, so only item-side
    variation affects support-set diversification and neighbor search.
    """
    if items_df.empty:
        empty_pairs_df = pd.DataFrame(columns=[preprocessor.user_id_col, preprocessor.item_id_col])
        _, empty_matrix = preprocessor.transform_pairs(
            empty_pairs_df,
            user_features_df=user_features_df,
            item_features_df=item_features_df,
        )
        return empty_pairs_df, empty_matrix

    if user_features_df.empty:
        anchor_user_df = pd.DataFrame({preprocessor.user_id_col: ["__anchor_user__"]})
    else:
        anchor_user_df = user_features_df.iloc[[0]].copy().reset_index(drop=True)

    anchor_user_id = str(anchor_user_df.iloc[0][preprocessor.user_id_col])
    pairs_df = pd.DataFrame(
        {
            preprocessor.user_id_col: [anchor_user_id] * len(items_df),
            preprocessor.item_id_col: items_df[preprocessor.item_id_col].astype(str).tolist(),
        }
    )
    pair_feature_df, feature_matrix = preprocessor.transform_pairs(
        pairs_df,
        user_features_df=anchor_user_df,
        item_features_df=item_features_df,
    )
    return pair_feature_df, feature_matrix


def generate_training_candidates(
    interactions_df: pd.DataFrame,
    als_model: ALSRecommender,
    cold_vector_map: dict[str, object],
    warm_item_ids: list[str],
    cold_item_ids: list[str],
    candidate_generator: CandidateGenerator,
    user_id_col: str,
) -> pd.DataFrame:
    """
    Generate retrieval-stage candidate pools for all training users.
    """
    candidate_frames: list[pd.DataFrame] = []
    user_ids = interactions_df[user_id_col].astype(str).drop_duplicates().tolist()

    for user_id in user_ids:
        candidate_result = candidate_generator.generate_for_user(
            als_model=als_model,
            user_id=user_id,
            cold_vector_map=cold_vector_map,
            warm_candidate_item_ids=warm_item_ids,
            cold_candidate_item_ids=cold_item_ids,
        )
        if not candidate_result.candidates_df.empty:
            candidate_frames.append(candidate_result.candidates_df)

    if not candidate_frames:
        return pd.DataFrame(
            columns=[
                user_id_col,
                als_model.item_col,
                "retrieval_score",
                "retrieval_source",
                "is_cold_item",
                "retrieval_rank",
            ]
        )

    return pd.concat(candidate_frames, ignore_index=True)


def sample_training_pairs(
    labeled_pairs_df: pd.DataFrame,
    user_id_col: str,
    negative_samples_per_user: int,
) -> pd.DataFrame:
    """
    Keep all positives and a bounded number of retrieval-top negatives per user.
    """
    if labeled_pairs_df.empty:
        return labeled_pairs_df.copy()

    sampled_groups: list[pd.DataFrame] = []
    sort_columns = [user_id_col]
    ascending = [True]
    if "retrieval_rank" in labeled_pairs_df.columns:
        sort_columns.append("retrieval_rank")
        ascending.append(True)
    elif "retrieval_score" in labeled_pairs_df.columns:
        sort_columns.append("retrieval_score")
        ascending.append(False)

    ordered_df = labeled_pairs_df.sort_values(sort_columns, ascending=ascending).reset_index(drop=True)

    for _, group_df in ordered_df.groupby(user_id_col, sort=False):
        positives_df = group_df[group_df["label"] > 0].copy()
        if positives_df.empty:
            continue

        negatives_df = group_df[group_df["label"] <= 0].copy()
        if negative_samples_per_user > 0:
            max_negatives = max(len(positives_df) * negative_samples_per_user, negative_samples_per_user)
            negatives_df = negatives_df.head(max_negatives).copy()

        sampled_group_df = pd.concat([positives_df, negatives_df], ignore_index=True)
        if len(sampled_group_df) >= 2:
            sampled_groups.append(sampled_group_df)

    if not sampled_groups:
        raise ValueError("No ranker training groups remained after candidate sampling.")

    sampled_pairs_df = pd.concat(sampled_groups, ignore_index=True)
    sampled_pairs_df = sampled_pairs_df.sort_values(
        [user_id_col, "label", "retrieval_rank"] if "retrieval_rank" in sampled_pairs_df.columns else [user_id_col, "label"],
        ascending=[True, False, True] if "retrieval_rank" in sampled_pairs_df.columns else [True, False],
    ).reset_index(drop=True)
    return sampled_pairs_df


def save_training_artifacts(
    config: ColdItemProjectConfig,
    preprocessing_artifacts: PreprocessingArtifacts,
    preprocessor: RecommendationDataPreprocessor,
    warm_cold_result: WarmColdSplitResult,
    als_model: ALSRecommender,
    support_selection: DiverseSelectionResult,
    neighbor_search: NeighborSearchResult,
    cold_vector_result: ColdVectorBuildResult,
    ranker_model: CatBoostItemRanker,
) -> None:
    """
    Persist the fitted artifacts required for inference.
    """
    config.paths.ensure_artifacts_dir()
    joblib.dump(config, config.paths.project_config_path)
    joblib.dump(preprocessing_artifacts, config.paths.preprocessing_artifacts_path)
    joblib.dump(preprocessor, config.paths.preprocessor_path)
    joblib.dump(warm_cold_result, config.paths.warm_cold_split_path)
    joblib.dump(support_selection, config.paths.support_items_path)
    joblib.dump(neighbor_search, config.paths.cold_neighbors_path)
    joblib.dump(cold_vector_result, config.paths.cold_vectors_path)
    als_model.save(config.paths.als_model_path)
    ranker_model.save(config.paths.ranker_model_path)


def train_cold_item_pipeline(
    train_csv_path: str,
    config: ColdItemProjectConfig = DEFAULT_CONFIG,
    save_artifacts: bool = True,
    use_interaction_value_as_label: bool = False,
) -> TrainPipelineResult:
    """
    Train the cold-item retrieval + ranking stack on one CSV dataset.
    """
    train_df, column_groups = load_training_csv_data(
        csv_path=train_csv_path,
        user_id_col=config.data.user_id_col,
        item_id_col=config.data.item_id_col,
        value_col=config.data.value_col,
        user_prefix=config.data.user_prefix,
        item_prefix=config.data.item_prefix,
        sep=config.data.csv_sep,
        encoding=config.data.csv_encoding,
    )

    preprocessor = RecommendationDataPreprocessor(
        user_id_col=config.data.user_id_col,
        item_id_col=config.data.item_id_col,
        value_col=config.data.value_col,
        user_prefix=config.data.user_prefix,
        item_prefix=config.data.item_prefix,
    )
    preprocessing_artifacts = preprocessor.fit(train_df)
    interactions_df = preprocessing_artifacts.interactions_df

    warm_cold_splitter = WarmColdItemSplitter(
        min_warm_interactions=config.warm_cold.min_warm_interactions,
        popularity_metric=config.warm_cold.popularity_metric,
        item_col=config.data.item_id_col,
        value_col=config.data.value_col,
    )
    warm_cold_result = warm_cold_splitter.split(interactions_df)

    warm_interactions_df = filter_interactions_by_items(
        interactions_df=interactions_df,
        item_ids=warm_cold_result.warm_items,
        item_col=config.data.item_id_col,
    )
    if warm_interactions_df.empty:
        raise ValueError("Warm-item ALS training set is empty. Lower the warm threshold or inspect the dataset.")

    als_model = ALSRecommender(
        user_col=config.data.user_id_col,
        item_col=config.data.item_id_col,
        value_col=config.data.value_col,
        factors=config.als.factors,
        regularization=config.als.regularization,
        iterations=config.als.iterations,
        alpha=config.als.alpha,
        random_state=config.als.random_state,
    ).fit(warm_interactions_df)

    popular_selector = PopularItemsSelector(
        item_col=config.data.item_id_col,
        user_col=config.data.user_id_col,
        value_col=config.data.value_col,
    )
    popular_selection = popular_selector.select(
        interactions=warm_interactions_df,
        top_n=config.retrieval.top_n_popular,
    )

    support_candidate_ids = set(popular_selection.top_item_ids) & set(warm_cold_result.warm_items)
    support_candidate_df = preprocessing_artifacts.item_features_df[
        preprocessing_artifacts.item_features_df[config.data.item_id_col].astype(str).isin(support_candidate_ids)
    ].copy().reset_index(drop=True)

    _, support_feature_matrix = build_item_feature_matrix(
        preprocessor=preprocessor,
        items_df=support_candidate_df,
        user_features_df=preprocessing_artifacts.user_features_df,
        item_features_df=preprocessing_artifacts.item_features_df,
    )

    maxvol_selector = MaxVolSelector(
        item_id_col=config.data.item_id_col,
        max_projection_dim=config.als.factors,
        random_state=config.als.random_state,
    )
    support_selection = maxvol_selector.select(
        items_df=support_candidate_df,
        feature_matrix=support_feature_matrix,
        top_k=config.retrieval.top_k_diverse,
    )

    cold_items_df = preprocessing_artifacts.item_features_df[
        preprocessing_artifacts.item_features_df[config.data.item_id_col].astype(str).isin(warm_cold_result.cold_items)
    ].copy().reset_index(drop=True)
    _, cold_feature_matrix = build_item_feature_matrix(
        preprocessor=preprocessor,
        items_df=cold_items_df,
        user_features_df=preprocessing_artifacts.user_features_df,
        item_features_df=preprocessing_artifacts.item_features_df,
    )
    _, support_diverse_feature_matrix = build_item_feature_matrix(
        preprocessor=preprocessor,
        items_df=support_selection.selected_items_df,
        user_features_df=preprocessing_artifacts.user_features_df,
        item_features_df=preprocessing_artifacts.item_features_df,
    )

    similarity_index = ItemSimilarityIndex(
        cold_item_col=config.data.item_id_col,
        support_item_col=config.data.item_id_col,
        similarity_metric=config.retrieval.similarity_metric,
        max_projection_dim=config.als.factors,
        random_state=config.als.random_state,
    )
    neighbor_search = similarity_index.find_neighbors(
        cold_items_df=cold_items_df,
        cold_feature_matrix=cold_feature_matrix,
        support_items_df=support_selection.selected_items_df,
        support_feature_matrix=support_diverse_feature_matrix,
        top_m=config.retrieval.top_m_neighbors,
    )

    cold_vector_builder = ColdItemVectorBuilder(cold_item_col=config.data.item_id_col)
    cold_vector_result = cold_vector_builder.build(
        neighbors_df=neighbor_search.neighbors_df,
        als_model=als_model,
    )

    candidate_generator = CandidateGenerator(
        warm_candidates_per_user=config.retrieval.warm_candidates_per_user,
        cold_candidates_per_user=config.retrieval.cold_candidates_per_user,
        final_candidate_pool_size=config.retrieval.final_candidate_pool_size,
        exclude_seen=False,
    )
    candidate_pairs_df = generate_training_candidates(
        interactions_df=interactions_df,
        als_model=als_model,
        cold_vector_map=cold_vector_result.cold_vector_map,
        warm_item_ids=sorted(warm_cold_result.warm_items),
        cold_item_ids=sorted(cold_vector_result.cold_vector_map.keys()),
        candidate_generator=candidate_generator,
        user_id_col=config.data.user_id_col,
    )
    if candidate_pairs_df.empty:
        raise ValueError("Candidate generation produced an empty training pool.")

    labeled_candidate_pairs_df = attach_labels_to_pairs(
        pairs_df=candidate_pairs_df,
        interactions_df=interactions_df,
        user_id_col=config.data.user_id_col,
        item_id_col=config.data.item_id_col,
        value_col=config.data.value_col,
        use_interaction_value_as_label=use_interaction_value_as_label,
    )
    sampled_candidate_pairs_df = sample_training_pairs(
        labeled_pairs_df=labeled_candidate_pairs_df,
        user_id_col=config.data.user_id_col,
        negative_samples_per_user=config.ranker.negative_samples_per_user,
    ).drop(columns=["label"])

    feature_builder = RankerFeatureBuilder(
        preprocessor=preprocessor,
        use_interaction_value_as_label=use_interaction_value_as_label,
    )
    ranker_dataset = feature_builder.build_training_dataset(
        candidate_pairs_df=sampled_candidate_pairs_df,
        interactions_df=interactions_df,
        user_features_df=preprocessing_artifacts.user_features_df,
        item_features_df=preprocessing_artifacts.item_features_df,
    )

    ranker_model = CatBoostItemRanker(
        iterations=config.ranker.iterations,
        learning_rate=config.ranker.learning_rate,
        depth=config.ranker.depth,
        loss_function=config.ranker.loss_function,
        random_seed=config.ranker.random_seed,
    ).fit(ranker_dataset)

    if save_artifacts:
        save_training_artifacts(
            config=config,
            preprocessing_artifacts=preprocessing_artifacts,
            preprocessor=preprocessor,
            warm_cold_result=warm_cold_result,
            als_model=als_model,
            support_selection=support_selection,
            neighbor_search=neighbor_search,
            cold_vector_result=cold_vector_result,
            ranker_model=ranker_model,
        )

    training_summary = {
        "train_csv_path": str(train_csv_path),
        "num_rows": int(len(interactions_df)),
        "num_users": int(interactions_df[config.data.user_id_col].nunique()),
        "num_items": int(interactions_df[config.data.item_id_col].nunique()),
        "num_warm_items": int(len(warm_cold_result.warm_items)),
        "num_cold_items": int(len(warm_cold_result.cold_items)),
        "num_support_candidates": int(len(support_candidate_df)),
        "num_support_items": int(len(support_selection.selected_items_df)),
        "num_cold_vectors": int(cold_vector_result.num_built_vectors),
        "num_candidate_pairs": int(len(candidate_pairs_df)),
        "num_ranker_pairs": int(len(ranker_dataset.pairs_df)),
        "ranker_positive_pairs": int((ranker_dataset.labels > 0).sum()) if ranker_dataset.labels is not None else 0,
        "ranker_label_mode": "interaction_value" if use_interaction_value_as_label else "binary_relevance",
        "user_feature_cols": column_groups.user_feature_cols,
        "item_feature_cols": column_groups.item_feature_cols,
        "other_optional_cols": column_groups.other_optional_cols,
        "artifacts_dir": str(config.paths.artifacts_dir) if save_artifacts else None,
    }

    artifacts = TrainPipelineArtifacts(
        column_groups=column_groups,
        preprocessing_artifacts=preprocessing_artifacts,
        warm_cold_result=warm_cold_result,
        als_model=als_model,
        popular_selection=popular_selection,
        support_selection=support_selection,
        neighbor_search=neighbor_search,
        cold_vector_result=cold_vector_result,
        candidate_pairs_df=candidate_pairs_df,
        ranker_dataset=ranker_dataset,
        ranker_model=ranker_model,
    )
    return TrainPipelineResult(
        artifacts=artifacts,
        training_summary=training_summary,
    )


@dataclass(slots=True)
class ColdItemTrainPipeline:
    """
    Thin object wrapper around the cold-item training pipeline.
    """

    config: ColdItemProjectConfig = field(default_factory=ColdItemProjectConfig)
    save_artifacts: bool = True

    def run(self, train_csv_path: str) -> TrainPipelineResult:
        """
        Execute full training on the provided CSV path.
        """
        return train_cold_item_pipeline(
            train_csv_path=train_csv_path,
            config=self.config,
            save_artifacts=self.save_artifacts,
        )
