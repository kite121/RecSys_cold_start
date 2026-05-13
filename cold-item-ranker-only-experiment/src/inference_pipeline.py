from __future__ import annotations

from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from scipy import sparse

from config import ColdItemProjectConfig, DEFAULT_CONFIG
from src.als_model import ALSRecommender
from src.candidate_generator import CandidateGenerator
from src.cold_vector_builder import ColdVectorBuildResult
from src.feature_builder import RankerDataset, RankerFeatureBuilder
from src.preprocessing import PreprocessingArtifacts, RecommendationDataPreprocessor
from src.ranker_model import CatBoostItemRanker, RankerPredictionResult
from src.split_warm_cold import WarmColdSplitResult


@dataclass(slots=True)
class InferenceArtifacts:
    """
    Loaded artifacts required to score recommendations.
    """

    preprocessing_artifacts: PreprocessingArtifacts
    preprocessor: RecommendationDataPreprocessor
    warm_cold_result: WarmColdSplitResult
    als_model: ALSRecommender
    cold_vector_result: ColdVectorBuildResult
    ranker_model: CatBoostItemRanker


@dataclass(slots=True)
class InferencePipelineResult:
    """
    Final inference output for one user request.
    """

    candidates_df: pd.DataFrame
    ranker_dataset: RankerDataset
    prediction_result: RankerPredictionResult
    recommendations_df: pd.DataFrame
    inference_summary: dict[str, object]


def load_inference_artifacts(config: ColdItemProjectConfig = DEFAULT_CONFIG) -> InferenceArtifacts:
    """
    Load all train-time artifacts needed for inference.
    """
    preprocessing_artifacts = joblib.load(config.paths.preprocessing_artifacts_path)
    preprocessor = joblib.load(config.paths.preprocessor_path)
    warm_cold_result = joblib.load(config.paths.warm_cold_split_path)
    cold_vector_result = joblib.load(config.paths.cold_vectors_path)

    als_model = ALSRecommender.load(config.paths.als_model_path)
    ranker_model = CatBoostItemRanker.load(config.paths.ranker_model_path)
    return InferenceArtifacts(
        preprocessing_artifacts=preprocessing_artifacts,
        preprocessor=preprocessor,
        warm_cold_result=warm_cold_result,
        als_model=als_model,
        cold_vector_result=cold_vector_result,
        ranker_model=ranker_model,
    )


def build_target_user_feature_table(
    user_id: str,
    preprocessing_artifacts: PreprocessingArtifacts,
    preprocessor: RecommendationDataPreprocessor,
    user_context: dict[str, object] | None = None,
) -> pd.DataFrame:
    """
    Build a one-user feature table for the target inference request.
    """
    user_id = str(user_id)
    base_user_rows = preprocessing_artifacts.user_features_df[
        preprocessing_artifacts.user_features_df[preprocessor.user_id_col].astype(str) == user_id
    ].copy()

    if base_user_rows.empty:
        base_user_df = pd.DataFrame({preprocessor.user_id_col: [user_id]})
    else:
        base_user_df = base_user_rows.iloc[[0]].copy().reset_index(drop=True)

    if user_context:
        for feature_name, feature_value in user_context.items():
            if feature_name == preprocessor.user_id_col:
                continue
            base_user_df.loc[0, feature_name] = feature_value

    if preprocessor.feature_roles is not None:
        expected_user_columns = [preprocessor.user_id_col] + list(preprocessor.feature_roles.user_feature_cols)
        for column in expected_user_columns:
            if column not in base_user_df.columns:
                base_user_df[column] = pd.NA
        base_user_df = base_user_df[expected_user_columns]

    return base_user_df.reset_index(drop=True)


def build_candidate_pool_for_user(
    user_id: str,
    artifacts: InferenceArtifacts,
    config: ColdItemProjectConfig,
    warm_candidate_item_ids: list[str] | None = None,
    cold_candidate_item_ids: list[str] | None = None,
) -> pd.DataFrame:
    """
    Generate the retrieval-stage candidate pool for one inference request.
    """
    candidate_generator = CandidateGenerator(
        warm_candidates_per_user=config.retrieval.warm_candidates_per_user,
        cold_candidates_per_user=config.retrieval.cold_candidates_per_user,
        final_candidate_pool_size=config.retrieval.final_candidate_pool_size,
        exclude_seen=config.retrieval.exclude_seen_items,
    )
    candidate_result = candidate_generator.generate_for_user(
        als_model=artifacts.als_model,
        user_id=str(user_id),
        cold_vector_map=artifacts.cold_vector_result.cold_vector_map,
        warm_candidate_item_ids=warm_candidate_item_ids,
        cold_candidate_item_ids=cold_candidate_item_ids,
    )
    return candidate_result.candidates_df


def run_cold_item_inference(
    user_id: str,
    config: ColdItemProjectConfig = DEFAULT_CONFIG,
    top_k: int | None = None,
    user_context: dict[str, object] | None = None,
    warm_candidate_item_ids: list[str] | None = None,
    cold_candidate_item_ids: list[str] | None = None,
) -> InferencePipelineResult:
    """
    Run full retrieval + ranking inference for one user.
    """
    artifacts = load_inference_artifacts(config=config)
    effective_top_k = config.inference.top_k if top_k is None else int(top_k)

    candidates_df = build_candidate_pool_for_user(
        user_id=str(user_id),
        artifacts=artifacts,
        config=config,
        warm_candidate_item_ids=warm_candidate_item_ids,
        cold_candidate_item_ids=cold_candidate_item_ids,
    )
    if candidates_df.empty:
        empty_dataset = RankerDataset(
            pairs_df=pd.DataFrame(),
            feature_matrix=sparse.csr_matrix((0, 0), dtype=np.float32),
            feature_names=[],
            group_ids=np.empty(0, dtype=np.int64),
            group_keys=np.empty(0, dtype=object),
            labels=None,
        )
        empty_prediction = RankerPredictionResult(
            scored_pairs_df=pd.DataFrame(),
            scores=np.empty(0, dtype=np.float32),
        )
        return InferencePipelineResult(
            candidates_df=candidates_df,
            ranker_dataset=empty_dataset,
            prediction_result=empty_prediction,
            recommendations_df=pd.DataFrame(),
            inference_summary={
                "user_id": str(user_id),
                "num_candidates": 0,
                "num_recommendations": 0,
                "status": "empty_candidate_pool",
            },
        )

    user_features_df = build_target_user_feature_table(
        user_id=str(user_id),
        preprocessing_artifacts=artifacts.preprocessing_artifacts,
        preprocessor=artifacts.preprocessor,
        user_context=user_context,
    )
    item_features_df = artifacts.preprocessing_artifacts.item_features_df.copy()

    feature_builder = RankerFeatureBuilder(preprocessor=artifacts.preprocessor)
    ranker_dataset = feature_builder.build_inference_dataset(
        candidate_pairs_df=candidates_df,
        user_features_df=user_features_df,
        item_features_df=item_features_df,
    )
    prediction_result = artifacts.ranker_model.rank_candidates(
        dataset=ranker_dataset,
        top_k=effective_top_k,
    )
    recommendations_df = prediction_result.scored_pairs_df.reset_index(drop=True)

    inference_summary = {
        "user_id": str(user_id),
        "num_candidates": int(len(candidates_df)),
        "num_recommendations": int(len(recommendations_df)),
        "num_cold_candidates": int(candidates_df["is_cold_item"].fillna(False).astype(bool).sum()),
        "num_warm_candidates": int((~candidates_df["is_cold_item"].fillna(False).astype(bool)).sum()),
        "top_k": int(effective_top_k),
        "status": "ok",
    }
    return InferencePipelineResult(
        candidates_df=candidates_df,
        ranker_dataset=ranker_dataset,
        prediction_result=prediction_result,
        recommendations_df=recommendations_df,
        inference_summary=inference_summary,
    )


@dataclass(slots=True)
class ColdItemInferencePipeline:
    """
    Thin object wrapper around inference-time recommendation scoring.
    """

    config: ColdItemProjectConfig = field(default_factory=ColdItemProjectConfig)

    def run(
        self,
        user_id: str,
        top_k: int | None = None,
        user_context: dict[str, object] | None = None,
        warm_candidate_item_ids: list[str] | None = None,
        cold_candidate_item_ids: list[str] | None = None,
    ) -> InferencePipelineResult:
        """
        Execute full inference for one user.
        """
        return run_cold_item_inference(
            user_id=str(user_id),
            config=self.config,
            top_k=top_k,
            user_context=user_context,
            warm_candidate_item_ids=warm_candidate_item_ids,
            cold_candidate_item_ids=cold_candidate_item_ids,
        )
