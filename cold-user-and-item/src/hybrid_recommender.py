from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.als_model import ALSRecommender
from src.cold_user_recommender import ColdUserRecommendationResult, ColdUserRecommender
from src.feature_builder import build_inference_features
from src.maxvol_selector import MaxVolSelector
from src.popular_selector import PopularSelectionResult, PopularSelector
from src.preprocessing import PreprocessingArtifacts, RecommendationDataPreprocessor
from src.regressor_model import RegressorModel
from src.split_warm_cold import WarmColdItemSplitter, WarmColdSplitResult
from src.split_warm_cold_users import WarmColdUserSplitResult, WarmColdUserSplitter


@dataclass
class HybridRecommender:
    """
    Unified recommender router for warm-user, cold-user and global cold-start flows.
    """

    preprocessor: RecommendationDataPreprocessor
    warm_cold_splitter: WarmColdItemSplitter
    als_model: ALSRecommender
    regressor_model: RegressorModel
    negative_samples_per_user: int = 3
    random_state: int = 42
    warm_cold_user_splitter: WarmColdUserSplitter = field(default_factory=WarmColdUserSplitter)

    preprocessing_artifacts: PreprocessingArtifacts | None = None
    warm_cold_result: WarmColdSplitResult | None = None
    warm_cold_user_result: WarmColdUserSplitResult | None = None
    mode: str = "hybrid"
    popular_selector: PopularSelector | None = None
    maxvol_selector: MaxVolSelector | None = None
    popular_selection_result: PopularSelectionResult | None = None
    warm_interactions_df: pd.DataFrame | None = None

    def set_training_artifacts(
        self,
        preprocessing_artifacts: PreprocessingArtifacts,
        warm_cold_result: WarmColdSplitResult,
        warm_cold_user_result: WarmColdUserSplitResult,
        popular_selector: PopularSelector,
        maxvol_selector: MaxVolSelector,
        popular_selection_result: PopularSelectionResult,
        warm_interactions_df: pd.DataFrame,
        mode: str,
    ) -> "HybridRecommender":
        """
        Attach the artifacts produced by train_pipeline.py to the recommender.
        """
        self.preprocessing_artifacts = preprocessing_artifacts
        self.warm_cold_result = warm_cold_result
        self.warm_cold_user_result = warm_cold_user_result
        self.popular_selector = popular_selector
        self.maxvol_selector = maxvol_selector
        self.popular_selection_result = popular_selection_result
        self.warm_interactions_df = warm_interactions_df
        self.mode = mode
        return self

    def ensure_is_fitted(self) -> None:
        """
        Validate that core recommender artifacts are available.
        """
        if self.preprocessing_artifacts is None:
            raise RuntimeError("HybridRecommender must be fitted before inference.")
        if self.warm_cold_result is None or self.warm_cold_user_result is None:
            raise RuntimeError("Warm/cold splits are not available.")
        if self.popular_selector is None or self.maxvol_selector is None:
            raise RuntimeError("Fallback selectors are not available.")

    def has_fitted_regressor(self) -> bool:
        """
        Return whether the CatBoost regressor has already been trained.
        """
        return self.regressor_model.model is not None

    def is_global_cold_start(self) -> bool:
        """
        Return whether there is no usable warm collaborative zone.
        """
        self.ensure_split_state()
        return (
            len(self.warm_cold_user_result.warm_users) == 0
            or len(self.warm_cold_result.warm_items) == 0
            or self.warm_interactions_df is None
            or self.warm_interactions_df.empty
        )

    def ensure_split_state(self) -> None:
        """
        Validate that warm/cold split artifacts exist.
        """
        if self.warm_cold_result is None or self.warm_cold_user_result is None:
            raise RuntimeError("Warm/cold split state is not available.")

    def is_warm_user(self, user_id: str) -> bool:
        """
        Return whether the user belongs to the warm-user set.
        """
        self.ensure_is_fitted()
        return str(user_id) in set(map(str, self.warm_cold_user_result.warm_users))

    def is_cold_user(self, user_id: str) -> bool:
        """
        Return whether the user should be routed to the cold-user flow.
        """
        return not self.is_warm_user(user_id)

    def get_seen_items(self, user_id: str) -> set[str]:
        """
        Return the full set of items already seen by the user in the interaction history.
        """
        self.ensure_is_fitted()
        normalized_user_id = str(user_id)
        interactions_df = self.preprocessing_artifacts.interactions_df
        user_rows = interactions_df[
            interactions_df[self.preprocessor.user_id_col].astype(str) == normalized_user_id
        ]
        return set(user_rows[self.preprocessor.item_id_col].astype(str).tolist())

    def build_recommendation_frame(
        self,
        user_id: str,
        item_ids: list[str],
        scores: np.ndarray | list[float],
        source: str,
    ) -> pd.DataFrame:
        """
        Build a recommendation dataframe from aligned item ids and scores.
        """
        if not item_ids:
            return pd.DataFrame(
                columns=[
                    self.preprocessor.user_id_col,
                    self.preprocessor.item_id_col,
                    "score",
                    "source",
                ]
            )

        return pd.DataFrame(
            {
                self.preprocessor.user_id_col: [str(user_id)] * len(item_ids),
                self.preprocessor.item_id_col: [str(item_id) for item_id in item_ids],
                "score": np.asarray(scores, dtype=float),
                "source": source,
            }
        )

    def rank_and_cut(self, recommendations_df: pd.DataFrame, top_k: int) -> pd.DataFrame:
        """
        Sort recommendations by score and keep the top-K rows.
        """
        if recommendations_df.empty:
            return recommendations_df.copy()

        ranked_df = (
            recommendations_df.sort_values("score", ascending=False)
            .head(max(top_k, 0))
            .reset_index(drop=True)
        )
        ranked_df["rank"] = np.arange(1, len(ranked_df) + 1)
        return ranked_df

    def build_cold_user_recommender(self) -> ColdUserRecommender:
        """
        Construct the cold-user recommender using the cached fallback selectors.
        """
        self.ensure_is_fitted()
        return ColdUserRecommender(
            preprocessor=self.preprocessor,
            regressor_model=self.regressor_model,
            interactions_df=self.preprocessing_artifacts.interactions_df,
            user_features_df=self.preprocessing_artifacts.user_features_df,
            item_features_df=self.preprocessing_artifacts.item_features_df,
            popular_selector=self.popular_selector,
            maxvol_selector=self.maxvol_selector,
        )

    def recommend_warm_user(
        self,
        user_id: str,
        user_context: dict[str, object] | None = None,
        top_k: int = 10,
        candidate_item_ids: list[str] | None = None,
        exclude_seen: bool = True,
    ) -> pd.DataFrame:
        """
        Recommend for a warm user: ALS on warm items, regressor on cold items.
        """
        self.ensure_is_fitted()
        if self.mode == "global_cold_start":
            return self.recommend_global_cold_start(user_id=user_id, user_context=user_context, top_k=top_k)
        if self.is_cold_user(user_id):
            raise ValueError(f"User {user_id!r} is cold and should be routed to the cold-user flow.")

        normalized_user_id = str(user_id)
        warm_item_ids = list(map(str, self.warm_cold_result.warm_items))
        cold_item_ids = list(map(str, self.warm_cold_result.cold_items))

        if candidate_item_ids is not None:
            candidate_set = set(map(str, candidate_item_ids))
            warm_item_ids = [item_id for item_id in warm_item_ids if item_id in candidate_set]
            cold_item_ids = [item_id for item_id in cold_item_ids if item_id in candidate_set]

        if exclude_seen:
            seen_items = self.get_seen_items(normalized_user_id)
            warm_item_ids = [item_id for item_id in warm_item_ids if item_id not in seen_items]
            cold_item_ids = [item_id for item_id in cold_item_ids if item_id not in seen_items]

        warm_scores = self.als_model.score_user_items(normalized_user_id, warm_item_ids)
        warm_df = self.build_recommendation_frame(
            user_id=normalized_user_id,
            item_ids=warm_item_ids,
            scores=warm_scores,
            source="warm_user_warm_item_als",
        )

        if not cold_item_ids:
            return self.rank_and_cut(warm_df, top_k=top_k)

        if self.has_fitted_regressor():
            _, cold_feature_matrix = build_inference_features(
                preprocessor=self.preprocessor,
                user_id=normalized_user_id,
                item_ids=cold_item_ids,
                user_features_df=self.preprocessing_artifacts.user_features_df,
                item_features_df=self.preprocessing_artifacts.item_features_df,
                user_features=user_context,
            )
            if cold_feature_matrix.shape[1] == 0:
                cold_scores = self.popular_selector.get_scores(cold_item_ids).to_numpy(dtype=float)
                cold_source = "warm_user_cold_item_popularity_fallback"
            else:
                cold_scores = self.regressor_model.predict(cold_feature_matrix)
                cold_source = "warm_user_cold_item_catboost"
        else:
            cold_scores = self.popular_selector.get_scores(cold_item_ids).to_numpy(dtype=float)
            cold_source = "warm_user_cold_item_popularity_fallback"

        cold_df = self.build_recommendation_frame(
            user_id=normalized_user_id,
            item_ids=cold_item_ids,
            scores=cold_scores,
            source=cold_source,
        )
        return self.rank_and_cut(pd.concat([warm_df, cold_df], ignore_index=True), top_k=top_k)

    def recommend_cold_user(
        self,
        user_id: str,
        top_k: int = 10,
        top_n_popular: int = 200,
        top_k_diverse: int = 50,
        user_context: dict[str, object] | None = None,
    ) -> ColdUserRecommendationResult:
        """
        Recommend for a cold user through the dedicated fallback flow.
        """
        cold_user_recommender = self.build_cold_user_recommender()
        return cold_user_recommender.recommend(
            user_id=str(user_id),
            top_n_popular=top_n_popular,
            top_k_diverse=top_k_diverse,
            top_m=top_k,
            user_context=user_context,
        )

    def recommend_global_cold_start(
        self,
        user_id: str,
        user_context: dict[str, object] | None = None,
        top_k: int = 10,
    ) -> pd.DataFrame:
        """
        Recommend when no usable warm collaborative zone exists.
        """
        self.ensure_is_fitted()
        candidate_item_ids = self.maxvol_selector.select()
        if not candidate_item_ids:
            return self.build_recommendation_frame(
                user_id=str(user_id),
                item_ids=[],
                scores=[],
                source="global_cold_start_empty",
            )

        if self.has_fitted_regressor():
            _, feature_matrix = build_inference_features(
                preprocessor=self.preprocessor,
                user_id=str(user_id),
                item_ids=candidate_item_ids,
                user_features_df=self.preprocessing_artifacts.user_features_df,
                item_features_df=self.preprocessing_artifacts.item_features_df,
                user_features=user_context,
            )
            if feature_matrix.shape[1] == 0:
                scores = self.popular_selector.get_scores(candidate_item_ids).to_numpy(dtype=float)
                source = "global_cold_start_popular_maxvol"
            else:
                scores = self.regressor_model.predict(feature_matrix)
                source = "global_cold_start_catboost"
        else:
            scores = self.popular_selector.get_scores(candidate_item_ids).to_numpy(dtype=float)
            source = "global_cold_start_popular_maxvol"

        recommendation_df = self.build_recommendation_frame(
            user_id=str(user_id),
            item_ids=candidate_item_ids,
            scores=scores,
            source=source,
        )
        return self.rank_and_cut(recommendation_df, top_k=top_k)

    def recommend_for_user(
        self,
        user_id: str,
        user_context: dict[str, object] | None = None,
        top_k: int = 10,
        candidate_item_ids: list[str] | None = None,
        exclude_seen: bool = True,
        cold_top_n_popular: int = 200,
        cold_top_k_diverse: int = 50,
    ) -> pd.DataFrame | ColdUserRecommendationResult:
        """
        Route one user to the correct recommendation flow.
        """
        self.ensure_is_fitted()
        if self.mode == "global_cold_start":
            return self.recommend_global_cold_start(
                user_id=user_id,
                user_context=user_context,
                top_k=top_k,
            )
        if self.is_warm_user(user_id):
            return self.recommend_warm_user(
                user_id=user_id,
                user_context=user_context,
                top_k=top_k,
                candidate_item_ids=candidate_item_ids,
                exclude_seen=exclude_seen,
            )
        return self.recommend_cold_user(
            user_id=user_id,
            top_k=top_k,
            top_n_popular=cold_top_n_popular,
            top_k_diverse=cold_top_k_diverse,
            user_context=user_context,
        )

    def save(self, path: str | Path) -> None:
        """
        Save the full hybrid recommender artifact.
        """
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "HybridRecommender":
        """
        Load a saved hybrid recommender artifact.
        """
        return joblib.load(path)
