from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.feature_builder import build_inference_features
from src.maxvol_selector import DiverseSelectionResult, MaxVolSelector
from src.popular_selector import PopularSelectionResult, PopularSelector
from src.preprocessing import RecommendationDataPreprocessor
from src.regressor_model import RegressorModel


@dataclass
class ColdUserRecommendationResult:
    """
    Result of the cold-user recommendation flow.
    """

    user_id: str
    candidate_pairs_df: pd.DataFrame
    scored_candidates_df: pd.DataFrame
    recommendations_df: pd.DataFrame
    popular_selection: PopularSelectionResult
    diverse_selection: DiverseSelectionResult
    strategy: str


def build_cold_user_pairs(
    user_id: str,
    item_ids: list[str],
    user_col: str = "user_id",
    item_col: str = "item_id",
) -> pd.DataFrame:
    """
    Build a candidate pair dataframe for one cold user.
    """
    if not item_ids:
        return pd.DataFrame(columns=[user_col, item_col])

    return pd.DataFrame(
        {
            user_col: [str(user_id)] * len(item_ids),
            item_col: [str(item_id) for item_id in item_ids],
        }
    )


@dataclass
class ColdUserRecommender:
    """
    Cold-user fallback flow:
    - top-N popular items
    - maxvol diversification
    - CatBoostRegressor scoring
    - final sorting
    """

    preprocessor: RecommendationDataPreprocessor
    regressor_model: RegressorModel
    interactions_df: pd.DataFrame
    user_features_df: pd.DataFrame
    item_features_df: pd.DataFrame
    popular_selector: PopularSelector | None = None
    maxvol_selector: MaxVolSelector | None = None

    def __post_init__(self) -> None:
        if self.popular_selector is None:
            self.popular_selector = PopularSelector(
                item_col=self.preprocessor.item_id_col,
                user_col=self.preprocessor.user_id_col,
                value_col=self.preprocessor.value_col,
            )
        if self.maxvol_selector is None:
            self.maxvol_selector = MaxVolSelector(item_id_col=self.preprocessor.item_id_col)

    def build_candidate_pool(
        self,
        user_id: str,
        top_n_popular: int,
        top_k_diverse: int,
    ) -> tuple[pd.DataFrame, PopularSelectionResult, DiverseSelectionResult]:
        """
        Build the cold-user candidate pool: popular items -> maxvol.
        """
        if self.popular_selector.popularity_df is None:
            self.popular_selector.fit(self.interactions_df)
        popular_item_ids = self.popular_selector.select_top_n(top_n_popular)
        popularity_df = self.popular_selector.popularity_df.copy() if self.popular_selector.popularity_df is not None else pd.DataFrame()
        selected_items_df = popularity_df[popularity_df[self.preprocessor.item_id_col].astype(str).isin(popular_item_ids)].copy()
        selected_items_df = (
            selected_items_df.set_index(self.preprocessor.item_id_col)
            .reindex(popular_item_ids)
            .reset_index()
        ) if not selected_items_df.empty else selected_items_df
        popular_result = PopularSelectionResult(
            top_items_df=selected_items_df.reset_index(drop=True),
            popularity_df=popularity_df.reset_index(drop=True),
            top_item_ids=popular_item_ids,
            strategy=str(popularity_df.attrs.get("strategy", "popularity_selection")),
            input_size=int(len(popularity_df)),
            output_size=int(len(popular_item_ids)),
        )
        if popular_result.top_items_df.empty:
            empty_diverse_result = DiverseSelectionResult(
                selected_indices=[],
                selected_item_ids=[],
                selected_items_df=popular_result.top_items_df.copy(),
                input_size=0,
                output_size=0,
                strategy="empty_popular_pool",
            )
            empty_pairs_df = build_cold_user_pairs(
                user_id=user_id,
                item_ids=[],
                user_col=self.preprocessor.user_id_col,
                item_col=self.preprocessor.item_id_col,
            )
            return empty_pairs_df, popular_result, empty_diverse_result

        needs_refit = (
            self.maxvol_selector.selection_result is None
            or self.maxvol_selector.fitted_candidate_item_ids != list(popular_result.top_item_ids)
            or self.maxvol_selector.fitted_k != int(top_k_diverse)
        )
        if needs_refit:
            self.maxvol_selector.fit(
                candidate_item_ids=popular_result.top_item_ids,
                item_features=self.item_features_df,
                k=top_k_diverse,
            )
        diverse_result = self.maxvol_selector.get_result()

        diverse_pairs_df = build_cold_user_pairs(
            user_id=user_id,
            item_ids=diverse_result.selected_item_ids,
            user_col=self.preprocessor.user_id_col,
            item_col=self.preprocessor.item_id_col,
        )
        if diverse_pairs_df.empty:
            return diverse_pairs_df, popular_result, diverse_result

        popular_metadata_df = (
            popular_result.top_items_df[
                [
                    self.preprocessor.item_id_col,
                    "popularity",
                    "unique_users",
                    "selection_rank",
                ]
            ]
            .drop_duplicates(subset=[self.preprocessor.item_id_col], keep="last")
        )
        diverse_items_df = diverse_result.selected_items_df.merge(
            popular_metadata_df,
            on=self.preprocessor.item_id_col,
            how="left",
        )
        diverse_pairs_df = diverse_pairs_df.merge(
            diverse_items_df,
            on=self.preprocessor.item_id_col,
            how="left",
        )
        diverse_pairs_df["retrieval_source"] = "popular_maxvol_cold_user"
        diverse_pairs_df["is_cold_user"] = True
        return diverse_pairs_df, popular_result, diverse_result

    def score_candidates(
        self,
        candidate_pairs_df: pd.DataFrame,
        user_context: dict[str, object] | None = None,
    ) -> pd.DataFrame:
        """
        Score the cold-user candidate pool with CatBoost.

        If the fitted preprocessing pipeline has no usable feature columns, the
        method falls back to popularity-based ordering instead of failing.
        """
        if candidate_pairs_df.empty:
            return candidate_pairs_df.copy()

        user_id = str(candidate_pairs_df[self.preprocessor.user_id_col].iloc[0])
        merged_pairs_df, feature_matrix = build_inference_features(
            preprocessor=self.preprocessor,
            user_id=user_id,
            item_ids=candidate_pairs_df[self.preprocessor.item_id_col].astype(str).tolist(),
            user_features_df=self.user_features_df,
            item_features_df=self.item_features_df,
            user_features=user_context,
        )
        scored_df = merged_pairs_df.merge(
            candidate_pairs_df.drop(columns=[self.preprocessor.user_id_col, self.preprocessor.item_id_col]),
            left_index=True,
            right_index=True,
            how="left",
        )
 
        if feature_matrix.shape[1] == 0:
            scored_df["regressor_score"] = scored_df.get("popularity", pd.Series(0.0, index=scored_df.index))
            scored_df["final_score"] = scored_df["regressor_score"].astype(float)
            scored_df["scoring_strategy"] = "popularity_fallback_no_features"
            return scored_df

        scored_df["regressor_score"] = self.regressor_model.predict(feature_matrix)
        scored_df["final_score"] = scored_df["regressor_score"].astype(float)
        scored_df["scoring_strategy"] = "catboost_regressor"
        return scored_df

    def recommend(
        self,
        user_id: str,
        top_n_popular: int,
        top_k_diverse: int,
        top_m: int = 10,
        user_context: dict[str, object] | None = None,
    ) -> ColdUserRecommendationResult:
        """
        Run the full cold-user recommendation flow.
        """
        candidate_pairs_df, popular_result, diverse_result = self.build_candidate_pool(
            user_id=user_id,
            top_n_popular=top_n_popular,
            top_k_diverse=top_k_diverse,
        )
        scored_candidates_df = self.score_candidates(
            candidate_pairs_df=candidate_pairs_df,
            user_context=user_context,
        )

        if scored_candidates_df.empty:
            recommendations_df = scored_candidates_df.copy()
        else:
            sort_columns = ["final_score"]
            sort_ascending = [False]
            if "popularity" in scored_candidates_df.columns:
                sort_columns.append("popularity")
                sort_ascending.append(False)
            recommendations_df = (
                scored_candidates_df.sort_values(sort_columns, ascending=sort_ascending)
                .head(max(top_m, 0))
                .reset_index(drop=True)
            )
            recommendations_df["rank"] = np.arange(1, len(recommendations_df) + 1)

        return ColdUserRecommendationResult(
            user_id=str(user_id),
            candidate_pairs_df=candidate_pairs_df,
            scored_candidates_df=scored_candidates_df,
            recommendations_df=recommendations_df,
            popular_selection=popular_result,
            diverse_selection=diverse_result,
            strategy="popular_maxvol_catboost_cold_user",
        )
