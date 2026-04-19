from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.als_model import ALSRecommender
from src.feature_builder import RegressorTrainingDataset, build_regressor_training_dataset
from src.preprocessing import PreprocessingArtifacts, RecommendationDataPreprocessor
from src.ranker_model import CatBoostRegressorModel
from src.split_warm_cold import WarmColdItemSplitter, WarmColdSplitResult


@dataclass
class HybridRecommender:
    """
    Hybrid recommender that combines ALS and CatBoostRegressor.

    Training:
    - fit preprocessing on the training CSV
    - split items into warm/cold groups
    - train ALS on interaction data
    - train CatBoostRegressor to approximate ALS scores on warm items

    Inference:
    - compute ALS score
    - compute CatBoostRegressor score
    - take the maximum of the two scores
    """

    preprocessor: RecommendationDataPreprocessor
    warm_cold_splitter: WarmColdItemSplitter
    als_model: ALSRecommender
    regressor_model: CatBoostRegressorModel
    negative_samples_per_user: int = 3
    random_state: int = 42

    preprocessing_artifacts: PreprocessingArtifacts | None = None
    warm_cold_result: WarmColdSplitResult | None = None

    def fit(self, train_df: pd.DataFrame) -> "HybridRecommender":
        """
        Train the full hybrid recommender on a training dataframe.
        """
        self.preprocessing_artifacts = self.preprocessor.fit(train_df)
        self.warm_cold_result = self.warm_cold_splitter.split(self.preprocessing_artifacts.interactions_df)
        self.als_model.fit(self.preprocessing_artifacts.interactions_df)

        regressor_dataset: RegressorTrainingDataset = build_regressor_training_dataset(
            interactions_df=self.preprocessing_artifacts.interactions_df,
            user_features_df=self.preprocessing_artifacts.user_features_df,
            item_features_df=self.preprocessing_artifacts.item_features_df,
            warm_cold_split=self.warm_cold_result,
            preprocessor=self.preprocessor,
            als_model=self.als_model,
            negative_samples_per_user=self.negative_samples_per_user,
            random_state=self.random_state,
            user_col=self.preprocessor.user_id_col,
            item_col=self.preprocessor.item_id_col,
        )

        self.regressor_model.fit(
            X=regressor_dataset.X,
            y=regressor_dataset.y.to_numpy(),
        )
        return self

    def predict(self, candidate_pairs_df: pd.DataFrame) -> pd.DataFrame:
        """
        Score candidate user-item pairs with ALS and CatBoostRegressor.
        """
        if self.preprocessing_artifacts is None or self.warm_cold_result is None:
            raise RuntimeError("HybridRecommender must be fitted before prediction.")

        merged_pairs_df, feature_matrix = self.preprocessor.transform_pairs(
            pairs_df=candidate_pairs_df,
            user_features_df=self.preprocessing_artifacts.user_features_df,
            item_features_df=self.preprocessing_artifacts.item_features_df,
        )

        merged_pairs_df["als_score"] = self.als_model.score_pairs(
            merged_pairs_df[[self.preprocessor.user_id_col, self.preprocessor.item_id_col]]
        )
        merged_pairs_df["regressor_score"] = self.regressor_model.predict(feature_matrix)
        merged_pairs_df["item_group"] = merged_pairs_df[self.preprocessor.item_id_col].apply(
            lambda item_id: "warm" if str(item_id) in self.warm_cold_result.warm_items else "cold"
        )
        merged_pairs_df["final_score"] = np.maximum(
            merged_pairs_df["als_score"].to_numpy(dtype=float),
            merged_pairs_df["regressor_score"].to_numpy(dtype=float),
        )
        return merged_pairs_df

    def recommend(self, candidate_pairs_df: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
        """
        Return top-K recommendations per user from candidate pairs.
        """
        scored_df = self.predict(candidate_pairs_df)
        recommendation_df = (
            scored_df.sort_values(
                [self.preprocessor.user_id_col, "final_score"],
                ascending=[True, False],
            )
            .groupby(self.preprocessor.user_id_col, as_index=False)
            .head(top_k)
            .reset_index(drop=True)
        )
        recommendation_df["rank"] = recommendation_df.groupby(self.preprocessor.user_id_col).cumcount() + 1
        return recommendation_df

    def save(self, path: str | Path) -> None:
        """
        Save the full hybrid recommender.
        """
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "HybridRecommender":
        """
        Load a saved hybrid recommender.
        """
        return joblib.load(path)
