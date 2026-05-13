from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse

from src.feature_builder import RankerDataset


@dataclass(slots=True)
class RankerPredictionResult:
    """
    Ranking-stage output for one prepared dataset.
    """

    scored_pairs_df: pd.DataFrame
    scores: np.ndarray


def ensure_feature_matrix(feature_matrix: sparse.spmatrix | np.ndarray) -> sparse.spmatrix | np.ndarray:
    """
    Validate the feature matrix passed to CatBoost.
    """
    if feature_matrix.shape[0] == 0:
        raise ValueError("Feature matrix must contain at least one row.")
    if feature_matrix.shape[1] == 0:
        raise ValueError("Feature matrix must contain at least one feature column.")
    return feature_matrix


def ensure_ranker_labels(labels: np.ndarray | None) -> np.ndarray:
    """
    Validate and normalize training labels for the ranker.
    """
    if labels is None:
        raise ValueError("Ranker training requires labels, but dataset.labels is None.")

    label_array = np.asarray(labels, dtype=np.float32)
    if label_array.ndim != 1:
        raise ValueError("Ranker labels must be a 1-dimensional array.")
    if label_array.shape[0] == 0:
        raise ValueError("Ranker labels must contain at least one value.")
    return label_array


def ensure_group_ids(group_ids: np.ndarray, expected_length: int) -> np.ndarray:
    """
    Validate ranker group ids.
    """
    group_id_array = np.asarray(group_ids, dtype=np.int64)
    if group_id_array.ndim != 1:
        raise ValueError("group_ids must be a 1-dimensional array.")
    if group_id_array.shape[0] != expected_length:
        raise ValueError("group_ids length must match the number of dataset rows.")
    return group_id_array


def ensure_training_signal(labels: np.ndarray, group_ids: np.ndarray) -> None:
    """
    Fail early when the ranker dataset has no useful ranking signal.
    """
    if np.all(labels <= 0):
        raise ValueError("Ranker dataset contains no positive labels.")

    group_sizes = pd.Series(group_ids).value_counts(sort=False)
    if int((group_sizes >= 2).sum()) == 0:
        raise ValueError("Ranker training requires at least one group with two or more candidates.")


def build_ranker_pool(
    dataset: RankerDataset,
    labels: np.ndarray | None = None,
):
    """
    Build a CatBoost Pool with query-group information.
    """
    try:
        from catboost import Pool
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The `catboost` package is required for CatBoostRanker. Install project requirements first."
        ) from exc

    feature_matrix = ensure_feature_matrix(dataset.feature_matrix)
    group_ids = ensure_group_ids(dataset.group_ids, expected_length=feature_matrix.shape[0])

    pool_kwargs: dict[str, object] = {
        "data": feature_matrix,
        "group_id": group_ids,
    }
    if dataset.feature_names:
        pool_kwargs["feature_names"] = dataset.feature_names
    if labels is not None:
        pool_kwargs["label"] = labels
    return Pool(**pool_kwargs)


@dataclass
class CatBoostItemRanker:
    """
    CatBoost-based ranking model for final candidate ordering.
    """

    iterations: int = 300
    learning_rate: float = 0.05
    depth: int = 6
    loss_function: str = "YetiRankPairwise"
    random_seed: int = 42

    model: object | None = None
    feature_names: list[str] = field(default_factory=list)

    def fit(self, dataset: RankerDataset) -> "CatBoostItemRanker":
        """
        Fit CatBoostRanker on a prepared supervised dataset.
        """
        try:
            from catboost import CatBoostRanker
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "The `catboost` package is required to train CatBoostRanker. Install project requirements first."
            ) from exc

        labels = ensure_ranker_labels(dataset.labels)
        group_ids = ensure_group_ids(dataset.group_ids, expected_length=dataset.feature_matrix.shape[0])
        ensure_training_signal(labels, group_ids)

        self.feature_names = list(dataset.feature_names)
        train_pool = build_ranker_pool(dataset, labels=labels)
        self.model = CatBoostRanker(
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            loss_function=self.loss_function,
            random_seed=self.random_seed,
            verbose=False,
        )
        self.model.fit(train_pool)
        return self

    def predict_scores(self, dataset: RankerDataset) -> np.ndarray:
        """
        Predict ranking scores for prepared candidate pairs.
        """
        if self.model is None:
            raise RuntimeError("CatBoostRanker is not fitted.")

        prediction_pool = build_ranker_pool(dataset, labels=None)
        scores = self.model.predict(prediction_pool)
        return np.asarray(scores, dtype=np.float32)

    def rank_candidates(
        self,
        dataset: RankerDataset,
        top_k: int | None = None,
        score_column: str = "ranker_score",
    ) -> RankerPredictionResult:
        """
        Score candidate pairs and sort them within each user group.
        """
        scores = self.predict_scores(dataset)
        scored_pairs_df = dataset.pairs_df.copy()
        scored_pairs_df["group_id"] = dataset.group_ids
        scored_pairs_df[score_column] = scores
        scored_pairs_df = scored_pairs_df.sort_values(
            by=["group_id", score_column, "retrieval_rank"],
            ascending=[True, False, True],
        ).reset_index(drop=True)

        if top_k is not None and top_k > 0:
            scored_pairs_df = (
                scored_pairs_df.groupby("group_id", sort=False, group_keys=False)
                .head(top_k)
                .reset_index(drop=True)
            )

        return RankerPredictionResult(
            scored_pairs_df=scored_pairs_df,
            scores=scores,
        )

    def save(self, path: str | Path) -> None:
        """
        Save the fitted ranker wrapper to disk.
        """
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "CatBoostItemRanker":
        """
        Load a saved ranker wrapper from disk.
        """
        return joblib.load(path)
