from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse

@dataclass
class ALSArtifacts:
    """
    Sparse interaction matrix and identifier mappings used by ALS.
    """

    user_item_matrix: sparse.csr_matrix
    user2idx: dict[str, int]
    item2idx: dict[str, int]
    idx2user: dict[int, str]
    idx2item: dict[int, str]


def ensure_columns_present(df: pd.DataFrame, required_columns: list[str] | tuple[str, ...]) -> None:
    """
    Validate that all required columns exist in the dataframe.
    """
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")


@dataclass
class ALSRecommender:
    """
    Wrapper around ``implicit`` ALS.

    The model is trained on the user-item interaction matrix built from the
    input training dataframe.
    """

    user_col: str = "user_id"
    item_col: str = "item_id"
    value_col: str = "value"
    factors: int = 20
    regularization: float = 0.01
    iterations: int = 150
    alpha: float = 20.0
    random_state: int = 42

    model: object | None = None
    artifacts: ALSArtifacts | None = None
    seen_items_by_user: dict[str, set[str]] = field(default_factory=dict)

    def prepare_interactions(self, interactions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate ALS input before matrix construction.

        This implementation follows the same general rule as ``recommender 2``:
        interaction values are converted to numeric, but zero values are not
        removed before building the sparse matrix.
        """
        ensure_columns_present(interactions_df, [self.user_col, self.item_col, self.value_col])

        prepared_df = interactions_df[[self.user_col, self.item_col, self.value_col]].copy()
        prepared_df[self.user_col] = prepared_df[self.user_col].astype(str)
        prepared_df[self.item_col] = prepared_df[self.item_col].astype(str)
        prepared_df[self.value_col] = pd.to_numeric(prepared_df[self.value_col], errors="coerce").fillna(0.0)
        prepared_df = prepared_df.dropna(subset=[self.user_col, self.item_col])
        return prepared_df.reset_index(drop=True)

    def build_interaction_matrix(self, interactions_df: pd.DataFrame) -> ALSArtifacts:
        """
        Build a sparse user-item matrix and id mappings.
        """
        prepared_df = self.prepare_interactions(interactions_df)
        if prepared_df.empty:
            raise ValueError("ALS requires at least one interaction row after preprocessing.")

        grouped_df = prepared_df.groupby([self.user_col, self.item_col], as_index=False)[self.value_col].sum()

        user_ids = grouped_df[self.user_col].drop_duplicates().tolist()
        item_ids = grouped_df[self.item_col].drop_duplicates().tolist()
        user2idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
        item2idx = {item_id: idx for idx, item_id in enumerate(item_ids)}
        idx2user = {idx: user_id for user_id, idx in user2idx.items()}
        idx2item = {idx: item_id for item_id, idx in item2idx.items()}

        row_idx = grouped_df[self.user_col].map(user2idx).to_numpy()
        col_idx = grouped_df[self.item_col].map(item2idx).to_numpy()
        values = grouped_df[self.value_col].astype(float).to_numpy()

        matrix = sparse.csr_matrix((values, (row_idx, col_idx)), shape=(len(user2idx), len(item2idx)))
        return ALSArtifacts(
            user_item_matrix=matrix,
            user2idx=user2idx,
            item2idx=item2idx,
            idx2user=idx2user,
            idx2item=idx2item,
        )

    def fit(self, interactions_df: pd.DataFrame) -> "ALSRecommender":
        """
        Train ALS on the input interactions dataframe.
        """
        try:
            from implicit.als import AlternatingLeastSquares  # type: ignore
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "The `implicit` package is required to train ALS. Install requirements first."
            ) from exc

        prepared_df = self.prepare_interactions(interactions_df)
        self.artifacts = self.build_interaction_matrix(prepared_df)
        self.seen_items_by_user = (
            prepared_df.groupby(self.user_col)[self.item_col]
            .agg(lambda values: set(pd.unique(values.astype(str))))
            .to_dict()
        )

        self.model = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            random_state=self.random_state,
            dtype=np.float32,
        )
        item_user_matrix = (self.artifacts.user_item_matrix.T * self.alpha).tocsr()
        self.model.fit(item_user_matrix)
        return self

    def score(self, user_id: str, item_id: str) -> float | None:
        """
        Return ALS score for a user-item pair, or ``None`` if ids are unseen.
        """
        if self.model is None or self.artifacts is None:
            raise RuntimeError("ALS model is not fitted.")

        user_idx = self.artifacts.user2idx.get(user_id)
        item_idx = self.artifacts.item2idx.get(item_id)
        if user_idx is None or item_idx is None:
            return None

        user_vector = np.asarray(self.model.user_factors[user_idx], dtype=np.float32)
        item_vector = np.asarray(self.model.item_factors[item_idx], dtype=np.float32)
        return float(np.dot(user_vector, item_vector))

    def score_pairs(self, pairs_df: pd.DataFrame) -> pd.Series:
        """
        Score many pairs at once and return a pandas Series aligned with the input order.
        """
        ensure_columns_present(pairs_df, [self.user_col, self.item_col])
        scores = [
            self.score(str(row[self.user_col]), str(row[self.item_col])) or 0.0
            for _, row in pairs_df.iterrows()
        ]
        return pd.Series(scores, index=pairs_df.index, name="als_score")

    def recommend(
        self,
        user_id: str,
        candidate_item_ids: list[str] | None = None,
        top_k: int = 10,
        exclude_seen: bool = True,
    ) -> list[tuple[str, float]]:
        """
        Recommend items for a user.

        If ``candidate_item_ids`` is provided, ALS scores only that candidate pool.
        Otherwise the method uses the full catalog through ``implicit.recommend``.
        """
        if self.model is None or self.artifacts is None:
            raise RuntimeError("ALS model is not fitted.")

        user_idx = self.artifacts.user2idx.get(user_id)
        if user_idx is None:
            return []

        if candidate_item_ids is None:
            item_indices, scores = self.model.recommend(
                userid=user_idx,
                user_items=self.artifacts.user_item_matrix,
                N=top_k,
                filter_already_liked_items=exclude_seen,
            )
            return [
                (self.artifacts.idx2item[int(item_idx)], float(score))
                for item_idx, score in zip(item_indices, scores)
            ]

        known_candidates = [str(item_id) for item_id in candidate_item_ids if str(item_id) in self.artifacts.item2idx]
        if not known_candidates:
            return []

        candidate_scores = []
        for item_id in known_candidates:
            if exclude_seen and item_id in self.seen_items_by_user.get(user_id, set()):
                continue
            score = self.score(user_id, item_id)
            if score is not None:
                candidate_scores.append((item_id, score))
        candidate_scores.sort(key=lambda pair: pair[1], reverse=True)
        return candidate_scores[:top_k]

    def save(self, path: str | Path) -> None:
        """
        Save the fitted ALS wrapper.
        """
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "ALSRecommender":
        """
        Load a saved ALS wrapper.
        """
        return joblib.load(path)
