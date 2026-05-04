from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
from scipy import sparse


def ensure_feature_matrix(X: sparse.spmatrix | np.ndarray) -> sparse.spmatrix | np.ndarray:
    """
    Validate the feature matrix passed to CatBoost.
    """
    if X.shape[0] == 0:
        raise ValueError("X must contain at least one row.")
    if X.shape[1] == 0:
        raise ValueError("X must contain at least one feature column.")
    return X


def ensure_regression_target(y: np.ndarray) -> np.ndarray:
    """
    Validate and normalize regression target values.
    """
    y_array = np.asarray(y, dtype=float)
    if y_array.ndim != 1:
        raise ValueError("y must be a 1-dimensional array.")
    if y_array.shape[0] == 0:
        raise ValueError("y must contain at least one element.")
    return y_array


@dataclass
class CatBoostRegressorModel:
    """
    Feature model that learns to predict ALS scores from user/item features.

    This matches the new hybrid idea:
    - ALS is the teacher on warm items
    - CatBoostRegressor learns to approximate ALS scores
    - at inference, ALS handles warm items and CatBoost handles cold items
    """

    iterations: int = 300
    learning_rate: float = 0.05
    depth: int = 5
    loss_function: str = "RMSE"
    random_seed: int = 42

    model: object | None = None
    feature_names: list[str] = field(default_factory=list)

    def fit(
        self,
        X: sparse.spmatrix | np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> "CatBoostRegressorModel":
        """
        Train CatBoostRegressor on prepared pair features with target = ALS score.
        """
        try:
            from catboost import CatBoostRegressor, Pool
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "The `catboost` package is required to train CatBoostRegressor. Install requirements first."
            ) from exc

        X = ensure_feature_matrix(X)
        y_array = ensure_regression_target(y)
        if X.shape[0] != y_array.shape[0]:
            raise ValueError("X and y must contain the same number of rows.")

        self.feature_names = feature_names or []
        train_pool = Pool(
            data=X,
            label=y_array,
            feature_names=self.feature_names or None,
        )
        self.model = CatBoostRegressor(
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            loss_function=self.loss_function,
            random_seed=self.random_seed,
            verbose=False,
        )
        self.model.fit(train_pool)
        return self

    def predict(self, X: sparse.spmatrix | np.ndarray) -> np.ndarray:
        """
        Predict ALS-like scores for a prepared feature matrix.
        """
        if self.model is None:
            raise RuntimeError("CatBoostRegressor is not fitted.")
        X = ensure_feature_matrix(X)
        return np.asarray(self.model.predict(X), dtype=float)

    def save(self, path: str | Path) -> None:
        """
        Save the fitted regressor wrapper.
        """
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "CatBoostRegressorModel":
        """
        Load a saved regressor wrapper.
        """
        return joblib.load(path)

