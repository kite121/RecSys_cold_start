from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class PathConfig:
    """
    Filesystem paths used by the cold-user project.
    """

    project_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    artifacts_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "artifacts")
    default_model_output_path: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent / "artifacts" / "hybrid_model.joblib"
    )
    default_recommendations_output_path: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent / "outputs" / "recommendations.csv"
    )


@dataclass(slots=True)
class DataConfig:
    """
    Input schema configuration for train and inference.
    """

    user_id_col: str = "user_id"
    item_id_col: str = "item_id"
    value_col: str = "value"
    user_prefix: str = "user_"
    item_prefix: str = "item_"
    csv_sep: str = ","
    csv_encoding: str = "utf-8"


@dataclass(slots=True)
class SplitConfig:
    """
    Warm/cold split thresholds and popularity mode.
    """

    min_user_interactions: int = 5
    min_item_interactions: int = 5
    popularity_metric: str = "count"


@dataclass(slots=True)
class ALSConfig:
    """
    Hyperparameters for ALS.
    """

    factors: int = 20
    regularization: float = 0.01
    iterations: int = 150
    alpha: float = 20.0
    random_state: int = 42


@dataclass(slots=True)
class RegressorConfig:
    """
    Hyperparameters for CatBoostRegressor.
    """

    iterations: int = 300
    learning_rate: float = 0.05
    depth: int = 5
    loss_function: str = "RMSE"
    random_seed: int = 42
    negative_samples_per_user: int = 3


@dataclass(slots=True)
class CandidateConfig:
    """
    Parameters for fallback candidate generation.
    """

    top_n_popular: int = 500
    top_k_diverse: int = 100
    exclude_seen_items: bool = True


@dataclass(slots=True)
class InferenceConfig:
    """
    Default inference-time parameters.
    """

    default_top_k: int = 10
    cold_top_n_popular: int = 200
    cold_top_k_diverse: int = 50


@dataclass(slots=True)
class ModeConfig:
    """
    Named runtime modes used by the recommender.
    """

    hybrid_mode: str = "hybrid"
    global_cold_start_mode: str = "global_cold_start"


@dataclass(slots=True)
class ColdUserProjectConfig:
    """
    Root config object for the current cold-user project.
    """

    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    als: ALSConfig = field(default_factory=ALSConfig)
    regressor: RegressorConfig = field(default_factory=RegressorConfig)
    candidate: CandidateConfig = field(default_factory=CandidateConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    modes: ModeConfig = field(default_factory=ModeConfig)


DEFAULT_CONFIG = ColdUserProjectConfig()
