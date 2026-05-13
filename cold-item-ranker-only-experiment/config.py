from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class PathConfig:
    """
    Filesystem layout for the new cold-item implementation.
    """

    project_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    artifacts_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "artifacts")
    baseline_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "cold-item-baseline")

    als_model_name: str = "als_model.joblib"
    ranker_model_name: str = "ranker_model.joblib"
    project_config_name: str = "project_config.joblib"
    preprocessor_name: str = "preprocessor.joblib"
    preprocessing_artifacts_name: str = "preprocessing_artifacts.joblib"
    warm_cold_split_name: str = "warm_cold_split.joblib"
    support_items_name: str = "support_items.joblib"
    cold_neighbors_name: str = "cold_neighbors.joblib"
    cold_vectors_name: str = "cold_vectors.joblib"

    def ensure_artifacts_dir(self) -> Path:
        """
        Create the artifacts directory if it does not exist.
        """
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        return self.artifacts_dir

    @property
    def als_model_path(self) -> Path:
        return self.artifacts_dir / self.als_model_name

    @property
    def ranker_model_path(self) -> Path:
        return self.artifacts_dir / self.ranker_model_name

    @property
    def project_config_path(self) -> Path:
        return self.artifacts_dir / self.project_config_name

    @property
    def preprocessor_path(self) -> Path:
        return self.artifacts_dir / self.preprocessor_name

    @property
    def preprocessing_artifacts_path(self) -> Path:
        return self.artifacts_dir / self.preprocessing_artifacts_name

    @property
    def warm_cold_split_path(self) -> Path:
        return self.artifacts_dir / self.warm_cold_split_name

    @property
    def support_items_path(self) -> Path:
        return self.artifacts_dir / self.support_items_name

    @property
    def cold_neighbors_path(self) -> Path:
        return self.artifacts_dir / self.cold_neighbors_name

    @property
    def cold_vectors_path(self) -> Path:
        return self.artifacts_dir / self.cold_vectors_name


@dataclass(slots=True)
class DataConfig:
    """
    Input schema configuration for train and inference datasets.
    """

    user_id_col: str = "user_id"
    item_id_col: str = "item_id"
    value_col: str = "value"
    user_prefix: str = "user_"
    item_prefix: str = "item_"
    csv_sep: str = ","
    csv_encoding: str = "utf-8"

    @property
    def required_train_columns(self) -> tuple[str, str, str]:
        return (self.user_id_col, self.item_id_col, self.value_col)

    @property
    def required_infer_columns(self) -> tuple[str, str]:
        return (self.user_id_col, self.item_id_col)


@dataclass(slots=True)
class WarmColdConfig:
    """
    Rules for warm/cold item splitting.
    """

    min_warm_interactions: int = 5
    popularity_metric: str = "count"


@dataclass(slots=True)
class ALSConfig:
    """
    Hyperparameters for ALS retrieval.
    """

    factors: int = 64
    regularization: float = 0.01
    iterations: int = 100
    alpha: float = 20.0
    random_state: int = 42


@dataclass(slots=True)
class RetrievalConfig:
    """
    Parameters for support-set construction and candidate generation.
    """

    top_n_popular: int = 5000
    top_k_diverse: int = 500
    top_m_neighbors: int = 20
    warm_candidates_per_user: int = 200
    cold_candidates_per_user: int = 200
    final_candidate_pool_size: int = 400
    exclude_seen_items: bool = True
    similarity_metric: str = "cosine"


@dataclass(slots=True)
class RankerConfig:
    """
    Hyperparameters for CatBoostRanker.
    """

    iterations: int = 300
    learning_rate: float = 0.05
    depth: int = 6
    loss_function: str = "YetiRankPairwise"
    random_seed: int = 42
    negative_samples_per_user: int = 3


@dataclass(slots=True)
class InferenceConfig:
    """
    Default inference-time parameters.
    """

    top_k: int = 10


@dataclass(slots=True)
class ColdItemProjectConfig:
    """
    Root config object for the new cold-item pipeline.
    """

    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    warm_cold: WarmColdConfig = field(default_factory=WarmColdConfig)
    als: ALSConfig = field(default_factory=ALSConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    ranker: RankerConfig = field(default_factory=RankerConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)


DEFAULT_CONFIG = ColdItemProjectConfig()
