from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    user_id_col: str | None = None
    item_id_col: str | None = None
    value_col: str | None = None
    user_prefix: str | None = None
    item_prefix: str | None = None
    csv_sep: str | None = None
    csv_encoding: str | None = None


class WarmColdConfig(BaseModel):
    min_warm_interactions: int | None = Field(default=None, ge=1)
    popularity_metric: str | None = None


class ALSConfig(BaseModel):
    factors: int | None = Field(default=None, ge=1)
    regularization: float | None = Field(default=None, ge=0)
    iterations: int | None = Field(default=None, ge=1)
    alpha: float | None = Field(default=None, ge=0)
    random_state: int | None = None


class RetrievalConfig(BaseModel):
    top_n_popular: int | None = Field(default=None, ge=1)
    top_k_diverse: int | None = Field(default=None, ge=1)
    top_m_neighbors: int | None = Field(default=None, ge=1)
    warm_candidates_per_user: int | None = Field(default=None, ge=1)
    cold_candidates_per_user: int | None = Field(default=None, ge=0)
    final_candidate_pool_size: int | None = Field(default=None, ge=1)
    exclude_seen_items: bool | None = None
    similarity_metric: str | None = None


class RankerConfig(BaseModel):
    iterations: int | None = Field(default=None, ge=1)
    learning_rate: float | None = Field(default=None, gt=0)
    depth: int | None = Field(default=None, ge=1)
    loss_function: str | None = None
    random_seed: int | None = None
    negative_samples_per_user: int | None = Field(default=None, ge=0)


class TrainConfig(BaseModel):

    artifacts_dir: str | None = None
    data: DataConfig | None = None
    warm_cold: WarmColdConfig | None = None
    als: ALSConfig | None = None
    retrieval: RetrievalConfig | None = None
    ranker: RankerConfig | None = None
    use_interaction_value_as_label: bool | None = None


class TrainResponse(BaseModel):
    status: Literal["ok", "error"]
    model_id: UUID
    summary: dict[str, Any]


class PredictRequest(BaseModel):
    model_id: UUID
    user_id: str | int
    top_k: int | None = Field(default=None, ge=1)
    user_context: dict[str, object] | None = None
    warm_candidate_item_ids: list[str] | None = None
    cold_candidate_item_ids: list[str] | None = None


class RecommendationItem(BaseModel):
    item_id: str
    score: float | None = None
    rank: int | None = Field(default=None, ge=1)


class PredictResponse(BaseModel):
    status: Literal["ok", "error"]
    model_id: UUID
    user_id: str
    top_k: int
    recommendations: list[RecommendationItem]
    summary: dict[str, Any] | None = None

