from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    """
    Response schema for the liveness endpoint.
    """

    model_config = ConfigDict(extra="forbid")

    status: str
    service: str
    timestamp_utc: str


class ReadinessResponse(BaseModel):
    """
    Response schema for the readiness endpoint.
    """

    model_config = ConfigDict(extra="forbid")

    status: str
    model_loaded: bool
    model_path: str | None = None


class RecommendationRequest(BaseModel):
    """
    Input schema for recommendation requests.
    """

    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(..., min_length=1, description="Target user identifier.")
    user_features: dict[str, Any] | None = Field(
        default=None,
        description="Optional user-level feature values for cold-user inference.",
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=1000,
        description="Number of recommendations to return.",
    )


class RecommendationItem(BaseModel):
    """
    One recommendation row returned by the API.
    """

    model_config = ConfigDict(extra="forbid")

    item_id: str
    score: float
    rank: int
    source: str


class RecommendationResponse(BaseModel):
    """
    Output schema for recommendation requests.
    """

    model_config = ConfigDict(extra="forbid")

    user_id: str
    mode: str
    route: str
    strategy: str
    num_recommendations: int
    recommendations: list[RecommendationItem]
