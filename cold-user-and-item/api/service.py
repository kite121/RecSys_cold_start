from __future__ import annotations

import math

import pandas as pd

from api.schemas import RecommendationItem, RecommendationRequest, RecommendationResponse
from src.hybrid_recommender import HybridRecommender


def _coerce_float(value: object, default: float = 0.0) -> float:
    """
    Convert a value to float while handling NaN-like cases safely.
    """
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return float(default)
    if math.isnan(converted):
        return float(default)
    return converted


def _coerce_int(value: object, default: int) -> int:
    """
    Convert a value to int with a fallback value.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def build_recommendation_items(recommendations_df: pd.DataFrame) -> list[RecommendationItem]:
    """
    Convert the internal recommendation dataframe into API response items.
    """
    if recommendations_df.empty:
        return []

    item_column = "item_id" if "item_id" in recommendations_df.columns else recommendations_df.columns[1]
    items: list[RecommendationItem] = []
    for default_rank, row in enumerate(recommendations_df.to_dict(orient="records"), start=1):
        items.append(
            RecommendationItem(
                item_id=str(row.get(item_column, "")),
                score=_coerce_float(row.get("score", row.get("final_score", 0.0))),
                rank=_coerce_int(row.get("rank"), default=default_rank),
                source=str(row.get("source", row.get("retrieval_source", "unknown"))),
            )
        )
    return items


def recommend_with_model(
    model: HybridRecommender,
    request: RecommendationRequest,
) -> RecommendationResponse:
    """
    Execute the current hybrid recommender flow and convert it to an API response.
    """
    recommendation_result = model.recommend_for_user(
        user_id=request.user_id,
        user_context=request.user_features,
        top_k=request.top_k,
    )

    if hasattr(recommendation_result, "recommendations_df"):
        recommendations_df = recommendation_result.recommendations_df.copy()
        route = "cold_user"
        strategy = str(getattr(recommendation_result, "strategy", "cold_user_flow"))
    else:
        recommendations_df = recommendation_result.copy()
        route = "warm_user_or_global_cold_start"
        strategy = "hybrid_router"

    recommendation_items = build_recommendation_items(recommendations_df)
    return RecommendationResponse(
        user_id=str(request.user_id),
        mode=str(getattr(model, "mode", "unknown")),
        route=route,
        strategy=strategy,
        num_recommendations=len(recommendation_items),
        recommendations=recommendation_items,
    )
