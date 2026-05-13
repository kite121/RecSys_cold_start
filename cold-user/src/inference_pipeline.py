from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.cold_user_recommender import ColdUserRecommendationResult
from src.hybrid_recommender import HybridRecommender
from src.utils import ensure_parent_dir


def save_dataframe(df: pd.DataFrame, output_path: str | Path) -> Path:
    """
    Save a dataframe to CSV and return the normalized output path.
    """
    normalized_path = ensure_parent_dir(output_path)
    df.to_csv(normalized_path, index=False)
    return normalized_path


def resolve_recommendation_output(
    recommendation_result: pd.DataFrame | ColdUserRecommendationResult,
) -> tuple[pd.DataFrame, dict]:
    """
    Normalize warm-user and cold-user outputs to a common dataframe + metadata form.
    """
    if isinstance(recommendation_result, ColdUserRecommendationResult):
        recommendations_df = recommendation_result.recommendations_df.copy()
        metadata = {
            "route": "cold_user",
            "strategy": recommendation_result.strategy,
            "num_candidate_pairs": int(len(recommendation_result.candidate_pairs_df)),
            "num_scored_candidates": int(len(recommendation_result.scored_candidates_df)),
        }
        return recommendations_df, metadata

    recommendations_df = recommendation_result.copy()
    metadata = {
        "route": "warm_user_or_global_cold_start",
        "strategy": "hybrid_router",
        "num_candidate_pairs": None,
        "num_scored_candidates": int(len(recommendations_df)),
    }
    return recommendations_df, metadata


def run_inference(
    model_path: str | Path,
    user_id: str,
    user_features: dict[str, object] | None = None,
    top_k: int = 10,
    recommendations_output_path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Run the main inference flow by user_id.

    Steps:
    1. load the fitted HybridRecommender
    2. route the user to warm-user, cold-user or global cold-start flow
    3. collect final recommendations
    4. optionally save recommendations to CSV
    """
    hybrid_model = HybridRecommender.load(model_path)
    recommendation_result = hybrid_model.recommend_for_user(
        user_id=str(user_id),
        user_context=user_features,
        top_k=top_k,
    )
    recommendations_df, route_metadata = resolve_recommendation_output(recommendation_result)

    if recommendations_output_path is not None:
        save_dataframe(recommendations_df, recommendations_output_path)

    inference_summary = {
        "model_path": str(model_path),
        "user_id": str(user_id),
        "mode": hybrid_model.mode,
        "top_k": int(top_k),
        "num_recommendations": int(len(recommendations_df)),
        "recommendations_output_path": (
            str(recommendations_output_path) if recommendations_output_path is not None else None
        ),
        **route_metadata,
    }
    return recommendations_df, inference_summary
