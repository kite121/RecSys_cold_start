from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data_loader import load_csv_data
from src.hybrid_recommender import HybridRecommender
from src.train_pipeline import ensure_parent_dir


def save_dataframe(df: pd.DataFrame, output_path: str | Path) -> Path:
    """
    Save a dataframe to CSV and return the normalized output path.
    """
    normalized_path = ensure_parent_dir(output_path)
    df.to_csv(normalized_path, index=False)
    return normalized_path


def run_inference(
    model_path: str | Path,
    candidate_pairs_csv_path: str | Path,
    top_k: int = 10,
    scored_output_path: str | Path | None = None,
    recommendations_output_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Run inference with a trained HybridRecommender.

    Steps:
    1. load the fitted hybrid model
    2. load candidate user-item pairs from CSV
    3. compute ALS and CatBoostRegressor scores
    4. take the maximum of both scores
    5. optionally save scored pairs and top-k recommendations
    """
    hybrid_model = HybridRecommender.load(model_path)
    candidate_pairs_df = load_csv_data(
        csv_path=candidate_pairs_csv_path,
        required_columns=[
            hybrid_model.preprocessor.user_id_col,
            hybrid_model.preprocessor.item_id_col,
        ],
    )

    scored_df = hybrid_model.predict(candidate_pairs_df)
    recommendations_df = hybrid_model.recommend(candidate_pairs_df, top_k=top_k)

    if scored_output_path is not None:
        save_dataframe(scored_df, scored_output_path)
    if recommendations_output_path is not None:
        save_dataframe(recommendations_df, recommendations_output_path)

    inference_summary = {
        "model_path": str(model_path),
        "candidate_pairs_csv_path": str(candidate_pairs_csv_path),
        "num_candidate_pairs": int(len(scored_df)),
        "num_users": int(scored_df[hybrid_model.preprocessor.user_id_col].nunique()),
        "top_k": int(top_k),
        "scored_output_path": str(scored_output_path) if scored_output_path is not None else None,
        "recommendations_output_path": (
            str(recommendations_output_path) if recommendations_output_path is not None else None
        ),
    }
    return scored_df, recommendations_df, inference_summary
