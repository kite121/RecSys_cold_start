from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import ValidationError

from app.schemas import PredictRequest, PredictResponse, RecommendationItem
from config import PathConfig
from src.inference_pipeline import run_cold_item_inference
from src.utils import load_project_config

router = APIRouter(tags=["predict"])


@router.post("/predict")
def predict_endpoint(body: dict) -> PredictResponse:
    try:
        payload = PredictRequest.model_validate(body)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc

    project_dir = Path(__file__).resolve().parents[2]
    artifacts_dir = (project_dir / "artifacts" / str(payload.model_id)).resolve()

    config_path = artifacts_dir / PathConfig().project_config_name
    try:
        config = load_project_config(config_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Model artifacts not found: {payload.model_id}") from exc

    config.paths = PathConfig(
        project_dir=project_dir,
        artifacts_dir=artifacts_dir,
        baseline_dir=project_dir.parent / "cold-item-baseline",
    )

    try:
        result = run_cold_item_inference(
            user_id=str(payload.user_id),
            config=config,
            top_k=payload.top_k,
            user_context=payload.user_context,
            warm_candidate_item_ids=payload.warm_candidate_item_ids,
            cold_candidate_item_ids=payload.cold_candidate_item_ids,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    recs: list[RecommendationItem] = []
    rec_df = result.recommendations_df
    if not rec_df.empty:
        item_col = config.data.item_id_col
        score_col = "score"
        rank_col = "rank"
        for _, row in rec_df.iterrows():
            recs.append(
                RecommendationItem(
                    item_id=str(row[item_col]) if item_col in rec_df.columns else str(row.get("item_id")),
                    score=float(row[score_col]) if score_col in rec_df.columns else None,
                    rank=int(row[rank_col]) if rank_col in rec_df.columns and row[rank_col] is not None else None,
                )
            )

    return PredictResponse(
        status="ok",
        model_id=payload.model_id,
        user_id=str(payload.user_id),
        top_k=int(result.inference_summary.get("top_k", payload.top_k or config.inference.top_k)),
        recommendations=recs,
        summary=dict(result.inference_summary),
    )