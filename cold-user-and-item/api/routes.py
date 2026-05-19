from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request

from .model_store import model_store
from .schemas import (
    HealthResponse,
    ReadinessResponse,
    RecommendationRequest,
    RecommendationResponse,
)
from .service import recommend_with_model

router = APIRouter(tags=["service"])


@router.get("/health", response_model=HealthResponse)
def healthcheck() -> HealthResponse:
    """
    Liveness probe for the API process.
    """
    return HealthResponse(
        status="ok",
        service="cold-user-api",
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )


@router.get("/ready", response_model=ReadinessResponse)
def readiness(request: Request) -> ReadinessResponse:
    """
    Readiness probe for the API application state.
    """
    app_model_store = getattr(request.app.state, "model_store", model_store)
    status = app_model_store.get_status()
    return ReadinessResponse(
        status="ready" if bool(status.get("model_loaded")) else "not_ready",
        model_loaded=bool(status.get("model_loaded")),
        model_path=str(status.get("model_path")) if status.get("model_path") is not None else None,
    )


@router.post("/recommend", response_model=RecommendationResponse)
def recommend(request: Request, payload: RecommendationRequest) -> RecommendationResponse:
    """
    Main recommendation endpoint.
    """
    app_model_store = getattr(request.app.state, "model_store", model_store)
    if not app_model_store.is_model_loaded():
        raise HTTPException(status_code=503, detail="Recommender model is not loaded.")

    try:
        recommender_model = app_model_store.get_model()
        return recommend_with_model(recommender_model, payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
