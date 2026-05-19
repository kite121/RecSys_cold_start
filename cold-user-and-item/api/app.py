from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI

from config import DEFAULT_CONFIG

from .model_store import model_store
from .routes import router as api_router


def build_lifespan(model_path: str | None = None):
    """
    Build an application lifespan handler.

    The actual recommender model loading will be added in the API model-store
    layer. For now we initialize shared app state so health and readiness
    endpoints already have predictable behaviour.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.started_at_utc = datetime.now(timezone.utc)
        resolved_model_path = model_path or str(DEFAULT_CONFIG.paths.default_model_output_path)
        loaded_model = model_store.load_model(resolved_model_path)
        app.state.model_store = model_store
        app.state.model_path = model_store.model_path
        app.state.model = loaded_model
        app.state.project_mode = DEFAULT_CONFIG.modes.hybrid_mode
        try:
            yield
        finally:
            model_store.clear()
            app.state.model = None

    return lifespan


def create_app(model_path: str | None = None) -> FastAPI:
    """
    Create and configure the FastAPI application.
    """
    app = FastAPI(
        title="Cold-User Hybrid Recommender API",
        version="0.1.0",
        description=(
            "HTTP interface for the cold-user hybrid recommender. "
            "Supports warm-user, cold-user and global cold-start inference modes."
        ),
        lifespan=build_lifespan(model_path=model_path),
    )
    app.include_router(api_router)
    return app


app = create_app()
