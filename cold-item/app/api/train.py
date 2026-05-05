import tempfile
import traceback
import uuid
from dataclasses import replace
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.schemas import TrainConfig, TrainResponse
from config import (
    ColdItemProjectConfig,
    PathConfig,
)
from src.train_pipeline import train_cold_item_pipeline

router = APIRouter(tags=["train"])

@router.post("/train")
async def train_endpoint(
    train_csv: UploadFile = File(..., description="Training interactions CSV."),
    config: UploadFile | None = File(
        default=None,
        description="Optional JSON config (application/json) matching app/schemas.py::TrainConfig.",
    ),
) -> TrainResponse:
    try:
        overrides = TrainConfig.model_validate_json(await config.read()) if config is not None else TrainConfig()
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    model_id = uuid.uuid4()
    project_dir = Path(__file__).resolve().parents[2]
    artifacts_dir = (
        Path(overrides.artifacts_dir)
        if overrides.artifacts_dir
        else (project_dir / "artifacts" / str(model_id))
    )
    if not artifacts_dir.is_absolute():
        artifacts_dir = project_dir / artifacts_dir

    paths = PathConfig(
        project_dir=project_dir,
        artifacts_dir=artifacts_dir,
        baseline_dir=project_dir.parent / "cold-item-baseline",
    )

    base_config = ColdItemProjectConfig(paths=paths)
    if overrides.data:
        base_config.data = replace(base_config.data, **overrides.data.model_dump(exclude_none=True))
    if overrides.warm_cold:
        base_config.warm_cold = replace(base_config.warm_cold, **overrides.warm_cold.model_dump(exclude_none=True))
    if overrides.als:
        base_config.als = replace(base_config.als, **overrides.als.model_dump(exclude_none=True))
    if overrides.retrieval:
        base_config.retrieval = replace(base_config.retrieval, **overrides.retrieval.model_dump(exclude_none=True))
    if overrides.ranker:
        base_config.ranker = replace(base_config.ranker, **overrides.ranker.model_dump(exclude_none=True))

    use_value_label = bool(overrides.use_interaction_value_as_label) if overrides.use_interaction_value_as_label is not None else False

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / "train.csv"
            tmp_path.write_bytes(await train_csv.read())
            training_result = train_cold_item_pipeline(
                train_csv_path=str(tmp_path),
                config=base_config,
                save_artifacts=True,
                use_interaction_value_as_label=use_value_label,
            )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        ) from exc

    summary = dict[str, object](training_result.training_summary)
    summary["artifacts_dir"] = str(artifacts_dir)
    return TrainResponse(status="ok", model_id=model_id, summary=summary)

