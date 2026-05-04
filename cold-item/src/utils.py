from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from config import ColdItemProjectConfig, DEFAULT_CONFIG
from src.als_model import ALSRecommender
from src.cold_vector_builder import ColdVectorBuildResult
from src.preprocessing import PreprocessingArtifacts, RecommendationDataPreprocessor
from src.ranker_model import CatBoostItemRanker
from src.split_warm_cold import WarmColdSplitResult


@dataclass(slots=True)
class LoadedColdItemArtifacts:
    """
    Full set of persisted artifacts needed after training.
    """

    preprocessing_artifacts: PreprocessingArtifacts
    preprocessor: RecommendationDataPreprocessor
    als_model: ALSRecommender
    ranker_model: CatBoostItemRanker
    warm_cold_result: WarmColdSplitResult
    support_selection_result: Any
    neighbors_result: Any
    cold_vector_result: ColdVectorBuildResult


def ensure_parent_dir(path: str | Path) -> Path:
    """
    Ensure the parent directory of a file path exists.
    """
    normalized_path = Path(path)
    normalized_path.parent.mkdir(parents=True, exist_ok=True)
    return normalized_path


def ensure_file_exists(path: str | Path) -> Path:
    """
    Ensure that a file exists before loading it.
    """
    normalized_path = Path(path)
    if not normalized_path.exists():
        raise FileNotFoundError(f"Artifact file not found: {normalized_path}")
    return normalized_path


def dump_joblib_object(obj: object, path: str | Path) -> Path:
    """
    Persist an object with joblib and return the normalized output path.
    """
    normalized_path = ensure_parent_dir(path)
    joblib.dump(obj, normalized_path)
    return normalized_path


def load_joblib_object(path: str | Path) -> object:
    """
    Load a joblib-serialized object from disk.
    """
    return joblib.load(ensure_file_exists(path))


def build_artifact_paths(config: ColdItemProjectConfig | None = None) -> dict[str, Path]:
    """
    Return the canonical artifact paths for the current project config.
    """
    effective_config = config or DEFAULT_CONFIG
    return {
        "artifacts_dir": effective_config.paths.artifacts_dir,
        "als_model_path": effective_config.paths.als_model_path,
        "ranker_model_path": effective_config.paths.ranker_model_path,
        "project_config_path": effective_config.paths.project_config_path,
        "preprocessor_path": effective_config.paths.preprocessor_path,
        "preprocessing_artifacts_path": effective_config.paths.preprocessing_artifacts_path,
        "warm_cold_split_path": effective_config.paths.warm_cold_split_path,
        "support_items_path": effective_config.paths.support_items_path,
        "cold_neighbors_path": effective_config.paths.cold_neighbors_path,
        "cold_vectors_path": effective_config.paths.cold_vectors_path,
    }


def load_project_config(path: str | Path) -> ColdItemProjectConfig:
    """
    Load a persisted project config from disk.
    """
    loaded_config = load_joblib_object(path)
    if not isinstance(loaded_config, ColdItemProjectConfig):
        raise TypeError("Saved project config has an unexpected type.")
    return loaded_config


def load_cold_item_artifacts(config: ColdItemProjectConfig | None = None) -> LoadedColdItemArtifacts:
    """
    Load the full trained cold-item artifact set from disk.
    """
    effective_config = config or DEFAULT_CONFIG

    preprocessing_artifacts = load_joblib_object(effective_config.paths.preprocessing_artifacts_path)
    preprocessor = load_joblib_object(effective_config.paths.preprocessor_path)
    warm_cold_result = load_joblib_object(effective_config.paths.warm_cold_split_path)
    support_selection_result = load_joblib_object(effective_config.paths.support_items_path)
    neighbors_result = load_joblib_object(effective_config.paths.cold_neighbors_path)
    cold_vector_result = load_joblib_object(effective_config.paths.cold_vectors_path)

    if not isinstance(preprocessing_artifacts, PreprocessingArtifacts):
        raise TypeError("Saved preprocessing_artifacts has an unexpected type.")
    if not isinstance(preprocessor, RecommendationDataPreprocessor):
        raise TypeError("Saved preprocessor has an unexpected type.")
    if not isinstance(warm_cold_result, WarmColdSplitResult):
        raise TypeError("Saved warm/cold split artifact has an unexpected type.")
    if not isinstance(cold_vector_result, ColdVectorBuildResult):
        raise TypeError("Saved cold vector artifact has an unexpected type.")

    als_model = ALSRecommender.load(effective_config.paths.als_model_path)
    ranker_model = CatBoostItemRanker.load(effective_config.paths.ranker_model_path)

    return LoadedColdItemArtifacts(
        preprocessing_artifacts=preprocessing_artifacts,
        preprocessor=preprocessor,
        als_model=als_model,
        ranker_model=ranker_model,
        warm_cold_result=warm_cold_result,
        support_selection_result=support_selection_result,
        neighbors_result=neighbors_result,
        cold_vector_result=cold_vector_result,
    )


def summarize_artifact_paths(config: ColdItemProjectConfig | None = None) -> dict[str, str]:
    """
    Return artifact paths as strings for API responses or summaries.
    """
    return {name: str(path) for name, path in build_artifact_paths(config).items()}


def summarize_loaded_artifacts(artifacts: LoadedColdItemArtifacts) -> dict[str, object]:
    """
    Build a compact summary of loaded training artifacts.
    """
    interactions_df = artifacts.preprocessing_artifacts.interactions_df
    return {
        "num_rows": int(len(interactions_df)),
        "num_users": int(interactions_df[artifacts.preprocessor.user_id_col].nunique()),
        "num_items": int(interactions_df[artifacts.preprocessor.item_id_col].nunique()),
        "num_warm_items": int(len(artifacts.warm_cold_result.warm_items)),
        "num_cold_items": int(len(artifacts.warm_cold_result.cold_items)),
        "num_cold_vectors": int(artifacts.cold_vector_result.num_built_vectors),
        "num_user_feature_rows": int(len(artifacts.preprocessing_artifacts.user_features_df)),
        "num_item_feature_rows": int(len(artifacts.preprocessing_artifacts.item_features_df)),
    }


def dataframe_to_record_dicts(df: pd.DataFrame) -> list[dict[str, object]]:
    """
    Convert a dataframe to JSON-friendly records.
    """
    if df.empty:
        return []
    return df.to_dict(orient="records")
