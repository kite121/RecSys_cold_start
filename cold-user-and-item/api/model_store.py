from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from src.hybrid_recommender import HybridRecommender


@dataclass
class ModelStore:
    """
    In-memory holder for the currently active recommender model.
    """

    model: HybridRecommender | None = None
    model_path: str | None = None
    loaded_at_utc: datetime | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def load_model(self, model_path: str | Path) -> HybridRecommender:
        """
        Load a HybridRecommender artifact and keep it in memory.
        """
        normalized_path = str(Path(model_path).expanduser().resolve())
        loaded_model = HybridRecommender.load(normalized_path)

        self.model = loaded_model
        self.model_path = normalized_path
        self.loaded_at_utc = datetime.now(timezone.utc)
        self.metadata = {
            "mode": getattr(loaded_model, "mode", None),
            "loaded_at_utc": self.loaded_at_utc.isoformat(),
            "model_path": normalized_path,
        }
        return loaded_model

    def get_model(self) -> HybridRecommender:
        """
        Return the loaded model or raise if it is not available.
        """
        if self.model is None:
            raise RuntimeError("No recommender model is loaded.")
        return self.model

    def is_model_loaded(self) -> bool:
        """
        Return whether a model is currently available in memory.
        """
        return self.model is not None

    def clear(self) -> None:
        """
        Drop the currently loaded model from memory.
        """
        self.model = None
        self.model_path = None
        self.loaded_at_utc = None
        self.metadata = {}

    def get_status(self) -> dict[str, object]:
        """
        Return a lightweight status snapshot for readiness checks.
        """
        return {
            "model_loaded": self.is_model_loaded(),
            "model_path": self.model_path,
            "loaded_at_utc": self.loaded_at_utc.isoformat() if self.loaded_at_utc is not None else None,
            **self.metadata,
        }


model_store = ModelStore()
