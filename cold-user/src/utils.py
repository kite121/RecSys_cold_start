from __future__ import annotations

from pathlib import Path


def ensure_parent_dir(path: str | Path) -> Path:
    """
    Ensure that the parent directory for an output file exists.
    """
    normalized_path = Path(path)
    normalized_path.parent.mkdir(parents=True, exist_ok=True)
    return normalized_path
