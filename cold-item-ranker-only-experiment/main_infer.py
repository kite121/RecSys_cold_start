from __future__ import annotations

import argparse
from pathlib import Path

from config import ColdItemProjectConfig, PathConfig
from src.inference_pipeline import run_cold_item_inference
from src.utils import load_project_config


def build_parser() -> argparse.ArgumentParser:
    """
    Build CLI arguments for cold-item inference.
    """
    parser = argparse.ArgumentParser(description="Run cold-item retrieval + ranking inference.")
    parser.add_argument("--user-id", required=True, help="Target user id.")
    parser.add_argument("--artifacts-dir", default="artifacts", help="Directory with trained artifacts.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of recommendations to return.")
    parser.add_argument(
        "--user-context",
        action="append",
        default=[],
        help="Optional user feature override in key=value format. Repeat the flag for multiple values.",
    )
    parser.add_argument(
        "--warm-candidate-item-ids",
        default=None,
        help="Optional comma-separated whitelist of warm candidate item ids.",
    )
    parser.add_argument(
        "--cold-candidate-item-ids",
        default=None,
        help="Optional comma-separated whitelist of cold candidate item ids.",
    )
    return parser


def parse_key_value_pairs(raw_pairs: list[str]) -> dict[str, object]:
    """
    Parse repeated key=value CLI flags into a dictionary.
    """
    parsed: dict[str, object] = {}
    for raw_pair in raw_pairs:
        if "=" not in raw_pair:
            raise ValueError(f"Expected key=value format, got: {raw_pair}")
        key, value = raw_pair.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def parse_item_id_list(raw_value: str | None) -> list[str] | None:
    """
    Parse a comma-separated item-id list.
    """
    if raw_value is None:
        return None
    item_ids = [item_id.strip() for item_id in raw_value.split(",") if item_id.strip()]
    return item_ids or None


def build_config(args: argparse.Namespace) -> ColdItemProjectConfig:
    """
    Load the persisted training config and rebind its artifact directory.
    """
    project_dir = Path(__file__).resolve().parent
    artifacts_dir = Path(args.artifacts_dir)
    if not artifacts_dir.is_absolute():
        artifacts_dir = project_dir / artifacts_dir

    config_path = artifacts_dir / PathConfig().project_config_name
    config = load_project_config(config_path)
    config.paths = PathConfig(
        project_dir=project_dir,
        artifacts_dir=artifacts_dir,
        baseline_dir=project_dir.parent / "cold-item-baseline",
    )
    return config


def main() -> None:
    """
    CLI entrypoint for cold-item inference.
    """
    args = build_parser().parse_args()
    config = build_config(args)
    user_context = parse_key_value_pairs(args.user_context)

    inference_result = run_cold_item_inference(
        user_id=str(args.user_id),
        config=config,
        top_k=args.top_k,
        user_context=user_context or None,
        warm_candidate_item_ids=parse_item_id_list(args.warm_candidate_item_ids),
        cold_candidate_item_ids=parse_item_id_list(args.cold_candidate_item_ids),
    )

    print("Inference completed.")
    for key, value in inference_result.inference_summary.items():
        print(f"{key}: {value}")

    if inference_result.recommendations_df.empty:
        print("recommendations: []")
        return

    print("recommendations:")
    print(inference_result.recommendations_df.to_string(index=False))


if __name__ == "__main__":
    main()
