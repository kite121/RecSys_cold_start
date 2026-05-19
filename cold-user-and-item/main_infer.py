from __future__ import annotations

import argparse
import json

from config import DEFAULT_CONFIG
from src.inference_pipeline import run_inference


def parse_user_features_json(raw_value: str | None) -> dict[str, object] | None:
    """
    Parse optional user features from a JSON string.
    """
    if raw_value is None:
        return None
    parsed_value = json.loads(raw_value)
    if not isinstance(parsed_value, dict):
        raise ValueError("user_features_json must decode to a JSON object.")
    return parsed_value


def build_parser() -> argparse.ArgumentParser:
    """
    Build CLI arguments for hybrid model inference.
    """
    parser = argparse.ArgumentParser(description="Run inference with the cold-user hybrid recommender.")
    parser.add_argument("--model-path", required=True, help="Path to the saved HybridRecommender model.")
    parser.add_argument("--user-id", required=True, help="Target user identifier.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_CONFIG.inference.default_top_k,
        help="Number of recommendations to return.",
    )
    parser.add_argument(
        "--user-features-json",
        default=None,
        help="Optional JSON object with user features for cold-user inference.",
    )
    parser.add_argument(
        "--recommendations-output",
        default=str(DEFAULT_CONFIG.paths.default_recommendations_output_path),
        help="Optional output CSV path for final recommendations.",
    )
    return parser


def main() -> None:
    """
    CLI entrypoint for hybrid model inference by user_id.
    """
    args = build_parser().parse_args()
    user_features = parse_user_features_json(args.user_features_json)

    recommendations_df, inference_summary = run_inference(
        model_path=args.model_path,
        user_id=args.user_id,
        user_features=user_features,
        top_k=args.top_k,
        recommendations_output_path=args.recommendations_output,
    )

    print("Inference completed.")
    for key, value in inference_summary.items():
        print(f"{key}: {value}")
    print(f"num_recommendations_rows: {len(recommendations_df)}")


if __name__ == "__main__":
    main()
