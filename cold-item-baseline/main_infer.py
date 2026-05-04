from __future__ import annotations

import argparse

from src.inference_pipeline import run_inference


def build_parser() -> argparse.ArgumentParser:
    """
    Build CLI arguments for hybrid model inference.
    """
    parser = argparse.ArgumentParser(description="Run inference with the BPMSoft cold-item hybrid recommender.")
    parser.add_argument("--model-path", required=True, help="Path to the saved HybridRecommender model.")
    parser.add_argument("--input-csv", required=True, help="Path to CSV with candidate user-item pairs.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of recommendations per user.")
    parser.add_argument(
        "--scored-output",
        default="outputs/scored_pairs.csv",
        help="Optional output CSV path for scored candidate pairs.",
    )
    parser.add_argument(
        "--recommendations-output",
        default="outputs/recommendations.csv",
        help="Optional output CSV path for final top-k recommendations.",
    )
    return parser


def main() -> None:
    """
    CLI entrypoint for hybrid model inference.
    """
    args = build_parser().parse_args()

    _, recommendations_df, inference_summary = run_inference(
        model_path=args.model_path,
        candidate_pairs_csv_path=args.input_csv,
        top_k=args.top_k,
        scored_output_path=args.scored_output,
        recommendations_output_path=args.recommendations_output,
    )

    print("Inference completed.")
    for key, value in inference_summary.items():
        print(f"{key}: {value}")
    print(f"num_recommendations_rows: {len(recommendations_df)}")


if __name__ == "__main__":
    main()
