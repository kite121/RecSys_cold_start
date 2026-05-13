from __future__ import annotations

import argparse
from pathlib import Path

from src.train_pipeline import train_hybrid_model


def build_parser() -> argparse.ArgumentParser:
    """
    Build CLI arguments for hybrid model training.
    """
    parser = argparse.ArgumentParser(description="Train the BPMSoft cold-item hybrid recommender.")
    parser.add_argument("--train-csv", required=True, help="Path to the training CSV file.")
    parser.add_argument("--model-output", default="artifacts/hybrid_model.joblib", help="Path to save the model.")

    parser.add_argument("--user-id-col", default="user_id", help="User id column name.")
    parser.add_argument("--item-id-col", default="item_id", help="Item id column name.")
    parser.add_argument("--value-col", default="value", help="Interaction value column name.")
    parser.add_argument("--user-prefix", default="user_", help="Prefix for optional user feature columns.")
    parser.add_argument("--item-prefix", default="item_", help="Prefix for optional item feature columns.")

    parser.add_argument("--min-warm-interactions", type=int, default=5, help="Warm item threshold.")
    parser.add_argument(
        "--popularity-metric",
        default="count",
        choices=["count", "value_sum"],
        help="Metric used to split warm and cold items.",
    )

    parser.add_argument("--als-factors", type=int, default=20, help="ALS latent factor count.")
    parser.add_argument("--als-regularization", type=float, default=0.01, help="ALS regularization.")
    parser.add_argument("--als-iterations", type=int, default=150, help="ALS iteration count.")
    parser.add_argument("--als-alpha", type=float, default=20.0, help="ALS alpha coefficient.")
    parser.add_argument("--als-random-state", type=int, default=42, help="ALS random seed.")

    parser.add_argument("--regressor-iterations", type=int, default=300, help="CatBoost iteration count.")
    parser.add_argument("--regressor-learning-rate", type=float, default=0.05, help="CatBoost learning rate.")
    parser.add_argument("--regressor-depth", type=int, default=5, help="CatBoost tree depth.")
    parser.add_argument("--regressor-loss-function", default="RMSE", help="CatBoost loss function.")
    parser.add_argument("--regressor-random-seed", type=int, default=42, help="CatBoost random seed.")

    parser.add_argument(
        "--negative-samples-per-user",
        type=int,
        default=3,
        help="Number of sampled warm negative examples per observed warm interaction.",
    )
    return parser


def main() -> None:
    """
    CLI entrypoint for training the hybrid recommender.
    """
    args = build_parser().parse_args()

    _, training_summary = train_hybrid_model(
        train_csv_path=args.train_csv,
        model_output_path=args.model_output,
        user_id_col=args.user_id_col,
        item_id_col=args.item_id_col,
        value_col=args.value_col,
        user_prefix=args.user_prefix,
        item_prefix=args.item_prefix,
        min_warm_interactions=args.min_warm_interactions,
        popularity_metric=args.popularity_metric,
        als_factors=args.als_factors,
        als_regularization=args.als_regularization,
        als_iterations=args.als_iterations,
        als_alpha=args.als_alpha,
        als_random_state=args.als_random_state,
        regressor_iterations=args.regressor_iterations,
        regressor_learning_rate=args.regressor_learning_rate,
        regressor_depth=args.regressor_depth,
        regressor_loss_function=args.regressor_loss_function,
        regressor_random_seed=args.regressor_random_seed,
        negative_samples_per_user=args.negative_samples_per_user,
    )

    print("Training completed.")
    for key, value in training_summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
