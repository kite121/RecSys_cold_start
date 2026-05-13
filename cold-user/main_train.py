from __future__ import annotations

import argparse

from config import DEFAULT_CONFIG
from src.train_pipeline import train_hybrid_model


def build_parser() -> argparse.ArgumentParser:
    """
    Build CLI arguments for hybrid model training.
    """
    parser = argparse.ArgumentParser(description="Train the cold-user hybrid recommender.")
    parser.add_argument("--train-csv", required=True, help="Path to the training CSV file.")
    parser.add_argument(
        "--model-output",
        default=str(DEFAULT_CONFIG.paths.default_model_output_path),
        help="Path to save the trained HybridRecommender artifact.",
    )

    parser.add_argument("--user-id-col", default=DEFAULT_CONFIG.data.user_id_col, help="User id column name.")
    parser.add_argument("--item-id-col", default=DEFAULT_CONFIG.data.item_id_col, help="Item id column name.")
    parser.add_argument("--value-col", default=DEFAULT_CONFIG.data.value_col, help="Interaction value column name.")
    parser.add_argument("--user-prefix", default=DEFAULT_CONFIG.data.user_prefix, help="Prefix for user features.")
    parser.add_argument("--item-prefix", default=DEFAULT_CONFIG.data.item_prefix, help="Prefix for item features.")

    parser.add_argument(
        "--min-user-interactions",
        type=int,
        default=DEFAULT_CONFIG.split.min_user_interactions,
        help="Minimum number of interactions for a user to be considered warm.",
    )
    parser.add_argument(
        "--min-item-interactions",
        type=int,
        default=DEFAULT_CONFIG.split.min_item_interactions,
        help="Minimum number of interactions for an item to be considered warm.",
    )
    parser.add_argument(
        "--popularity-metric",
        default=DEFAULT_CONFIG.split.popularity_metric,
        choices=["count", "value_sum"],
        help="Popularity metric used for warm/cold splitting and popular fallback.",
    )

    parser.add_argument("--als-factors", type=int, default=DEFAULT_CONFIG.als.factors, help="ALS factor count.")
    parser.add_argument(
        "--als-regularization",
        type=float,
        default=DEFAULT_CONFIG.als.regularization,
        help="ALS regularization.",
    )
    parser.add_argument(
        "--als-iterations",
        type=int,
        default=DEFAULT_CONFIG.als.iterations,
        help="ALS iteration count.",
    )
    parser.add_argument("--als-alpha", type=float, default=DEFAULT_CONFIG.als.alpha, help="ALS alpha.")
    parser.add_argument(
        "--als-random-state",
        type=int,
        default=DEFAULT_CONFIG.als.random_state,
        help="ALS random seed.",
    )

    parser.add_argument(
        "--regressor-iterations",
        type=int,
        default=DEFAULT_CONFIG.regressor.iterations,
        help="CatBoostRegressor iteration count.",
    )
    parser.add_argument(
        "--regressor-learning-rate",
        type=float,
        default=DEFAULT_CONFIG.regressor.learning_rate,
        help="CatBoostRegressor learning rate.",
    )
    parser.add_argument(
        "--regressor-depth",
        type=int,
        default=DEFAULT_CONFIG.regressor.depth,
        help="CatBoostRegressor tree depth.",
    )
    parser.add_argument(
        "--regressor-loss-function",
        default=DEFAULT_CONFIG.regressor.loss_function,
        help="CatBoostRegressor loss function.",
    )
    parser.add_argument(
        "--regressor-random-seed",
        type=int,
        default=DEFAULT_CONFIG.regressor.random_seed,
        help="CatBoostRegressor random seed.",
    )
    parser.add_argument(
        "--negative-samples-per-user",
        type=int,
        default=DEFAULT_CONFIG.regressor.negative_samples_per_user,
        help="Number of sampled warm negative pairs per user.",
    )

    parser.add_argument(
        "--top-n-popular",
        type=int,
        default=DEFAULT_CONFIG.candidate.top_n_popular,
        help="Size of the popular fallback pool.",
    )
    parser.add_argument(
        "--top-k-diverse",
        type=int,
        default=DEFAULT_CONFIG.candidate.top_k_diverse,
        help="Size of the diversified maxvol pool.",
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
        min_user_interactions=args.min_user_interactions,
        min_item_interactions=args.min_item_interactions,
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
        top_n_popular=args.top_n_popular,
        top_k_diverse=args.top_k_diverse,
    )

    print("Training completed.")
    for key, value in training_summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
