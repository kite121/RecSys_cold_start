from __future__ import annotations

import argparse
from pathlib import Path

from config import (
    ALSConfig,
    ColdItemProjectConfig,
    DataConfig,
    InferenceConfig,
    PathConfig,
    RankerConfig,
    RetrievalConfig,
    WarmColdConfig,
)
from src.train_pipeline import train_cold_item_pipeline


def build_parser() -> argparse.ArgumentParser:
    """
    Build CLI arguments for the new cold-item training pipeline.
    """
    parser = argparse.ArgumentParser(description="Train the cold-item retrieval + ranking pipeline.")
    parser.add_argument("--train-csv", required=True, help="Path to the training CSV file.")
    parser.add_argument("--artifacts-dir", default="artifacts", help="Directory to save trained artifacts.")

    parser.add_argument("--user-id-col", default="user_id", help="User id column name.")
    parser.add_argument("--item-id-col", default="item_id", help="Item id column name.")
    parser.add_argument("--value-col", default="value", help="Interaction value column name.")
    parser.add_argument("--user-prefix", default="user_", help="Prefix for optional user feature columns.")
    parser.add_argument("--item-prefix", default="item_", help="Prefix for optional item feature columns.")
    parser.add_argument("--csv-sep", default=",", help="CSV separator.")
    parser.add_argument("--csv-encoding", default="utf-8", help="CSV encoding.")

    parser.add_argument("--min-warm-interactions", type=int, default=5, help="Warm item threshold.")
    parser.add_argument(
        "--popularity-metric",
        default="count",
        choices=["count", "value_sum"],
        help="Metric used to split warm and cold items.",
    )

    parser.add_argument("--als-factors", type=int, default=64, help="ALS latent factor count.")
    parser.add_argument("--als-regularization", type=float, default=0.01, help="ALS regularization.")
    parser.add_argument("--als-iterations", type=int, default=100, help="ALS iteration count.")
    parser.add_argument("--als-alpha", type=float, default=20.0, help="ALS alpha coefficient.")
    parser.add_argument("--als-random-state", type=int, default=42, help="ALS random seed.")

    parser.add_argument("--top-n-popular", type=int, default=5000, help="Top-N popular warm items for support pool.")
    parser.add_argument("--top-k-diverse", type=int, default=500, help="Top-K diverse support items after maxvol.")
    parser.add_argument("--top-m-neighbors", type=int, default=20, help="Top-M support neighbors per cold item.")
    parser.add_argument(
        "--similarity-metric",
        default="cosine",
        choices=["cosine", "dot", "euclidean"],
        help="Similarity metric used for cold/support neighbor search.",
    )
    parser.add_argument("--warm-candidates-per-user", type=int, default=200, help="Warm ALS candidates per user.")
    parser.add_argument("--cold-candidates-per-user", type=int, default=200, help="Cold-vector candidates per user.")
    parser.add_argument("--final-candidate-pool-size", type=int, default=400, help="Final retrieval pool size.")

    parser.add_argument("--ranker-iterations", type=int, default=300, help="CatBoostRanker iteration count.")
    parser.add_argument("--ranker-learning-rate", type=float, default=0.05, help="CatBoostRanker learning rate.")
    parser.add_argument("--ranker-depth", type=int, default=6, help="CatBoostRanker tree depth.")
    parser.add_argument(
        "--ranker-loss-function",
        default="YetiRankPairwise",
        help="CatBoostRanker loss function.",
    )
    parser.add_argument("--ranker-random-seed", type=int, default=42, help="CatBoostRanker random seed.")
    parser.add_argument(
        "--negative-samples-per-user",
        type=int,
        default=3,
        help="Number of retrieval negatives kept per positive training pair.",
    )
    parser.add_argument(
        "--use-interaction-value-as-label",
        action="store_true",
        help="Use aggregated interaction value as the ranker label instead of binary relevance.",
    )

    return parser


def build_config(args: argparse.Namespace) -> ColdItemProjectConfig:
    """
    Build the project config from parsed CLI arguments.
    """
    project_dir = Path(__file__).resolve().parent
    artifacts_dir = Path(args.artifacts_dir)
    if not artifacts_dir.is_absolute():
        artifacts_dir = project_dir / artifacts_dir

    return ColdItemProjectConfig(
        paths=PathConfig(
            project_dir=project_dir,
            artifacts_dir=artifacts_dir,
            baseline_dir=project_dir.parent / "cold-item-baseline",
        ),
        data=DataConfig(
            user_id_col=args.user_id_col,
            item_id_col=args.item_id_col,
            value_col=args.value_col,
            user_prefix=args.user_prefix,
            item_prefix=args.item_prefix,
            csv_sep=args.csv_sep,
            csv_encoding=args.csv_encoding,
        ),
        warm_cold=WarmColdConfig(
            min_warm_interactions=args.min_warm_interactions,
            popularity_metric=args.popularity_metric,
        ),
        als=ALSConfig(
            factors=args.als_factors,
            regularization=args.als_regularization,
            iterations=args.als_iterations,
            alpha=args.als_alpha,
            random_state=args.als_random_state,
        ),
        retrieval=RetrievalConfig(
            top_n_popular=args.top_n_popular,
            top_k_diverse=args.top_k_diverse,
            top_m_neighbors=args.top_m_neighbors,
            warm_candidates_per_user=args.warm_candidates_per_user,
            cold_candidates_per_user=args.cold_candidates_per_user,
            final_candidate_pool_size=args.final_candidate_pool_size,
            similarity_metric=args.similarity_metric,
        ),
        ranker=RankerConfig(
            iterations=args.ranker_iterations,
            learning_rate=args.ranker_learning_rate,
            depth=args.ranker_depth,
            loss_function=args.ranker_loss_function,
            random_seed=args.ranker_random_seed,
            negative_samples_per_user=args.negative_samples_per_user,
        ),
        inference=InferenceConfig(),
    )


def main() -> None:
    """
    CLI entrypoint for training the cold-item pipeline.
    """
    args = build_parser().parse_args()
    config = build_config(args)

    training_result = train_cold_item_pipeline(
        train_csv_path=args.train_csv,
        config=config,
        save_artifacts=True,
        use_interaction_value_as_label=args.use_interaction_value_as_label,
    )
    training_summary = training_result.training_summary

    print("Training completed.")
    for key, value in training_summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
