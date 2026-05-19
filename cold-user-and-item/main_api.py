from __future__ import annotations

import argparse

from config import DEFAULT_CONFIG
from api.app import create_app


def build_parser() -> argparse.ArgumentParser:
    """
    Build CLI arguments for starting the FastAPI inference service.
    """
    parser = argparse.ArgumentParser(description="Run the cold-user FastAPI service.")
    parser.add_argument(
        "--model-path",
        default=str(DEFAULT_CONFIG.paths.default_model_output_path),
        help="Path to the saved HybridRecommender model.",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_CONFIG.api.host,
        help="Host interface for the API server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_CONFIG.api.port,
        help="Port for the API server.",
    )
    parser.add_argument(
        "--reload",
        default=DEFAULT_CONFIG.api.reload,
        action="store_true",
        help="Enable uvicorn auto-reload for local development.",
    )
    return parser


def main() -> None:
    """
    CLI entrypoint for the FastAPI recommender service.
    """
    args = build_parser().parse_args()

    try:
        import uvicorn
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The `uvicorn` package is required to run the FastAPI service. Install dependencies first."
        ) from exc

    app = create_app(model_path=args.model_path)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
