from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.als_model import ALSRecommender


@dataclass(slots=True)
class CandidateGenerationResult:
    """
    Result of retrieval-stage candidate generation.
    """

    candidates_df: pd.DataFrame
    num_warm_candidates: int
    num_cold_candidates: int
    total_candidates: int


def extract_user_vector(als_model: ALSRecommender, user_id: str) -> np.ndarray | None:
    """
    Extract a user's latent vector from a fitted ALS model.
    """
    if als_model.model is None or als_model.artifacts is None:
        raise RuntimeError("ALS model must be fitted before candidate generation.")

    user_index = als_model.artifacts.user2idx.get(str(user_id))
    if user_index is None:
        return None
    return np.asarray(als_model.model.user_factors[user_index], dtype=np.float32).copy()


def get_seen_items(als_model: ALSRecommender, user_id: str) -> set[str]:
    """
    Return the set of items already seen by the user in ALS history.
    """
    return set(als_model.seen_items_by_user.get(str(user_id), set()))


def generate_warm_candidates(
    als_model: ALSRecommender,
    user_id: str,
    top_k: int,
    candidate_item_ids: list[str] | None = None,
    exclude_seen: bool = True,
) -> pd.DataFrame:
    """
    Generate warm candidates for a user via ALS retrieval.
    """
    warm_recommendations = als_model.recommend(
        user_id=str(user_id),
        candidate_item_ids=candidate_item_ids,
        top_k=top_k,
        exclude_seen=exclude_seen,
    )

    rows = [
        {
            "user_id": str(user_id),
            "item_id": str(item_id),
            "retrieval_score": float(score),
            "retrieval_source": "als_warm",
            "is_cold_item": False,
        }
        for item_id, score in warm_recommendations
    ]
    return pd.DataFrame(rows)


def score_cold_items(
    user_vector: np.ndarray,
    cold_vector_map: dict[str, np.ndarray],
    candidate_item_ids: list[str] | None = None,
) -> list[tuple[str, float]]:
    """
    Score cold items for one user with a dot product in latent space.
    """
    candidate_ids = (
        [str(item_id) for item_id in candidate_item_ids if str(item_id) in cold_vector_map]
        if candidate_item_ids is not None
        else list(cold_vector_map.keys())
    )

    scored_items: list[tuple[str, float]] = []
    for item_id in candidate_ids:
        cold_vector = np.asarray(cold_vector_map[item_id], dtype=np.float32)
        score = float(np.dot(user_vector, cold_vector))
        scored_items.append((item_id, score))

    scored_items.sort(key=lambda pair: pair[1], reverse=True)
    return scored_items


def generate_cold_candidates(
    als_model: ALSRecommender,
    user_id: str,
    cold_vector_map: dict[str, np.ndarray],
    top_k: int,
    candidate_item_ids: list[str] | None = None,
    exclude_seen: bool = True,
) -> pd.DataFrame:
    """
    Generate cold candidates for a user from synthetic cold-item vectors.
    """
    user_vector = extract_user_vector(als_model, str(user_id))
    if user_vector is None or top_k <= 0:
        return pd.DataFrame(columns=["user_id", "item_id", "retrieval_score", "retrieval_source", "is_cold_item"])

    seen_items = get_seen_items(als_model, str(user_id)) if exclude_seen else set()
    filtered_candidate_ids = None
    if candidate_item_ids is not None:
        filtered_candidate_ids = [str(item_id) for item_id in candidate_item_ids if str(item_id) not in seen_items]

    scored_items = score_cold_items(
        user_vector=user_vector,
        cold_vector_map=cold_vector_map,
        candidate_item_ids=filtered_candidate_ids,
    )

    rows = []
    for item_id, score in scored_items[:top_k]:
        if exclude_seen and item_id in seen_items:
            continue
        rows.append(
            {
                "user_id": str(user_id),
                "item_id": str(item_id),
                "retrieval_score": float(score),
                "retrieval_source": "cold_vector",
                "is_cold_item": True,
            }
        )
    return pd.DataFrame(rows)


def merge_candidate_frames(
    warm_candidates_df: pd.DataFrame,
    cold_candidates_df: pd.DataFrame,
    final_pool_size: int,
) -> pd.DataFrame:
    """
    Merge warm and cold candidates into one deduplicated retrieval pool.
    """
    candidate_frames = [df for df in [warm_candidates_df, cold_candidates_df] if not df.empty]
    if not candidate_frames:
        return pd.DataFrame(columns=["user_id", "item_id", "retrieval_score", "retrieval_source", "is_cold_item"])

    merged_df = pd.concat(candidate_frames, ignore_index=True)
    merged_df = merged_df.sort_values("retrieval_score", ascending=False).reset_index(drop=True)

    # Keep the highest-scoring occurrence if the same item appears multiple times.
    merged_df = merged_df.drop_duplicates(subset=["user_id", "item_id"], keep="first").reset_index(drop=True)

    if final_pool_size > 0:
        merged_df = merged_df.head(final_pool_size).reset_index(drop=True)

    merged_df["retrieval_rank"] = np.arange(1, len(merged_df) + 1)
    return merged_df


def generate_candidates_for_user(
    als_model: ALSRecommender,
    user_id: str,
    cold_vector_map: dict[str, np.ndarray],
    warm_candidates_per_user: int,
    cold_candidates_per_user: int,
    final_candidate_pool_size: int,
    warm_candidate_item_ids: list[str] | None = None,
    cold_candidate_item_ids: list[str] | None = None,
    exclude_seen: bool = True,
) -> CandidateGenerationResult:
    """
    Generate a unified candidate pool for one user.
    """
    warm_candidates_df = generate_warm_candidates(
        als_model=als_model,
        user_id=str(user_id),
        top_k=warm_candidates_per_user,
        candidate_item_ids=warm_candidate_item_ids,
        exclude_seen=exclude_seen,
    )
    cold_candidates_df = generate_cold_candidates(
        als_model=als_model,
        user_id=str(user_id),
        cold_vector_map=cold_vector_map,
        top_k=cold_candidates_per_user,
        candidate_item_ids=cold_candidate_item_ids,
        exclude_seen=exclude_seen,
    )
    candidates_df = merge_candidate_frames(
        warm_candidates_df=warm_candidates_df,
        cold_candidates_df=cold_candidates_df,
        final_pool_size=final_candidate_pool_size,
    )

    return CandidateGenerationResult(
        candidates_df=candidates_df,
        num_warm_candidates=int(len(warm_candidates_df)),
        num_cold_candidates=int(len(cold_candidates_df)),
        total_candidates=int(len(candidates_df)),
    )


@dataclass(slots=True)
class CandidateGenerator:
    """
    Retrieval-stage candidate generator for warm and cold items.
    """

    warm_candidates_per_user: int = 200
    cold_candidates_per_user: int = 200
    final_candidate_pool_size: int = 400
    exclude_seen: bool = True

    def generate_for_user(
        self,
        als_model: ALSRecommender,
        user_id: str,
        cold_vector_map: dict[str, np.ndarray],
        warm_candidate_item_ids: list[str] | None = None,
        cold_candidate_item_ids: list[str] | None = None,
    ) -> CandidateGenerationResult:
        """
        Generate the final candidate pool for one user.
        """
        return generate_candidates_for_user(
            als_model=als_model,
            user_id=str(user_id),
            cold_vector_map=cold_vector_map,
            warm_candidates_per_user=self.warm_candidates_per_user,
            cold_candidates_per_user=self.cold_candidates_per_user,
            final_candidate_pool_size=self.final_candidate_pool_size,
            warm_candidate_item_ids=warm_candidate_item_ids,
            cold_candidate_item_ids=cold_candidate_item_ids,
            exclude_seen=self.exclude_seen,
        )
