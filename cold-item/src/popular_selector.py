from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd


DEFAULT_EVENT_WEIGHTS = {
    "view": 1.0,
    "click": 2.0,
    "cart": 3.0,
    "purchase": 5.0,
}


@dataclass(slots=True)
class PopularSelectionResult:
    """
    Result of top-popular selection.
    """

    top_items_df: pd.DataFrame
    popularity_df: pd.DataFrame
    top_item_ids: list[str]
    strategy: str
    input_size: int
    output_size: int


def ensure_columns_present(df: pd.DataFrame, required_columns: list[str]) -> None:
    """
    Validate that all required columns exist in the dataframe.
    """
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")


def normalize_interactions_input(interactions: pd.DataFrame | list[dict]) -> pd.DataFrame:
    """
    Normalize interactions input to a pandas dataframe.
    """
    if isinstance(interactions, pd.DataFrame):
        return interactions.copy()
    if isinstance(interactions, list):
        return pd.DataFrame(interactions)
    raise TypeError("interactions must be a pandas.DataFrame or a list of dictionaries.")


def compute_time_decay_weights(
    interactions_df: pd.DataFrame,
    timestamp_col: str,
    decay_rate: float,
    now: datetime | None = None,
) -> pd.Series:
    """
    Compute exponential time decay weights for interaction timestamps.
    """
    reference_time = now or datetime.now()
    timestamps = pd.to_datetime(interactions_df[timestamp_col], errors="coerce")
    age_in_days = (reference_time - timestamps).dt.total_seconds().div(86400.0)
    age_in_days = age_in_days.fillna(0.0).clip(lower=0.0)
    return age_in_days.map(lambda value: math.exp(-decay_rate * float(value)))


def compute_popularity_scores(
    interactions: pd.DataFrame | list[dict],
    item_col: str = "item_id",
    user_col: str = "user_id",
    value_col: str = "value",
    event_type_col: str | None = None,
    timestamp_col: str | None = None,
    decay_rate: float = 0.05,
    user_cap: float = 5.0,
    unique_user_bonus: float = 0.2,
    event_weights: dict[str, float] | None = None,
    now: datetime | None = None,
) -> pd.DataFrame:
    """
    Compute item popularity with optional event weighting and time decay.

    Two modes are supported:
    - weighted event mode: uses ``event_type_col`` and optional ``timestamp_col``
    - value mode: uses ``value_col`` as the base interaction weight
    """
    interactions_df = normalize_interactions_input(interactions)
    required_columns = [item_col, user_col]
    if event_type_col is None:
        required_columns.append(value_col)
    else:
        required_columns.append(event_type_col)
    if timestamp_col is not None:
        required_columns.append(timestamp_col)
    ensure_columns_present(interactions_df, required_columns)

    if interactions_df.empty:
        return pd.DataFrame(columns=[item_col, "popularity", "unique_users", "selection_rank"])

    prepared_df = interactions_df.copy()
    prepared_df[item_col] = prepared_df[item_col].astype(str)
    prepared_df[user_col] = prepared_df[user_col].astype(str)

    if event_type_col is not None:
        effective_event_weights = event_weights or DEFAULT_EVENT_WEIGHTS
        base_weight = prepared_df[event_type_col].map(effective_event_weights).fillna(1.0)
        strategy = "event_weighted_popularity"
    else:
        base_weight = pd.to_numeric(prepared_df[value_col], errors="coerce").fillna(0.0)
        strategy = "value_weighted_popularity"

    if timestamp_col is not None:
        time_decay = compute_time_decay_weights(
            interactions_df=prepared_df,
            timestamp_col=timestamp_col,
            decay_rate=decay_rate,
            now=now,
        )
    else:
        time_decay = pd.Series(1.0, index=prepared_df.index, dtype=float)

    prepared_df["base_score"] = base_weight.astype(float) * time_decay.astype(float)

    user_item_scores: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    unique_users: dict[str, set[str]] = defaultdict(set)

    for row in prepared_df[[item_col, user_col, "base_score"]].to_dict(orient="records"):
        current_item_id = str(row[item_col])
        current_user_id = str(row[user_col])
        current_score = float(row["base_score"])
        user_item_scores[current_item_id][current_user_id] += current_score
        unique_users[current_item_id].add(current_user_id)

    rows: list[dict[str, float | int | str]] = []
    for current_item_id, users_scores in user_item_scores.items():
        capped_score_sum = 0.0
        for current_user_score in users_scores.values():
            capped_score_sum += min(float(current_user_score), user_cap)

        unique_users_count = len(unique_users[current_item_id])
        popularity = capped_score_sum * (1.0 + unique_user_bonus * math.log1p(unique_users_count))
        rows.append(
            {
                item_col: current_item_id,
                "popularity": popularity,
                "unique_users": unique_users_count,
            }
        )

    popularity_df = pd.DataFrame(rows)
    if popularity_df.empty:
        return pd.DataFrame(columns=[item_col, "popularity", "unique_users", "selection_rank"])

    popularity_df = popularity_df.sort_values("popularity", ascending=False).reset_index(drop=True)
    popularity_df["selection_rank"] = popularity_df.index + 1
    popularity_df.attrs["strategy"] = strategy
    return popularity_df


def select_top_popular_items(
    interactions: pd.DataFrame | list[dict],
    top_n: int,
    item_col: str = "item_id",
    user_col: str = "user_id",
    value_col: str = "value",
    event_type_col: str | None = None,
    timestamp_col: str | None = None,
    decay_rate: float = 0.05,
    user_cap: float = 5.0,
    unique_user_bonus: float = 0.2,
    event_weights: dict[str, float] | None = None,
    now: datetime | None = None,
) -> PopularSelectionResult:
    """
    Select top-N popular items with optional event weighting and time decay.
    """
    popularity_df = compute_popularity_scores(
        interactions=interactions,
        item_col=item_col,
        user_col=user_col,
        value_col=value_col,
        event_type_col=event_type_col,
        timestamp_col=timestamp_col,
        decay_rate=decay_rate,
        user_cap=user_cap,
        unique_user_bonus=unique_user_bonus,
        event_weights=event_weights,
        now=now,
    )

    selected_items_df = popularity_df.head(max(top_n, 0)).copy().reset_index(drop=True)
    strategy = str(popularity_df.attrs.get("strategy", "popularity_selection"))

    return PopularSelectionResult(
        top_items_df=selected_items_df,
        popularity_df=popularity_df,
        top_item_ids=selected_items_df[item_col].astype(str).tolist() if item_col in selected_items_df else [],
        strategy=strategy,
        input_size=int(len(popularity_df)),
        output_size=int(len(selected_items_df)),
    )


@dataclass(slots=True)
class PopularItemsSelector:
    """
    Reusable selector for building a top-popular support pool.
    """

    item_col: str = "item_id"
    user_col: str = "user_id"
    value_col: str = "value"
    event_type_col: str | None = None
    timestamp_col: str | None = None
    decay_rate: float = 0.05
    user_cap: float = 5.0
    unique_user_bonus: float = 0.2
    event_weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_EVENT_WEIGHTS))

    def select(self, interactions: pd.DataFrame | list[dict], top_n: int) -> PopularSelectionResult:
        """
        Select top-N popular items from the provided interactions.
        """
        return select_top_popular_items(
            interactions=interactions,
            top_n=top_n,
            item_col=self.item_col,
            user_col=self.user_col,
            value_col=self.value_col,
            event_type_col=self.event_type_col,
            timestamp_col=self.timestamp_col,
            decay_rate=self.decay_rate,
            user_cap=self.user_cap,
            unique_user_bonus=self.unique_user_bonus,
            event_weights=self.event_weights,
        )
