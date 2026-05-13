from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class WarmColdSplitResult:
    item_popularity_df: pd.DataFrame
    warm_items_df: pd.DataFrame
    cold_items_df: pd.DataFrame
    warm_items: set[str]
    cold_items: set[str]


def ensure_columns_present(df: pd.DataFrame, required_columns: list[str]) -> None:
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")


def count_item_interactions(
    interactions_df: pd.DataFrame,
    item_col: str = "item_id",
    value_col: str = "value",
    popularity_metric: str = "count",
) -> pd.DataFrame:
    """
    Count item popularity for warm/cold splitting.

    Supported metrics:
    - ``count``: number of rows for each item
    - ``value_sum``: sum of ``value`` for each item
    """
    ensure_columns_present(interactions_df, [item_col])

    if popularity_metric == "count":
        item_popularity_df = (
            interactions_df.groupby(item_col)
            .size()
            .rename("popularity")
            .reset_index()
        )
    elif popularity_metric == "value_sum":
        ensure_columns_present(interactions_df, [value_col])
        item_popularity_df = (
            interactions_df.groupby(item_col)[value_col]
            .sum()
            .rename("popularity")
            .reset_index()
        )
    else:
        raise ValueError("popularity_metric must be either 'count' or 'value_sum'.")

    return item_popularity_df.sort_values("popularity", ascending=False).reset_index(drop=True)


def split_warm_cold_items(
    interactions_df: pd.DataFrame,
    item_col: str = "item_id",
    value_col: str = "value",
    threshold: int | float = 5,
    popularity_metric: str = "count",
) -> WarmColdSplitResult:
    """
    Split items into warm and cold groups.

    Warm item:
    - popularity >= threshold

    Cold item:
    - popularity < threshold
    """
    item_popularity_df = count_item_interactions(
        interactions_df=interactions_df,
        item_col=item_col,
        value_col=value_col,
        popularity_metric=popularity_metric,
    )
    item_popularity_df["item_group"] = item_popularity_df["popularity"].apply(
        lambda popularity: "warm" if popularity >= threshold else "cold"
    )

    warm_items_df = item_popularity_df[item_popularity_df["item_group"] == "warm"].reset_index(drop=True)
    cold_items_df = item_popularity_df[item_popularity_df["item_group"] == "cold"].reset_index(drop=True)

    return WarmColdSplitResult(
        item_popularity_df=item_popularity_df,
        warm_items_df=warm_items_df,
        cold_items_df=cold_items_df,
        warm_items=set(warm_items_df[item_col].astype(str)),
        cold_items=set(cold_items_df[item_col].astype(str)),
    )


@dataclass
class WarmColdItemSplitter:
    """
    Small reusable wrapper around warm/cold split logic.

    This class is convenient inside the training pipeline, while the plain
    functions are convenient for direct usage and for documenting the project.
    """

    min_warm_interactions: int = 5
    popularity_metric: str = "count"
    item_col: str = "item_id"
    value_col: str = "value"

    def compute_item_popularity(self, interactions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute item popularity according to the configured metric.
        """
        return count_item_interactions(
            interactions_df=interactions_df,
            item_col=self.item_col,
            value_col=self.value_col,
            popularity_metric=self.popularity_metric,
        )

    def split(self, interactions_df: pd.DataFrame) -> WarmColdSplitResult:
        """
        Produce the final warm/cold item split.
        """
        return split_warm_cold_items(
            interactions_df=interactions_df,
            item_col=self.item_col,
            value_col=self.value_col,
            threshold=self.min_warm_interactions,
            popularity_metric=self.popularity_metric,
        )
