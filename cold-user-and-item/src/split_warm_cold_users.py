from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class WarmColdUserSplitResult:
    """
    Result of warm/cold user splitting.
    """

    user_popularity_df: pd.DataFrame
    warm_users_df: pd.DataFrame
    cold_users_df: pd.DataFrame
    warm_users: set[str]
    cold_users: set[str]


def ensure_columns_present(df: pd.DataFrame, required_columns: list[str] | tuple[str, ...]) -> None:
    """
    Validate that all required columns are present in the dataframe.
    """
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")


def count_user_interactions(
    interactions_df: pd.DataFrame,
    user_col: str = "user_id",
    value_col: str = "value",
    popularity_metric: str = "count",
) -> pd.DataFrame:
    """
    Count user popularity for warm/cold user splitting.

    Supported metrics:
    - ``count``: number of interaction rows for each user
    - ``value_sum``: sum of ``value`` for each user
    """
    ensure_columns_present(interactions_df, [user_col])

    if popularity_metric == "count":
        user_popularity_df = (
            interactions_df.groupby(user_col)
            .size()
            .rename("popularity")
            .reset_index()
        )
    elif popularity_metric == "value_sum":
        ensure_columns_present(interactions_df, [value_col])
        user_popularity_df = (
            interactions_df.groupby(user_col)[value_col]
            .sum()
            .rename("popularity")
            .reset_index()
        )
    else:
        raise ValueError("popularity_metric must be either 'count' or 'value_sum'.")

    return user_popularity_df.sort_values("popularity", ascending=False).reset_index(drop=True)


def split_warm_cold_users(
    interactions_df: pd.DataFrame,
    user_col: str = "user_id",
    value_col: str = "value",
    min_user_interactions: int | float = 5,
    popularity_metric: str = "count",
) -> WarmColdUserSplitResult:
    """
    Split users into warm and cold groups.

    Warm user:
    - popularity >= threshold

    Cold user:
    - popularity < threshold
    """
    user_popularity_df = count_user_interactions(
        interactions_df=interactions_df,
        user_col=user_col,
        value_col=value_col,
        popularity_metric=popularity_metric,
    )
    user_popularity_df["user_group"] = user_popularity_df["popularity"].apply(
        lambda popularity: "warm" if popularity >= min_user_interactions else "cold"
    )

    warm_users_df = user_popularity_df[user_popularity_df["user_group"] == "warm"].reset_index(drop=True)
    cold_users_df = user_popularity_df[user_popularity_df["user_group"] == "cold"].reset_index(drop=True)

    return WarmColdUserSplitResult(
        user_popularity_df=user_popularity_df,
        warm_users_df=warm_users_df,
        cold_users_df=cold_users_df,
        warm_users=set(warm_users_df[user_col].astype(str)),
        cold_users=set(cold_users_df[user_col].astype(str)),
    )


@dataclass
class WarmColdUserSplitter:
    """
    Reusable wrapper around warm/cold user split logic.
    """

    min_user_interactions: int = 5
    popularity_metric: str = "count"
    user_col: str = "user_id"
    value_col: str = "value"

    def compute_user_popularity(self, interactions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute user popularity according to the configured metric.
        """
        return count_user_interactions(
            interactions_df=interactions_df,
            user_col=self.user_col,
            value_col=self.value_col,
            popularity_metric=self.popularity_metric,
        )

    def split(self, interactions_df: pd.DataFrame) -> WarmColdUserSplitResult:
        """
        Produce the final warm/cold user split.
        """
        return split_warm_cold_users(
            interactions_df=interactions_df,
            user_col=self.user_col,
            value_col=self.value_col,
            min_user_interactions=self.min_user_interactions,
            popularity_metric=self.popularity_metric,
        )
