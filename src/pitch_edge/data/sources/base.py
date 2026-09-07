"""Common interface every match-data connector implements.

Downstream code (storage, features, models, backtests) only depends on this
interface, never on a specific vendor. Swapping in a paid provider later
(API-Football, Sportmonks, ...) means writing one new class here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from pitch_edge.data.contracts import validate_matches

REQUIRED_MATCH_COLUMNS = (
    "match_id",
    "date",
    "league",
    "season",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "source",
)


class MatchDataSource(ABC):
    """A pluggable connector that yields completed-match rows in the canonical schema."""

    name: str = "unnamed_source"

    @abstractmethod
    def fetch_matches(self, **kwargs) -> pd.DataFrame:
        raise NotImplementedError

    @classmethod
    def required_columns(cls) -> tuple[str, ...]:
        return REQUIRED_MATCH_COLUMNS

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fail loudly on schema drift rather than silently propagating bad data."""
        missing = set(REQUIRED_MATCH_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"{self.name}: connector output is missing required columns: {sorted(missing)}")
        if df.empty:
            return df
        if df["home_goals"].isna().any() or df["away_goals"].isna().any():
            raise ValueError(f"{self.name}: null goal values found in completed-match rows")
        df = df.copy()
        df["match_id"] = df["match_id"].astype(str)
        return validate_matches(df)
