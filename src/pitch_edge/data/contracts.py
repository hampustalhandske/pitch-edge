"""Data-quality contracts (pandera). Fail loudly on schema drift.

Every connector's output passes through `validate_matches` before it can be
stored; every feature frame passes through `validate_features` before a model
sees it. A source silently changing a column name or emitting negative goals
raises here instead of quietly poisoning a backtest.
"""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

MATCH_SCHEMA = DataFrameSchema(
    {
        "match_id": Column(str, unique=True, nullable=False),
        "date": Column(pa.DateTime, nullable=False),
        "league": Column(str, nullable=False),
        "season": Column(str, nullable=False),
        "home_team": Column(str, nullable=False),
        "away_team": Column(str, nullable=False),
        "home_goals": Column(float, Check.ge(0), nullable=False, coerce=True),
        "away_goals": Column(float, Check.ge(0), nullable=False, coerce=True),
        "source": Column(str, nullable=False),
    },
    strict=False,  # extra, source-specific columns (odds, shots, referee...) are welcome
    coerce=True,
)

ODDS_LONG_SCHEMA = DataFrameSchema(
    {
        "match_id": Column(str, nullable=False),
        "bookmaker": Column(str, nullable=False),
        "market": Column(str, nullable=False),  # "1x2" | "totals_2.5"
        "side": Column(str, nullable=False),  # home/draw/away | over/under
        "price": Column(float, Check.gt(1.0), nullable=False, coerce=True),
        "is_closing": Column(bool, nullable=False, coerce=True),
        "snapshot_ts": Column(pa.DateTime, nullable=True, coerce=True),
    },
    strict=False,
    coerce=True,
)


def validate_matches(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return MATCH_SCHEMA.validate(df, lazy=True)


def validate_odds_long(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return ODDS_LONG_SCHEMA.validate(df, lazy=True)
