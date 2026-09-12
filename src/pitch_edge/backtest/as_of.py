"""Point-in-time correctness for the `ask` agent.

There is no live-odds feed today (`ODDS_API_KEY` unset), so there is no genuine "now" a question
can honestly be answered as of. Every question is instead answered as of an explicit historical
instant, `as_of`, defaulting to a date strictly before the most recent real closing-odds price in
the warehouse. Every node that needs a cutoff — training data, feature values, RAG retrieval,
lineup confirmation — goes through this module so none of them can drift out of sync with another.
"""

from __future__ import annotations

from datetime import timedelta

import pandas as pd

from pitch_edge.data.storage import Warehouse

DEFAULT_AS_OF_LAG_DAYS = 30
LINEUP_LEAD_TIME = timedelta(hours=1)


def default_as_of(
    wh: Warehouse, bet_prefix: str = "PS", closing_prefix: str = "PSC", lag_days: int = DEFAULT_AS_OF_LAG_DAYS
) -> pd.Timestamp:
    """max(date with a real Pinnacle closing price) - `lag_days`. Only football-data.co.uk rows
    carry a genuine early/closing pair; picking a date this far back guarantees both training data
    before it and real matches with real early odds after it."""
    matches = wh.matches_with_closing_odds(bookmakers=(bet_prefix,))
    close_col = f"{closing_prefix}H"
    if matches.empty or close_col not in matches.columns:
        raise ValueError(f"no matches with a real {closing_prefix}* closing price found — cannot default as_of")
    real = matches[matches[close_col].notna()]
    if real.empty:
        raise ValueError(f"no matches with a real {closing_prefix}* closing price found — cannot default as_of")
    return pd.Timestamp(pd.to_datetime(real["date"]).max()) - pd.Timedelta(days=lag_days)


def filter_before(df: pd.DataFrame, as_of: pd.Timestamp, date_col: str = "date") -> pd.DataFrame:
    """Rows strictly before `as_of` — the one filter every as-of-scoped lookup reuses."""
    d = pd.to_datetime(df[date_col])
    return df[d < as_of]


def filter_window(df: pd.DataFrame, as_of: pd.Timestamp, window_days: int, date_col: str = "date") -> pd.DataFrame:
    """Rows in `[as_of, as_of + window_days)` — candidate fixtures for a question asked at `as_of`."""
    d = pd.to_datetime(df[date_col])
    return df[(d >= as_of) & (d < as_of + pd.Timedelta(days=window_days))]


def lineup_is_confirmed(as_of: pd.Timestamp, kickoff: pd.Timestamp) -> bool:
    """True once `as_of` is within `LINEUP_LEAD_TIME` of `kickoff`. A stated modeling assumption
    (real announcement timing varies match to match and isn't recorded), not a measured fact."""
    return bool(as_of >= (kickoff - LINEUP_LEAD_TIME))
