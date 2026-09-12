"""AS_OF point-in-time helpers: default cutoff, before/window filters, lineup confirmation."""

from __future__ import annotations

import pandas as pd
import pytest

from pitch_edge.backtest.as_of import (
    DEFAULT_AS_OF_LAG_DAYS,
    LINEUP_LEAD_TIME,
    default_as_of,
    filter_before,
    filter_window,
    lineup_is_confirmed,
)
from pitch_edge.data.ingest import store_matches

pytestmark = pytest.mark.unit


@pytest.fixture
def loaded(warehouse, synthetic_league_matches):
    store_matches(warehouse, synthetic_league_matches)
    return warehouse


def test_default_as_of_is_lagged_behind_the_real_closing_price(loaded):
    matches = loaded.matches_with_closing_odds(bookmakers=("PS",))
    latest_close = pd.to_datetime(matches[matches["PSCH"].notna()]["date"]).max()

    as_of = default_as_of(loaded)

    assert as_of == latest_close - pd.Timedelta(days=DEFAULT_AS_OF_LAG_DAYS)


def test_default_as_of_raises_without_real_closing_odds(warehouse):
    with pytest.raises(ValueError, match="closing price"):
        default_as_of(warehouse)


def test_filter_before_excludes_as_of_and_later():
    df = pd.DataFrame({"date": ["2024-01-01", "2024-01-02", "2024-01-03"]})
    out = filter_before(df, pd.Timestamp("2024-01-02"))
    assert out["date"].tolist() == ["2024-01-01"]


def test_filter_window_keeps_only_the_requested_span():
    df = pd.DataFrame({"date": ["2024-01-01", "2024-01-05", "2024-01-20"]})
    out = filter_window(df, pd.Timestamp("2024-01-01"), window_days=14)
    assert out["date"].tolist() == ["2024-01-01", "2024-01-05"]


def test_lineup_confirmed_flips_exactly_at_the_lead_time_boundary():
    kickoff = pd.Timestamp("2024-06-01T15:00:00")
    assert lineup_is_confirmed(kickoff - LINEUP_LEAD_TIME, kickoff) is True
    assert lineup_is_confirmed(kickoff - LINEUP_LEAD_TIME - pd.Timedelta(minutes=1), kickoff) is False
