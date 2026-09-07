"""Integration test for the odds provider layer against real-shaped match rows."""

from __future__ import annotations

import pandas as pd
import pytest

from pitch_edge.odds.providers import HistoricalClosingOddsProvider, MockLiveOddsProvider
from pitch_edge.odds.utils import no_vig_probabilities

pytestmark = pytest.mark.integration


@pytest.fixture
def matches_with_odds() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "match_id": ["m1", "m2"],
            "date": [pd.Timestamp("2023-08-12"), pd.Timestamp("2023-08-13")],
            "B365H": [1.80, 2.90],
            "B365D": [3.60, 3.30],
            "B365A": [4.50, 2.50],
            "PSH": [1.85, 2.95],
            "PSD": [3.70, 3.35],
            "PSA": [4.40, 2.45],
        }
    )


def test_historical_provider_returns_multiple_bookmakers(matches_with_odds):
    provider = HistoricalClosingOddsProvider(matches_with_odds)
    quotes = provider.get_quotes("m1")
    bookmakers = {q.bookmaker for q in quotes}
    assert "Bet365" in bookmakers
    assert "Pinnacle" in bookmakers
    assert all(q.is_closing for q in quotes)


def test_historical_provider_unknown_match_returns_empty(matches_with_odds):
    provider = HistoricalClosingOddsProvider(matches_with_odds)
    assert provider.get_quotes("does-not-exist") == []


def test_no_vig_probabilities_consistent_across_bookmakers(matches_with_odds):
    provider = HistoricalClosingOddsProvider(matches_with_odds)
    quotes = provider.get_quotes("m1")
    for q in quotes:
        probs = no_vig_probabilities(q.home_odds, q.draw_odds, q.away_odds)
        assert sum(probs) == pytest.approx(1.0, abs=1e-9)


def test_mock_live_provider_is_labeled_synthetic(matches_with_odds):
    closing = HistoricalClosingOddsProvider(matches_with_odds)
    live = MockLiveOddsProvider(closing, seed=1)
    quotes = live.get_quotes("m1")
    assert len(quotes) > 0
    for q in quotes:
        assert "synthetic" in q.bookmaker.lower()
        assert q.is_closing is False


def test_mock_live_provider_deterministic_given_seed(matches_with_odds):
    closing = HistoricalClosingOddsProvider(matches_with_odds)
    live_a = MockLiveOddsProvider(closing, seed=99)
    live_b = MockLiveOddsProvider(closing, seed=99)
    assert live_a.get_quotes("m1") == live_b.get_quotes("m1")
