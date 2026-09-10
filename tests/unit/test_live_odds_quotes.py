"""live_quotes_from_odds_api: matches real odds by team name even when the Odds API's own spelling
differs from our canonical form — the `side` outcome must resolve against the event's own raw team
names, not get normalized independently and silently fail to match."""

from __future__ import annotations

import pandas as pd
import pytest

from pitch_edge.pipeline import live_quotes_from_odds_api

pytestmark = pytest.mark.unit


def _fixtures():
    return pd.DataFrame(
        {
            "match_id": ["m1"],
            "home_team": ["Tottenham"],  # our canonical spelling
            "away_team": ["Man United"],
        }
    )


def test_live_quotes_match_when_odds_api_spells_teams_differently(warehouse):
    warehouse.upsert(
        "live_odds",
        pd.DataFrame(
            {
                "match_id": ["evt1", "evt1", "evt1"],
                "bookmaker": ["pinnacle"] * 3,
                "market": ["h2h"] * 3,
                "side": ["Tottenham Hotspur", "Draw", "Manchester United"],  # differs from canonical
                "price": [2.1, 3.4, 3.6],
                "snapshot_ts": [pd.Timestamp("2025-01-01T10:00:00")] * 3,
                "home_team": ["Tottenham Hotspur", "Tottenham Hotspur", "Tottenham Hotspur"],
                "away_team": ["Manchester United", "Manchester United", "Manchester United"],
            }
        ),
    )
    quotes = live_quotes_from_odds_api(warehouse, _fixtures())
    assert "m1" in quotes
    assert quotes["m1"]["home"] == pytest.approx(2.1)
    assert quotes["m1"]["draw"] == pytest.approx(3.4)
    assert quotes["m1"]["away"] == pytest.approx(3.6)


def test_live_quotes_empty_without_live_odds_table(warehouse):
    assert live_quotes_from_odds_api(warehouse, _fixtures()) == {}


def test_live_quotes_uses_latest_snapshot_and_ignores_non_h2h_markets(warehouse):
    warehouse.upsert(
        "live_odds",
        pd.DataFrame(
            {
                "match_id": ["evt1"] * 4,
                "bookmaker": ["pinnacle"] * 4,
                "market": ["h2h", "h2h", "h2h", "totals"],
                "side": ["Tottenham", "Draw", "Man United", "Over"],
                "price": [1.9, 3.5, 3.8, 1.9],
                "snapshot_ts": [
                    pd.Timestamp("2025-01-01T09:00:00"),
                    pd.Timestamp("2025-01-01T09:00:00"),
                    pd.Timestamp("2025-01-01T09:00:00"),
                    pd.Timestamp("2025-01-01T09:00:00"),
                ],
                "home_team": ["Tottenham"] * 4,
                "away_team": ["Man United"] * 4,
            }
        ),
    )
    quotes = live_quotes_from_odds_api(warehouse, _fixtures())
    assert quotes["m1"]["home"] == pytest.approx(1.9)
    assert quotes["m1"]["draw"] == pytest.approx(3.5)
