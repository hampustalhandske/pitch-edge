"""Confirmed-lineup live features: no-op without a key or an unpublished lineup, real signal once
API-Football has a starting XI, matched against Transfermarkt valuations/usual-XI."""

from __future__ import annotations

import pandas as pd
import pytest

from pitch_edge.features.live_lineups import LIVE_LINEUP_FEATURES, confirmed_lineup_features, resolve_fixture_ids

pytestmark = pytest.mark.unit


class _StubAPIFootball:
    name = "api_football"
    enabled = True

    def __init__(self, fixtures_df: pd.DataFrame, lineups_by_fixture: dict[int, pd.DataFrame]):
        self._fixtures_df = fixtures_df
        self._lineups = lineups_by_fixture

    def fixtures(self, league_code, season, next_n=30):
        return self._fixtures_df

    def lineups(self, fixture_id):
        return self._lineups.get(fixture_id, pd.DataFrame(columns=["fixture_id", "team", "player", "slot"]))


def _seed_transfermarkt(wh):
    wh.upsert(
        "tm_clubs",
        pd.DataFrame({"club_id": ["1", "2"], "name": ["Team00 FC", "Team01 FC"], "last_season": [2025, 2025]}),
        keys=["club_id"],
    )
    wh.upsert(
        "tm_players",
        pd.DataFrame({"player_id": [10, 11, 12, 20], "name": ["Player A", "Player B", "Player C", "Player X"]}),
        keys=["player_id"],
    )
    wh.upsert(
        "tm_player_valuations",
        pd.DataFrame(
            {
                "player_id": [10, 11, 12, 20],
                "date": pd.to_datetime(["2024-01-01"] * 4),
                "market_value_in_eur": [10_000_000, 5_000_000, 1_000_000, 8_000_000],
            }
        ),
        keys=["player_id", "date"],
    )
    apps = []
    for pid, club in ((10, 1), (11, 1), (20, 2)):
        for i in range(11):
            apps.append(
                {
                    "appearance_id": f"{pid}_{i}",
                    "player_id": pid,
                    "player_club_id": club,
                    "date": pd.Timestamp("2024-06-01"),
                    "minutes_played": 90,
                }
            )
    wh.upsert("tm_appearances", pd.DataFrame(apps), keys=["appearance_id"])


def test_no_op_without_key(warehouse):
    fixtures = pd.DataFrame(
        {
            "match_id": ["m1"],
            "home_team": ["Team00"],
            "away_team": ["Team01"],
            "date": [pd.Timestamp("2025-01-01")],
            "league_code": ["E0"],
        }
    )
    from pitch_edge.data.alt.api_football import APIFootballSource

    out = confirmed_lineup_features(warehouse, fixtures, api_src=APIFootballSource(api_key=None))
    assert out.empty


def test_no_op_when_lineup_not_yet_published(warehouse):
    fixtures = pd.DataFrame(
        {
            "match_id": ["m1"],
            "home_team": ["Team00"],
            "away_team": ["Team01"],
            "date": [pd.Timestamp("2025-01-01")],
            "league_code": ["E0"],
        }
    )
    live_fx = pd.DataFrame(
        {
            "fixture_id": [99],
            "date": [pd.Timestamp("2025-01-01", tz="UTC")],
            "home_team": ["Team00"],
            "away_team": ["Team01"],
        }
    )
    src = _StubAPIFootball(live_fx, {})
    out = confirmed_lineup_features(warehouse, fixtures, api_src=src)
    assert out.empty


def test_confirmed_lineup_computes_value_and_missing_pct(warehouse):
    _seed_transfermarkt(warehouse)
    fixtures = pd.DataFrame(
        {
            "match_id": ["m1"],
            "home_team": ["Team00"],
            "away_team": ["Team01"],
            "date": [pd.Timestamp("2025-01-01")],
            "league_code": ["E0"],
        }
    )
    live_fx = pd.DataFrame(
        {
            "fixture_id": [99],
            "date": [pd.Timestamp("2025-01-01", tz="UTC")],
            "home_team": ["Team00"],
            "away_team": ["Team01"],
        }
    )
    # Home XI missing "Player B" (usual starter) -> confirmed only has Player A. Away XI matches usual XI.
    lineup = pd.DataFrame(
        {
            "fixture_id": [99, 99],
            "team": ["Team00", "Team01"],
            "player": ["Player A", "Player X"],
            "slot": ["start", "start"],
        }
    )
    src = _StubAPIFootball(live_fx, {99: lineup})
    out = confirmed_lineup_features(warehouse, fixtures, api_src=src)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["lineup_source"] == "confirmed"
    assert set(LIVE_LINEUP_FEATURES) <= set(out.columns)
    # home usual XI = Player A (10m) + Player B (5m) = 15m; confirmed only has A -> missing 5/15
    assert row["sv_home_missing_pct"] == pytest.approx(5_000_000 / 15_000_000)
    # away usual XI = Player X only -> fully present -> 0 missing
    assert row["sv_away_missing_pct"] == pytest.approx(0.0)


def test_missing_pct_diff_treats_unresolved_side_as_zero_not_nan(warehouse):
    # Away club ("Team01") is deliberately absent from tm_clubs, so club_id_map can't resolve it and
    # sv_away_missing_pct stays NaN — the diff must still come out finite (fillna(0.0) semantics),
    # not NaN, which `(x or 0.0)` would silently produce since NaN is truthy in Python.
    wh = warehouse
    wh.upsert(
        "tm_clubs", pd.DataFrame({"club_id": ["1"], "name": ["Team00 FC"], "last_season": [2025]}), keys=["club_id"]
    )
    wh.upsert(
        "tm_players",
        pd.DataFrame({"player_id": [10, 11], "name": ["Player A", "Player B"]}),
        keys=["player_id"],
    )
    wh.upsert(
        "tm_player_valuations",
        pd.DataFrame(
            {
                "player_id": [10, 11],
                "date": pd.to_datetime(["2024-01-01"] * 2),
                "market_value_in_eur": [10_000_000, 5_000_000],
            }
        ),
        keys=["player_id", "date"],
    )
    apps = [
        {
            "appearance_id": f"{pid}_{i}",
            "player_id": pid,
            "player_club_id": 1,
            "date": pd.Timestamp("2024-06-01"),
            "minutes_played": 90,
        }
        for pid in (10, 11)
        for i in range(11)
    ]
    wh.upsert("tm_appearances", pd.DataFrame(apps), keys=["appearance_id"])

    fixtures = pd.DataFrame(
        {
            "match_id": ["m1"],
            "home_team": ["Team00"],
            "away_team": ["Team01"],
            "date": [pd.Timestamp("2025-01-01")],
            "league_code": ["E0"],
        }
    )
    live_fx = pd.DataFrame(
        {
            "fixture_id": [99],
            "date": [pd.Timestamp("2025-01-01", tz="UTC")],
            "home_team": ["Team00"],
            "away_team": ["Team01"],
        }
    )
    lineup = pd.DataFrame(
        {
            "fixture_id": [99, 99],
            "team": ["Team00", "Team01"],
            "player": ["Player A", "Some Player"],
            "slot": ["start", "start"],
        }
    )
    src = _StubAPIFootball(live_fx, {99: lineup})
    out = confirmed_lineup_features(wh, fixtures, api_src=src)
    row = out.iloc[0]
    assert pd.isna(row["sv_away_missing_pct"])  # unresolved club -> genuinely unknown, stays NaN
    assert not pd.isna(row["sv_missing_pct_diff"])  # but the diff must not silently become NaN
    assert row["sv_missing_pct_diff"] == pytest.approx(0.0 - (5_000_000 / 15_000_000))


def test_resolve_fixture_ids_matches_by_team_and_date(warehouse):
    fixtures = pd.DataFrame(
        {"match_id": ["m1"], "home_team": ["Team00"], "away_team": ["Team01"], "date": [pd.Timestamp("2025-01-01")]}
    )
    live_fx = pd.DataFrame(
        {
            "fixture_id": [99],
            "date": [pd.Timestamp("2025-01-01", tz="UTC")],
            "home_team": ["Team00"],
            "away_team": ["Team01"],
        }
    )
    src = _StubAPIFootball(live_fx, {})
    mapping = resolve_fixture_ids(src, fixtures, "E0", 2024)
    assert mapping == {"m1": 99}
