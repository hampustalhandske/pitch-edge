"""Transfermarkt rotation/referee context: strictly pre-match, joins by (date, clubs), activates referee features."""

from __future__ import annotations

import pandas as pd
import pytest

from pitch_edge.features.build import FeatureBuilder, feature_columns
from pitch_edge.features.context import club_id_map, load_context, transfermarkt_context

pytestmark = pytest.mark.unit


def _seed_tm(wh):
    wh.upsert(
        "tm_clubs",
        pd.DataFrame(
            {
                "club_id": ["1", "2", "3"],
                "name": ["Team00 FC", "Team01 FC", "Team02 FC"],
                "last_season": [2025, 2025, 2025],
            }
        ),
        keys=["club_id"],
    )
    games = pd.DataFrame(
        {
            "game_id": ["g1", "g2", "g3", "g4"],
            "date": pd.to_datetime(["2020-03-01", "2020-03-04", "2020-03-07", "2020-02-20"]),
            "home_club_id": [1, 1, 1, 2],
            "away_club_id": [2, 3, 2, 3],
            "competition_type": ["domestic_league", "domestic_cup", "domestic_league", "domestic_league"],
            "referee": ["A Ref", "B Ref", "C Ref", "A Ref"],
        }
    )
    wh.upsert("tm_games", games, keys=["game_id"])
    apps = []
    for gid, d, club in (("g1", "2020-03-01", 1), ("g2", "2020-03-04", 1), ("g1", "2020-03-01", 2)):
        for p in range(11):
            apps.append(
                {
                    "appearance_id": f"{gid}_{club}_{p}",
                    "game_id": gid,
                    "player_club_id": club,
                    "date": pd.Timestamp(d),
                    "minutes_played": 90,
                }
            )
    wh.upsert("tm_appearances", pd.DataFrame(apps), keys=["appearance_id"])


def test_transfermarkt_context_rotation_and_referee(warehouse):
    _seed_tm(warehouse)
    matches = pd.DataFrame(
        {"match_id": ["x1"], "date": [pd.Timestamp("2020-03-07")], "home_team": ["Team00"], "away_team": ["Team01"]}
    )
    assert club_id_map(warehouse, ["Team00", "Team01"]) == {"Team00": 1, "Team01": 2}
    ctx = transfermarkt_context(warehouse, matches).iloc[0]
    # home played league (Mar 1) + cup (Mar 4) in the 7 days before Mar 7: 2 x 11 x 90 / 11 = 180 minutes per starter
    assert ctx["rot_home_minutes_7d"] == pytest.approx(180.0)
    assert ctx["rot_home_days_since_any"] == 3 and ctx["rot_home_midweek_cup"] == 1
    assert ctx["rot_away_minutes_7d"] == pytest.approx(90.0) and ctx["rot_away_midweek_cup"] == 0
    assert ctx["tm_referee"] == "C Ref"  # the Mar 7 game itself, never a later one


def test_context_activates_referee_features_in_builder(warehouse, synthetic_league_matches):
    m = synthetic_league_matches.drop(columns=["referee"]).copy()
    ctx = pd.DataFrame(
        {"match_id": m["match_id"], "tm_referee": synthetic_league_matches["referee"], "rot_home_minutes_7d": 90.0}
    )
    f = FeatureBuilder().build(m, context=ctx)
    assert f["ref_cards_per_game"].notna().any() and "rot_home_minutes_7d" in feature_columns(f)
    f0 = FeatureBuilder().build(m)
    assert "ref_cards_per_game" not in f0  # no referee column on the bare spine -> no referee features at all


def test_load_context_none_without_tables(warehouse):
    m = pd.DataFrame({"match_id": ["x"], "date": [pd.Timestamp("2020-01-01")], "home_team": ["A"], "away_team": ["B"]})
    assert load_context(warehouse, m) is None
