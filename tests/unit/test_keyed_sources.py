"""Optional keyed connectors: disabled without a key, parse correctly with one (mocked)."""

from __future__ import annotations

import pytest
import responses

from pitch_edge.data.alt.api_football import APIFootballSource
from pitch_edge.data.sources.everysport import EverysportSource

pytestmark = pytest.mark.unit


def test_api_football_disabled_without_key(tmp_path, monkeypatch):
    monkeypatch.delenv("API_FOOTBALL_KEY", raising=False)
    src = APIFootballSource(cache_dir=tmp_path)
    assert not src.enabled and src.injuries("E0", 2025).empty and src.lineups(1).empty


@responses.activate
def test_api_football_parses_injuries_lineups_subs(tmp_path):
    base = "https://v3.football.api-sports.io"
    responses.add(
        responses.GET,
        f"{base}/injuries",
        json={
            "response": [
                {
                    "fixture": {"id": 10, "date": "2025-09-01T14:00:00+00:00"},
                    "team": {"name": "Arsenal"},
                    "player": {"name": "Saka", "type": "Missing Fixture", "reason": "Hamstring"},
                }
            ]
        },
    )
    responses.add(
        responses.GET,
        f"{base}/fixtures/lineups",
        json={
            "response": [
                {
                    "team": {"name": "Arsenal"},
                    "formation": "4-3-3",
                    "startXI": [{"player": {"name": "Raya", "pos": "G"}}],
                    "substitutes": [{"player": {"name": "Nwaneri", "pos": "M"}}],
                }
            ]
        },
    )
    responses.add(
        responses.GET,
        f"{base}/fixtures/events",
        json={
            "response": [
                {
                    "time": {"elapsed": 63},
                    "team": {"name": "Arsenal"},
                    "player": {"name": "Odegaard"},
                    "assist": {"name": "Nwaneri"},
                }
            ]
        },
    )
    src = APIFootballSource(api_key="k", cache_dir=tmp_path)
    inj = src.injuries("E0", 2025)
    assert len(inj) == 1 and inj.iloc[0]["reason"] == "Hamstring"
    lu = src.lineups(10)
    assert set(lu["slot"]) == {"start", "bench"} and lu.iloc[0]["formation"] == "4-3-3"
    subs = src.substitutions(10)
    assert subs.iloc[0]["player_in"] == "Nwaneri" and subs.iloc[0]["minute"] == 63
    assert responses.calls[0].request.headers["x-apisports-key"] == "k"


# The Odds API connector was removed (see `data/alt/pmxt_archive.py` for the market-probability
# source going forward).


def test_everysport_disabled_without_key(tmp_path, monkeypatch):
    monkeypatch.delenv("EVERYSPORT_API_KEY", raising=False)
    src = EverysportSource(cache_dir=tmp_path)
    assert not src.enabled and src.fetch_matches(league_id=1).empty and src.leagues().empty


@responses.activate
def test_everysport_parses_events(tmp_path):
    responses.add(
        responses.GET,
        "https://api.everysport.com/v1/events",
        json={
            "events": [
                {
                    "id": 5,
                    "startDate": "2025-05-10T15:00:00+02:00",
                    "league": {"name": "Superettan"},
                    "homeTeam": {"name": "Örgryte IS"},
                    "visitingTeam": {"name": "Kalmar FF"},
                    "homeTeamScore": 2,
                    "visitingTeamScore": 1,
                },
                {
                    "id": 6,
                    "startDate": "2025-05-11T15:00:00+02:00",
                    "league": {"name": "Superettan"},
                    "homeTeam": {"name": "IK Brage"},
                    "visitingTeam": {"name": "Umeå FC"},
                    "homeTeamScore": None,
                    "visitingTeamScore": None,
                },
            ]
        },
    )
    df = EverysportSource(api_key="k", cache_dir=tmp_path).fetch_matches(league_id=99)
    assert len(df) == 1 and df.iloc[0]["home_team"] == "Örgryte IS" and df.iloc[0]["league_code"] == "ES99"
    assert df.iloc[0]["country"] == "Sweden" and df.iloc[0]["season"] == "2025"
