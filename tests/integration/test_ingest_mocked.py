"""Ingestion orchestrator against mocked endpoints: health logging, isolation of failures, Elo attach."""

from __future__ import annotations

import json

import pandas as pd
import pytest
import responses

from pitch_edge.data.ingest import (
    ingest_club_elo,
    ingest_club_football_match_data,
    ingest_football_data,
    ingest_news,
    ingest_openfootball,
    ingest_polymarket,
    ingest_statsbomb,
)

pytestmark = pytest.mark.integration

FD = "https://www.football-data.co.uk"


@responses.activate
def test_football_data_ingest_logs_health_and_isolates_failures(
    warehouse, football_data_csv_bytes, extra_league_csv_bytes
):
    responses.add(responses.GET, f"{FD}/mmz4281/2324/E0.csv", body=football_data_csv_bytes)
    responses.add(responses.GET, f"{FD}/mmz4281/2324/D1.csv", status=404)
    responses.add(responses.GET, f"{FD}/new/SWE.csv", body=extra_league_csv_bytes)
    responses.add(responses.GET, f"{FD}/new/ARG.csv", status=500)  # counts toward the host circuit breaker
    for stem in ("AUT", "BRA", "CHN", "DNK", "FIN", "IRL", "JPN", "MEX", "NOR", "POL", "ROU", "RUS", "SWZ", "USA"):
        responses.add(responses.GET, f"{FD}/new/{stem}.csv", status=404)  # 404s do not trip the breaker
    counts = ingest_football_data(warehouse, leagues=["E0", "D1"], seasons=[2023])
    assert counts["E0"] == 4 and counts["D1"] == 0 and counts["SWE"] == 2
    h = warehouse.health().set_index("source")
    assert h.loc["football_data_co_uk:E0", "last_status"] == "ok"
    assert h.loc["football_data_co_uk:new/ARG", "last_status"] == "error"
    assert warehouse.count("matches") == 6


@responses.activate
def test_cfmd_attaches_elo_to_existing_spine_rows(warehouse, football_data_csv_bytes):
    responses.add(responses.GET, f"{FD}/mmz4281/2324/E0.csv", body=football_data_csv_bytes)
    ingest_football_data(warehouse, leagues=["E0"], seasons=[2023], extra_leagues=False)
    cfmd = (
        "Division,MatchDate,MatchTime,HomeTeam,AwayTeam,HomeElo,AwayElo,Form3Home,Form5Home,Form3Away,Form5Away,FTHome,FTAway,FTResult\n"
        "E0,2023-08-12,,Arsenal,Man United,1900.5,1850.2,0,2.0,0,1.5,2,1,H\n"
        "SP1,2023-08-12,,Barcelona,Getafe,1950.0,1650.0,0,0,0,0,0,0,D\n"
    )
    responses.add(
        responses.GET,
        "https://raw.githubusercontent.com/xgabora/Club-Football-Match-Data-2000-2025/main/data/Matches.csv",
        body=cfmd,
    )
    inserted = ingest_club_football_match_data(warehouse)
    assert inserted == 1  # Barcelona row is new; Arsenal row already in spine
    row = warehouse.read("matches", "home_team = 'Arsenal'").iloc[0]
    assert row["home_elo"] == 1900.5 and row["away_form5"] == 1.5


@responses.activate
def test_club_elo_ratings_resolved_to_canonical_names(warehouse, football_data_csv_bytes):
    responses.add(responses.GET, f"{FD}/mmz4281/2324/E0.csv", body=football_data_csv_bytes)
    ingest_football_data(warehouse, leagues=["E0"], seasons=[2023], extra_leagues=False)
    responses.add(
        responses.GET,
        "http://api.clubelo.com/2024-01-01",
        body="Rank,Club,Country,Level,Elo,From,To\n1,ManUnited,ENG,1,1800.1,2024-01-01,2024-01-05\n2,Arsenal,ENG,1,1900.2,2024-01-01,2024-01-05\n",
    )
    n = ingest_club_elo(warehouse, dates=["2024-01-01"])
    assert n == 2
    tr = warehouse.read("team_ratings").set_index("team")
    assert tr.loc["Man United", "elo"] == 1800.1


@responses.activate
def test_openfootball_statsbomb_news_polymarket(warehouse, statsbomb_events_df):
    of = {
        "name": "x",
        "matches": [
            {"round": "1", "date": "2023-08-12", "team1": "Arsenal FC", "team2": "Chelsea FC", "score": {"ft": [2, 1]}},
            {"round": "2", "date": "2023-08-19", "team1": "Chelsea FC", "team2": "Arsenal FC"},
        ],
    }
    responses.add(
        responses.GET, "https://raw.githubusercontent.com/openfootball/football.json/master/2023-24/en.1.json", json=of
    )
    assert ingest_openfootball(warehouse, leagues=["en.1"], seasons=[2023]) == 1

    sb = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
    matches = [
        {
            "match_id": 999,
            "match_date": "2020-01-01",
            "kick_off": "20:00:00.000",
            "home_score": 1,
            "away_score": 0,
            "competition": {"country_name": "Test", "competition_name": "Cup"},
            "season": {"season_name": "2019/2020"},
            "home_team": {"home_team_name": "Home FC"},
            "away_team": {"away_team_name": "Away FC"},
            "referee": {"name": "R"},
            "stadium": {"name": "S"},
        }
    ]
    responses.add(responses.GET, f"{sb}/matches/1/2.json", json=matches)
    events = [
        {
            "id": "ev1",
            "index": 1,
            "period": 1,
            "minute": 5,
            "second": 0,
            "type": {"name": "Pass"},
            "team": {"name": "Home FC"},
            "player": {"name": "P", "id": 1},
            "location": [50, 40],
            "pass": {"end_location": [60, 40], "recipient": {"name": "Q", "id": 2}, "length": 10.0},
        },
        {
            "id": "ev2",
            "index": 2,
            "period": 1,
            "minute": 6,
            "second": 0,
            "type": {"name": "Shot"},
            "team": {"name": "Home FC"},
            "player": {"name": "Q", "id": 2},
            "location": [100, 40],
            "shot": {"statsbomb_xg": 0.3, "outcome": {"name": "Goal"}},
        },
    ]
    responses.add(responses.GET, f"{sb}/events/999.json", json=events)
    assert ingest_statsbomb(warehouse, competitions=[(1, 2)]) == 1
    assert warehouse.count("statsbomb_events") == 2

    rss = b"<rss><channel><item><title>Arsenal injury blow</title><guid>g</guid><link>l</link><description>d</description></item></channel></rss>"
    for url in (
        "https://feeds.bbci.co.uk/sport/football/rss.xml",
        "https://www.theguardian.com/football/rss",
        "https://www.skysports.com/rss/12040",
        "https://www.espn.com/espn/rss/soccer/news",
    ):
        responses.add(responses.GET, url, body=rss)
    assert ingest_news(warehouse) == 1
    assert warehouse.read("news_items").iloc[0]["is_injury_news"]

    pm = [
        {
            "id": "7",
            "question": "Will Arsenal win?",
            "outcomes": json.dumps(["Yes", "No"]),
            "outcomePrices": json.dumps(["0.3", "0.7"]),
        }
    ]
    responses.add(responses.GET, "https://gamma-api.polymarket.com/markets", json=pm)
    assert ingest_polymarket(warehouse) == 2
    assert pd.notna(warehouse.read("market_snapshots").iloc[0]["decimal_odds"])
    assert set(warehouse.health()["source"]) >= {"openfootball", "statsbomb_open_data", "news_rss", "polymarket"}
