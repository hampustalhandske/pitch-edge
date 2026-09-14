from __future__ import annotations

import duckdb
import pandas as pd
import pytest
import responses

from pitch_edge.data.alt.news import NewsScanner
from pitch_edge.data.sources.openfootball import OpenFootballSource, parse_openfootball_txt
from pitch_edge.data.sources.thesportsdb import TheSportsDBSource
from pitch_edge.data.sources.transfermarkt_open import TransfermarktOpenSource

pytestmark = pytest.mark.unit

SWEDEN_TXT = """= Sweden Superettan 2025

# Date       Sat Mar 29 - Sat Nov 8 2025 (224d)

▪ Matchday 1
  Sat Mar 29 2025
    15:00  GIF Sundsvall           v Helsingborgs IF          2-0 (0-0)
           Sandvikens IF           v Kalmar FF                0-0
    17:00  Örgryte IS              v Utsiktens BK             2-1 (0-0)
  Sun Mar 30
    13:00  Landskrona BoIS         v IK Brage                 2-2 (1-0)

▪ Matchday 30
  Sat Nov 8
    15:00  IK Brage                v Östersunds FK
"""


def test_parse_openfootball_txt_played_and_unplayed():
    played = parse_openfootball_txt(SWEDEN_TXT, "SWE2", "Sweden - Superettan")
    assert len(played) == 4
    assert played.iloc[0]["home_team"] == "GIF Sundsvall" and played.iloc[0]["ht_home_goals"] == 0.0
    assert pd.isna(played.iloc[1]["kickoff_time"]) and played.iloc[1]["home_goals"] == 0.0
    assert str(played.iloc[3]["date"].date()) == "2025-03-30"  # year carried from the header
    assert played["season"].iloc[0] == "2025" and played["country"].iloc[0] == "Sweden"
    both = parse_openfootball_txt(SWEDEN_TXT, "SWE2", "Sweden - Superettan", include_unplayed=True)
    assert len(both) == 5 and pd.isna(both.iloc[4]["home_goals"]) and str(both.iloc[4]["date"].date()) == "2025-11-08"


@responses.activate
def test_fetch_sweden_validates_played_rows(tmp_path):
    base = "https://raw.githubusercontent.com/openfootball/europe/master/sweden"
    responses.add(responses.GET, f"{base}/2025_se1.txt", body=SWEDEN_TXT.replace("Superettan", "Allsvenskan"))
    responses.add(responses.GET, f"{base}/2025_se2.txt", body=SWEDEN_TXT)
    responses.add(responses.GET, f"{base}/2025_se3s.txt", status=404)
    responses.add(responses.GET, f"{base}/2025_se3n.txt", status=404)
    df = OpenFootballSource(cache_dir=tmp_path).fetch_sweden((2025,))
    assert set(df["league_code"]) == {"SWE1", "SWE2"} and len(df) == 8
    assert df["match_id"].is_unique


# Kalshi/Polymarket live-snapshot connectors were removed (see `data/alt/pmxt_archive.py`);
# their coverage is tested in `tests/unit/test_pmxt_archive.py`.


@responses.activate
def test_thesportsdb_events_and_html_fallback(tmp_path):
    base = "https://www.thesportsdb.com/api/v1/json/3"
    responses.add(
        responses.GET,
        f"{base}/eventsseason.php",
        json={
            "events": [
                {
                    "idEvent": "1",
                    "dateEvent": "2025-03-29",
                    "strTime": "14:00:00",
                    "strHomeTeam": "Djurgaarden",
                    "strAwayTeam": "Malmo FF",
                    "intHomeScore": "0",
                    "intAwayScore": "1",
                    "strVenue": "Tele2 Arena",
                    "intRound": "1",
                    "strStatus": "Match Finished",
                }
            ]
        },
    )
    responses.add(responses.GET, f"{base}/searchplayers.php", body="<html>rate limited</html>")
    src = TheSportsDBSource(cache_dir=tmp_path)
    ev = src.season_events(4347, "2025")
    assert len(ev) == 1 and ev.iloc[0]["kickoff_time"] == "14:00" and ev.iloc[0]["away_goals"] == 1
    assert src.team_players("Malmo FF").empty


def test_transfermarkt_open_reads_local_duckdb(tmp_path):
    db = tmp_path / "transfermarkt-datasets.duckdb"
    con = duckdb.connect(str(db))
    con.execute(
        "CREATE TABLE games (game_id INT, date DATE, home_club_name VARCHAR, away_club_name VARCHAR, home_club_goals INT, away_club_goals INT, competition_id VARCHAR)"
    )
    con.execute("INSERT INTO games VALUES (1, '2024-01-01', 'Arsenal', 'Chelsea', 2, 1, 'GB1')")
    con.execute(
        "CREATE TABLE appearances (appearance_id VARCHAR, game_id INT, player_id INT, player_name VARCHAR, date DATE, competition_id VARCHAR, minutes_played INT, goals INT, assists INT, yellow_cards INT, red_cards INT)"
    )
    con.execute("INSERT INTO appearances VALUES ('1_7', 1, 7, 'Bukayo Saka', '2024-01-01', 'GB1', 90, 1, 0, 0, 0)")
    con.execute(
        "CREATE TABLE game_events (game_event_id VARCHAR, game_id INT, club_id INT, club_name VARCHAR, minute INT, type VARCHAR)"
    )
    con.execute(
        "INSERT INTO game_events VALUES ('e1', 1, 11, 'Arsenal', 61, 'Substitutions'), ('e2', 1, 11, 'Arsenal', 75, 'Substitutions')"
    )
    con.close()
    src = TransfermarktOpenSource(cache_dir=tmp_path)
    assert src.download() == db  # cached file, no network
    assert len(src.read_table("games")) == 1
    hist = src.player_match_history("saka")
    assert len(hist) == 1 and hist.iloc[0]["goals"] == 1 and hist.iloc[0]["home_club_name"] == "Arsenal"
    prof = src.substitution_profile(min_games=1)
    assert prof.iloc[0]["avg_first_sub_minute"] == 61 and prof.iloc[0]["avg_subs"] == 2
    with pytest.raises(ValueError):
        src.read_table("nope")


def test_swedish_news_keywords_and_language(tmp_path):
    items = pd.DataFrame(
        {
            "item_id": ["a", "b"],
            "feed": ["sportbladet", "bbc_football"],
            "published_at": pd.to_datetime(["2025-05-01", "2025-05-01"]),
            "title": ["Malmö FF-stjärnan skadad – missar derbyt", "Arsenal name unchanged starting XI"],
            "summary": ["", ""],
            "link": ["", ""],
        }
    )
    scored = NewsScanner(cache_dir=tmp_path).score(items, ["Malmö FF", "Arsenal"])
    assert (
        scored.iloc[0]["is_injury_news"] and scored.iloc[0]["language"] == "sv" and scored.iloc[0]["team"] == "Malmö FF"
    )
    assert scored.iloc[1]["is_lineup_news"] and scored.iloc[1]["language"] == "en"
