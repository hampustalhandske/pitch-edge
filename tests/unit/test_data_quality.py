"""Duplicate-fixture check and the purge of matches from sources no longer ingested."""

from __future__ import annotations

import pandas as pd
import pytest

from pitch_edge.data.ingest import purge_uningested_sources, store_matches
from pitch_edge.data.quality import find_duplicate_fixtures
from pitch_edge.data.storage import Warehouse

pytestmark = pytest.mark.unit


def _match(match_id: str, source: str, home: str, away: str, day: str = "2023-09-02", league_code: str = "D1"):
    return {
        "match_id": match_id,
        "date": pd.Timestamp(day),
        "league": "Germany - Bundesliga",
        "league_code": league_code,
        "season": "2023/24",
        "home_team": home,
        "away_team": away,
        "home_goals": 1.0,
        "away_goals": 0.0,
        "source": source,
    }


@pytest.fixture
def wh(tmp_path):
    with Warehouse(tmp_path / "wh.duckdb") as w:
        yield w


def test_same_fixture_under_two_spellings_is_reported(wh):
    wh.upsert(
        "matches",
        pd.DataFrame(
            [
                _match("fd_1", "football_data_co_uk", "M'gladbach", "Ein Frankfurt"),
                _match("cfmd_1", "club_football_match_data", "Mgladbach", "Eintracht Frankfurt"),
                _match("fd_2", "football_data_co_uk", "Bayern Munich", "Dortmund"),
            ]
        ),
    )
    dupes = find_duplicate_fixtures(wh)
    assert len(dupes) == 1
    assert {dupes.iloc[0]["match_id_a"], dupes.iloc[0]["match_id_b"]} == {"fd_1", "cfmd_1"}


def test_clean_warehouse_reports_none(wh):
    assert find_duplicate_fixtures(wh).empty  # no matches table yet
    wh.upsert(
        "matches",
        pd.DataFrame(
            [
                _match("fd_1", "football_data_co_uk", "Bayern Munich", "Dortmund"),
                _match("fd_2", "football_data_co_uk", "Mainz", "Ein Frankfurt"),
                _match("fd_3", "football_data_co_uk", "Bayern Munich", "Dortmund", day="2023-09-16"),
                _match("fd_4", "football_data_co_uk", "Bayern Munich", "Dortmund", league_code="D2"),
            ]
        ),
    )
    assert find_duplicate_fixtures(wh).empty


def test_purge_removes_uningested_source_rows_and_their_odds(wh):
    kept = pd.DataFrame([{**_match("fd_1", "football_data_co_uk", "Mainz", "Dortmund"), "PSH": 2.0, "PSD": 3.4}])
    gone = pd.DataFrame([{**_match("cfmd_1", "club_football_match_data", "Mainz", "Dortmund"), "PSH": 2.1, "PSD": 3.3}])
    store_matches(wh, kept)
    store_matches(wh, gone)
    wh.upsert("weather", pd.DataFrame({"match_id": ["fd_1", "cfmd_1"], "temp_c": [14.0, 14.0]}))
    assert wh.count("matches") == 2 and wh.count("odds") == 4

    assert purge_uningested_sources(wh) == 1
    assert wh.read("matches")["match_id"].tolist() == ["fd_1"]
    assert set(wh.read("odds")["match_id"]) == {"fd_1"}
    assert wh.read("weather")["match_id"].tolist() == ["fd_1"]
    assert purge_uningested_sources(wh) == 0  # idempotent
