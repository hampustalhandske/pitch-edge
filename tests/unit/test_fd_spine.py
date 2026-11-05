"""The warehouse spine's football-data rows built from the `lake_fd_matches` mirror (fake lake, no network)."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, date, datetime, timedelta

import pandas as pd
import pytest
from typer.testing import CliRunner

from pitch_edge.cli import app
from pitch_edge.config import get_settings
from pitch_edge.data.fd_vocab import odds_wide_to_long
from pitch_edge.data.ingest import build_football_data_spine, ingest_everything, store_matches
from pitch_edge.data.storage import Warehouse
from pitch_edge.lake.freshness import MAX_LAG_DAYS, football_data_lag, lake_newest_dates
from pitch_edge.lake.mirror import MIRRORS, sync_mirror
from pitch_edge.lake.spine import (
    STATE_RUN_ID,
    build_fd_spine,
    mirror_windows,
    odds_columns,
    spine_frame,
)

pytestmark = pytest.mark.unit
runner = CliRunner()
SPEC = MIRRORS["fd_matches"]
UNTIL = date(2023, 12, 31)

# The matches and odds the football-data CSV ingestion stored for these fixtures (captured before it was removed).
GOLDEN_MATCH_IDS = {
    "fd_E0_2324_20230812_Arsenal_ManUnited",
    "fd_E0_2324_20230813_Chelsea_Liverpool",
    "fd_E0_2324_20230819_ManUnited_Chelsea",
    "fd_E0_2324_20230820_Liverpool_Arsenal",
    "fdx_SWE_20230401_MalmoFF_AIK",
    "fdx_SWE_20230402_Hammarby_Djurgarden",
}
GOLDEN_ODDS_PER_MATCH = {i: 11 for i in GOLDEN_MATCH_IDS if i.startswith("fd_")} | {
    i: 12 for i in GOLDEN_MATCH_IDS if i.startswith("fdx_")
}
GOLDEN_ARSENAL_MATCH = {
    "date": pd.Timestamp("2023-08-12"),
    "country": "England",
    "league": "England - Premier League",
    "league_code": "E0",
    "season": "2023/24",
    "home_team": "Arsenal",
    "away_team": "Man United",
    "home_goals": 2.0,
    "away_goals": 1.0,
    "source": "football_data_co_uk",
    "referee": "M Oliver",
    "kickoff_time": "20:00",
    "ht_home_goals": 1,
    "ht_away_goals": 0,
    "home_shots": 15,
    "away_shots": 9,
    "home_shots_on_target": 6,
    "away_shots_on_target": 3,
    "home_fouls": 10,
    "away_fouls": 12,
    "home_corners": 7,
    "away_corners": 3,
    "home_yellows": 1,
    "away_yellows": 2,
    "home_reds": 0,
    "away_reds": 0,
}
GOLDEN_ARSENAL_ODDS = {  # (bookmaker, market, side, is_closing) -> price
    ("B365", "1x2", "home", False): 1.8,
    ("B365", "1x2", "draw", False): 3.6,
    ("B365", "1x2", "away", False): 4.5,
    ("B365", "totals_2.5", "over", False): 1.7,
    ("B365", "totals_2.5", "under", False): 2.2,
    ("PS", "1x2", "home", False): 1.85,
    ("PS", "1x2", "draw", False): 3.7,
    ("PS", "1x2", "away", False): 4.4,
    ("PS", "1x2", "home", True): 1.82,
    ("PS", "1x2", "draw", True): 3.75,
    ("PS", "1x2", "away", True): 4.55,
}
GOLDEN_MALMO_ODDS = {  # extra-league files carry closing prices only
    ("PS", "1x2", "home", True): 1.8,
    ("PS", "1x2", "draw", True): 3.6,
    ("PS", "1x2", "away", True): 4.6,
    ("Max", "1x2", "home", True): 1.9,
    ("Avg", "1x2", "away", True): 4.4,
    ("BFE", "1x2", "draw", True): 3.7,
}


def _synced(wh, lake, since="2023-01-01"):
    sync_mirror(wh, lake, SPEC, since=since, until=UNTIL, sleep=lambda s: None)
    return wh


@pytest.fixture
def mirrored(warehouse, fd_lake):
    return _synced(warehouse, fd_lake)


def _spine(wh) -> tuple[pd.DataFrame, pd.DataFrame]:
    m = wh.read("matches").sort_values("match_id").reset_index(drop=True)
    o = wh.read("odds").sort_values(["match_id", "bookmaker", "market", "side", "is_closing"]).reset_index(drop=True)
    return m, o


def test_spine_from_lake_rows_matches_the_csv_ingestion_output(mirrored):
    report = build_fd_spine(mirrored)
    assert report["matches"] == 6 and report["rebuilt"] == 1
    matches, odds = _spine(mirrored)
    assert set(matches["match_id"]) == GOLDEN_MATCH_IDS
    assert odds.groupby("match_id").size().to_dict() == GOLDEN_ODDS_PER_MATCH
    row = matches.set_index("match_id").loc["fd_E0_2324_20230812_Arsenal_ManUnited"]
    for col, want in GOLDEN_ARSENAL_MATCH.items():
        assert row[col] == want, col
    assert list(matches.columns) == [  # no attendance column when the lake has none
        "match_id", "date", "country", "league", "league_code", "season", "home_team", "away_team",
        "home_goals", "away_goals", "source", "referee", "kickoff_time", "ht_home_goals", "ht_away_goals",
        "home_shots", "away_shots", "home_shots_on_target", "away_shots_on_target", "home_fouls", "away_fouls",
        "home_corners", "away_corners", "home_yellows", "away_yellows", "home_reds", "away_reds",
    ]  # fmt: skip
    assert list(odds.columns) == ["match_id", "bookmaker", "market", "side", "price", "is_closing", "snapshot_ts"]
    for match_id, golden in (
        ("fd_E0_2324_20230812_Arsenal_ManUnited", GOLDEN_ARSENAL_ODDS),
        ("fdx_SWE_20230401_MalmoFF_AIK", GOLDEN_MALMO_ODDS),
    ):
        got = odds[odds["match_id"] == match_id].set_index(["bookmaker", "market", "side", "is_closing"])["price"]
        for key, price in golden.items():
            assert got[key] == price, key
    swe = matches.set_index("match_id").loc["fdx_SWE_20230401_MalmoFF_AIK"]
    assert swe["league"] == "Sweden - Allsvenskan" and swe["season"] == "2023" and swe["league_code"] == "SWE"


def test_odds_stamps_keep_known_at_semantics(mirrored):
    build_fd_spine(mirrored)
    matches, odds = _spine(mirrored)
    stamped = odds.merge(matches[["match_id", "date"]], on="match_id")
    # every stamp is the match date at 00:00 UTC, the same instant as the mirror's `open_known_by`
    assert (stamped["snapshot_ts"] == stamped["date"]).all()
    known = mirrored.query(
        "SELECT match_date, CAST(open_known_by AT TIME ZONE 'UTC' AS TIMESTAMP) AS t FROM lake_fd_matches"
    )
    assert (pd.to_datetime(known["match_date"]) == known["t"]).all()
    # opens and closes are distinguishable; an extra-league file has no opens at all
    assert set(odds["is_closing"]) == {True, False}
    assert odds[odds["match_id"].str.startswith("fdx_")]["is_closing"].all()


def test_rebuild_is_idempotent_and_skips_unchanged_windows(mirrored):
    first = build_fd_spine(mirrored)
    m1, o1 = _spine(mirrored)
    again = build_fd_spine(mirrored)
    assert again["rebuilt"] == 0 and again["skipped"] == first["windows"] and again["matches"] == 0
    forced = build_fd_spine(mirrored, force=True)
    assert forced["rebuilt"] == first["windows"]
    m2, o2 = _spine(mirrored)
    pd.testing.assert_frame_equal(m1, m2)
    pd.testing.assert_frame_equal(o1, o2)


def test_resync_that_changes_a_row_rebuilds_that_window_only(mirrored, fd_lake):
    build_fd_spine(mirrored)
    fd_lake.rows[0]["ft_home_goals"] = 5
    fd_lake.rows[0]["odds_all"] = json.dumps({"PSCH": 1.5, "PSCD": 4.0, "PSCA": 6.0})
    _synced(mirrored, fd_lake)
    report = build_fd_spine(mirrored)
    assert report["rebuilt"] == 1
    matches, odds = _spine(mirrored)
    assert matches.set_index("match_id").loc["fd_E0_2324_20230812_Arsenal_ManUnited", "home_goals"] == 5
    assert (odds["match_id"] == "fd_E0_2324_20230812_Arsenal_ManUnited").sum() == 3  # old open prices are gone


def test_rows_dropped_by_the_lake_leave_the_spine(mirrored, fd_lake):
    build_fd_spine(mirrored)
    fd_lake.rows = fd_lake.rows[1:]
    _synced(mirrored, fd_lake)
    build_fd_spine(mirrored)
    matches, odds = _spine(mirrored)
    assert "fd_E0_2324_20230812_Arsenal_ManUnited" not in set(matches["match_id"])
    assert "fd_E0_2324_20230812_Arsenal_ManUnited" not in set(odds["match_id"])


def test_a_failed_window_rolls_back_and_the_rerun_resumes(warehouse, fd_lake, monkeypatch):
    fd_lake.rows.append(
        {**fd_lake.rows[0], "match_date": date(2024, 8, 17), "home_team": "Spurs", "season": "2024-2025"}
    )
    sync_mirror(warehouse, fd_lake, SPEC, since="2023-01-01", until=date(2024, 12, 31), sleep=lambda s: None)
    real = store_matches
    calls = {"n": 0}

    def flaky(wh, df):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("lid closed")
        return real(wh, df)

    monkeypatch.setattr("pitch_edge.data.ingest.store_matches", flaky)
    with pytest.raises(RuntimeError):
        build_fd_spine(warehouse)
    done = warehouse.query("SELECT window_start FROM lake_sync_state WHERE run_id = ?", [STATE_RUN_ID])
    assert list(done["window_start"]) == ["2023-01-01"]  # the checkpoint exists only for the committed window
    assert warehouse.count("matches") == 6
    monkeypatch.setattr("pitch_edge.data.ingest.store_matches", real)
    report = build_fd_spine(warehouse)
    assert report["skipped"] == 1 and report["rebuilt"] == 1 and warehouse.count("matches") == 7


def test_other_sources_survive_and_a_secondary_duplicate_is_replaced(mirrored):
    secondary = pd.DataFrame(
        [
            {
                "match_id": "cfmd_x", "date": pd.Timestamp("2023-08-12"), "league": "x", "league_code": "E0",
                "season": "2023/24", "home_team": "Arsenal", "away_team": "Man United", "home_goals": 2.0,
                "away_goals": 1.0, "source": "club_football_match_data", "MktH": 1.9, "MktD": 3.5, "MktA": 4.0,
            },
            {
                "match_id": "cfmd_keep", "date": pd.Timestamp("2023-08-12"), "league": "x", "league_code": "E0",
                "season": "2023/24", "home_team": "Brighton", "away_team": "Luton", "home_goals": 4.0,
                "away_goals": 1.0, "source": "club_football_match_data", "MktH": 1.3, "MktD": 5.0, "MktA": 9.0,
            },
        ]
    )  # fmt: skip
    store_matches(mirrored, secondary)
    build_fd_spine(mirrored)
    matches, odds = _spine(mirrored)
    assert set(matches["match_id"]) == GOLDEN_MATCH_IDS | {"cfmd_keep"}
    assert (odds["match_id"] == "cfmd_keep").sum() == 3 and not (odds["match_id"] == "cfmd_x").any()
    assert (matches.set_index("match_id").loc["cfmd_keep", "source"]) == "club_football_match_data"


def test_spine_frame_filters_and_naming():
    base = {
        "division": "E0", "country": "England", "league": "Premier League", "season": "2023-2024",
        "match_date": date(2023, 8, 12), "kickoff_time": None, "home_team": " Arsenal ", "away_team": "Man United",
        "ft_home_goals": 1, "ft_away_goals": 0, "ht_home_goals": None, "ht_away_goals": None, "referee": None,
        "home_shots": None, "away_shots": None, "home_shots_on_target": None, "away_shots_on_target": None,
        "home_fouls": None, "away_fouls": None, "home_corners": None, "away_corners": None, "home_yellow": None,
        "away_yellow": None, "home_red": None, "away_red": None, "attendance": None, "odds_all": None,
    }  # fmt: skip
    rows = pd.DataFrame(
        [
            base,
            {**base, "home_team": "Chelsea", "ft_home_goals": None},  # not played
            {**base, "division": "ZZ9"},  # unknown division
            {**base, "season": "2004-2005", "match_date": date(2005, 3, 1), "home_team": "Old"},  # before 2005/06
            {**base, "division": "SWZ", "country": "Switzerland", "league": "Super League", "season": "2019-2020"},
            {**base, "division": "USA", "league": "MLS", "season": "2013", "home_team": "LA Galaxy"},
        ]
    )
    out = spine_frame(rows).set_index("match_id")
    assert set(out.index) == {
        "fd_E0_2324_20230812_Arsenal_ManUnited",
        "fdx_SWZ_20230812_Arsenal_ManUnited",
        "fdx_USA_20230812_LAGalaxy_ManUnited",
    }
    assert out.loc["fdx_SWZ_20230812_Arsenal_ManUnited", "season"] == "2019/20"
    assert out.loc["fdx_USA_20230812_LAGalaxy_ManUnited", "season"] == "2013"
    assert out.loc["fd_E0_2324_20230812_Arsenal_ManUnited", "home_team"] == "Arsenal"
    assert spine_frame(rows.iloc[0:0]).empty


def test_odds_columns_keep_only_1x2_and_totals_numbers():
    raw = json.dumps({"PSH": 1.5, "PSCH": 1.4, "B365>2.5": 1.9, "BbMxH": 2.0, "AHh": -0.5, "B365AHH": 1.9, "WHD": "x"})
    assert odds_columns(raw) == {"PSH": 1.5, "PSCH": 1.4, "B365>2.5": 1.9}
    assert odds_columns(None) == {} and odds_columns("not json") == {}


def test_odds_wide_to_long_marks_closing_and_totals():
    wide = pd.DataFrame(
        {"match_id": ["m"], "date": [pd.Timestamp("2023-08-12")], "PSH": [1.85], "PSCH": [1.82], "B365>2.5": [1.7],
         "PC<2.5": [2.5], "AvgD": [0.9]}
    )  # fmt: skip
    long = odds_wide_to_long(wide)
    ps = long[long["bookmaker"] == "PS"].set_index("is_closing")["price"]
    assert ps[False] == 1.85 and ps[True] == 1.82
    totals = long[long["market"] == "totals_2.5"]
    assert set(totals["side"]) == {"over", "under"} and bool(totals.loc[totals["bookmaker"] == "P", "is_closing"].all())
    assert (long["price"] > 1).all()


def test_ingest_everything_builds_the_spine_from_the_mirror(mirrored, monkeypatch):
    import pitch_edge.data.ingest as ingest

    for name in (
        "ingest_statsbomb", "ingest_venues_and_weather", "ingest_sweden", "ingest_news", "ingest_pmxt_soccer_markets",
        "ingest_transfermarkt_open", "ingest_thesportsdb_sweden", "ingest_wikipedia_attention",
        "ingest_referee_announcements", "ingest_espn_soccer_data",
    ):  # fmt: skip
        monkeypatch.setattr(ingest, name, lambda *a, **k: 0)
    report = ingest_everything(mirrored)
    assert report["football_data_spine"] == 6 and mirrored.count("matches") == 6
    assert mirrored.health().set_index("source").loc["football_data_spine", "last_status"] == "ok"
    assert not hasattr(ingest, "ingest_football_data")


def test_build_with_no_mirror_is_a_logged_no_op(warehouse):
    assert build_football_data_spine(warehouse) == 0
    assert not warehouse.table_exists("matches")
    assert mirror_windows(warehouse) == []


# ---------------------------------------------------------------------------------------------- lag in health
def test_lag_table_and_warnings(mirrored):
    build_fd_spine(mirrored)
    now = datetime.now(UTC).replace(tzinfo=None)
    ok = football_data_lag(mirrored, {"E0": date(2023, 8, 20)}, ["E0"], now=now)
    assert ok.warnings == []
    row = ok.table.iloc[0]
    assert row["mirror_newest"] == date(2023, 8, 20) and row["spine_newest"] == date(2023, 8, 20)
    assert row["lake_gap_days"] == 0 and ok.last_sync is not None and ok.last_spine_build is not None

    behind = football_data_lag(mirrored, {"E0": date(2023, 8, 20 + MAX_LAG_DAYS + 1)}, ["E0"], now=now)
    assert any("E0: mirror ends 2023-08-20, lake 2023-08-24" in w for w in behind.warnings)
    within = football_data_lag(mirrored, {"E0": date(2023, 8, 20 + MAX_LAG_DAYS)}, ["E0"], now=now)
    assert within.warnings == []

    stale = football_data_lag(mirrored, None, ["E0"], now=now + timedelta(days=3))
    assert any("last synced" in w for w in stale.warnings)
    assert pd.isna(stale.table.iloc[0]["lake_newest"])


def test_lag_warns_when_the_spine_is_behind_the_mirror_or_unsynced(mirrored, tmp_path):
    lag = football_data_lag(mirrored, None, ["E0"])  # mirror synced, spine never built
    assert any("mirror has matches but the spine has none" in w for w in lag.warnings)
    build_fd_spine(mirrored)
    mirrored.query("SELECT 1")
    mirrored._conn.execute("DELETE FROM matches WHERE match_id = 'fd_E0_2324_20230820_Liverpool_Arsenal'")
    behind = football_data_lag(mirrored, None, ["E0"])
    assert any("spine ends 2023-08-19, mirror 2023-08-20" in w for w in behind.warnings)
    with Warehouse(tmp_path / "other.duckdb") as empty_wh:
        empty = football_data_lag(empty_wh, None, ["E0"])
    assert any("empty or missing" in w for w in empty.warnings)


def test_lake_newest_dates_is_one_aggregate_query(fd_lake):
    got = lake_newest_dates(fd_lake, ["E0", "SWE", "D1"])
    assert got == {"E0": date(2023, 8, 20), "SWE": date(2023, 4, 2)}
    assert len(fd_lake.calls) == 1 and "GROUP BY" in fd_lake.calls[0][0] and "fd_matches" in fd_lake.calls[0][0]


def test_health_command_reports_lag(fd_lake, monkeypatch):
    with Warehouse(get_settings().db_path) as wh:
        _synced(wh, fd_lake)
        build_fd_spine(wh)
    monkeypatch.setenv("LAKEHOUSE_API_TOKEN", "t")
    monkeypatch.setattr("pitch_edge.lake.client.Lake", lambda: fd_lake)
    monkeypatch.setattr("pitch_edge.cli._backtest_leagues", lambda: ["E0"])
    res = runner.invoke(app, ["health"], env={"COLUMNS": "200"})
    assert res.exit_code == 0, res.output
    assert "Football-data lag" in res.output and "mirror last sync" in res.output
    assert "football-data lag: ok" in res.output and "WARN" not in res.output
    fd_lake.rows.append({**fd_lake.rows[0], "match_date": date(2023, 9, 30), "home_team": "Spurs"})  # lake moved on
    res = runner.invoke(app, ["health"], env={"COLUMNS": "200"})
    assert res.exit_code == 0, res.output
    assert "WARN football-data lag: E0: mirror ends 2023-08-20, lake 2023-09-30 (41 d behind, limit 3 d)" in res.output


# ---------------------------------------------------------------------------------------------- lake-sync CLI
def test_lake_sync_command_syncs_then_builds_the_spine(warehouse, fd_lake, monkeypatch):
    warehouse.close()
    monkeypatch.setenv("LAKEHOUSE_API_TOKEN", "t")
    monkeypatch.setattr("pitch_edge.lake.client.Lake", lambda: fd_lake)
    res = runner.invoke(app, ["lake-sync", "--only", "fd_matches", "--since", "2023-01-01"])
    assert res.exit_code == 0, res.output
    assert "football-data spine: 6 matches written" in res.output
    with Warehouse(get_settings().db_path, read_only=True) as wh:
        assert wh.count("matches") == 6 and wh.count("lake_fd_matches") == 6


def test_lake_sync_skips_cleanly_when_the_warehouse_is_locked(tmp_path, fd_lake, monkeypatch):
    db = get_settings().db_path
    db.parent.mkdir(parents=True, exist_ok=True)
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            f"import duckdb,time; c=duckdb.connect({str(db)!r}); print('ready', flush=True); time.sleep(120)",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "ready"
        monkeypatch.setenv("LAKEHOUSE_API_TOKEN", "t")
        monkeypatch.setattr("pitch_edge.lake.client.Lake", lambda: fd_lake)
        res = runner.invoke(app, ["lake-sync", "--only", "fd_matches", "--skip-if-busy"])
        assert res.exit_code == 0, res.output
        assert "skipped: warehouse busy" in res.output
        assert fd_lake.calls == []  # nothing was pulled
        loud = runner.invoke(app, ["lake-sync", "--only", "fd_matches"])
        assert loud.exit_code != 0  # without the flag a lock is an error
    finally:
        holder.kill()
        holder.wait()
