"""Transfermarkt open-dataset ingestion into the warehouse from a local DuckDB fixture (no network)."""

from __future__ import annotations

import duckdb
import pytest

from pitch_edge.data.ingest import ingest_transfermarkt_open

pytestmark = pytest.mark.integration


def test_ingest_transfermarkt_open_loads_tables(warehouse, tmp_path, monkeypatch):
    monkeypatch.setenv("PITCH_EDGE_DATA_DIR", str(tmp_path / "data"))
    raw = tmp_path / "data" / "raw" / "transfermarkt_open"
    raw.mkdir(parents=True)
    con = duckdb.connect(str(raw / "transfermarkt-datasets.duckdb"))
    con.execute("CREATE TABLE competitions (competition_id VARCHAR, name VARCHAR)")
    con.execute(
        "INSERT INTO competitions VALUES ('GB1', 'premier-league'), ('GB1', 'premier-league')"
    )  # duplicate key on purpose
    con.execute("CREATE TABLE players (player_id INT, name VARCHAR, position VARCHAR)")
    con.execute("INSERT INTO players VALUES (7, 'Bukayo Saka', 'Attack')")
    con.execute("CREATE TABLE game_events (game_event_id VARCHAR, game_id INT, type VARCHAR, minute INT)")
    con.execute("INSERT INTO game_events VALUES ('e1', 1, 'Substitutions', 61)")
    con.execute(
        "CREATE TABLE game_lineups (id VARCHAR, game_id INT, player_name VARCHAR)"
    )  # key column absent -> first column
    con.execute("INSERT INTO game_lineups VALUES ('l1', 1, 'Saka')")
    con.close()

    n = ingest_transfermarkt_open(warehouse, tables=("competitions", "players", "game_events", "game_lineups", "clubs"))
    assert n == 4  # duplicate competition row collapsed, missing 'clubs' table skipped without aborting
    assert warehouse.count("tm_competitions") == 1
    assert warehouse.count("tm_players") == 1 and warehouse.count("tm_game_lineups") == 1
    assert warehouse.health().set_index("source").loc["transfermarkt_open", "last_status"] == "ok"
