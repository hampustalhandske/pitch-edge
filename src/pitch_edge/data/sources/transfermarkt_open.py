"""Transfermarkt open dataset (github.com/dcaribou/transfermarkt-datasets, MIT).

A community-maintained, openly licensed extract — we never scrape transfermarkt.de
itself. One ~210 MB DuckDB file with 13 tables: games (89k), game_events (1.27M:
substitutions, cards, goals), appearances (1.9M player-match rows with minutes,
goals, assists, cards), game_lineups, players (50k, position/foot/DOB), player
valuations (656k), clubs, transfers, competitions. This is the player-history /
substitution / squad layer for the top European leagues + UCL/EL.
"""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import pandas as pd

from pitch_edge.config import get_settings
from pitch_edge.data.http import CachedHttpClient

logger = logging.getLogger(__name__)

DUCKDB_URL = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/transfermarkt-datasets.duckdb"

TABLES: dict[str, tuple[list[str], str | None]] = {  # table -> (key columns, optional row filter)
    "competitions": (["competition_id"], None),
    "clubs": (["club_id"], None),
    "players": (["player_id"], None),
    "games": (["game_id"], None),
    "game_events": (["game_event_id"], None),
    "appearances": (["appearance_id"], None),
    "game_lineups": (["game_lineups_id"], None),
    "player_valuations": (["player_id", "date"], None),
    "transfers": (["player_id", "transfer_date", "from_club_id", "to_club_id"], None),
}


class TransfermarktOpenSource:
    name = "transfermarkt_open"

    def __init__(self, cache_dir: str | Path | None = None, http: CachedHttpClient | None = None):
        settings = get_settings()
        self.cache_dir = Path(cache_dir) if cache_dir else settings.raw_dir / "transfermarkt_open"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.http = http or CachedHttpClient(self.cache_dir, min_interval_s=0, timeout_s=600)
        self.db_path = self.cache_dir / "transfermarkt-datasets.duckdb"

    def download(self, force: bool = False) -> Path:
        if self.db_path.exists() and not force:
            return self.db_path
        data = self.http.get_bytes(DUCKDB_URL, force=force, cache=False)
        self.db_path.write_bytes(data)
        return self.db_path

    def read_table(self, table: str, where: str | None = None, limit: int | None = None) -> pd.DataFrame:
        if table not in TABLES:
            raise ValueError(f"unknown table {table!r}; options: {sorted(TABLES)}")
        self.download()
        con = duckdb.connect(str(self.db_path), read_only=True)
        try:
            sql = f"SELECT * FROM {table}" + (f" WHERE {where}" if where else "") + (f" LIMIT {int(limit)}" if limit else "")
            return con.execute(sql).df()
        finally:
            con.close()

    def player_match_history(self, player_name: str) -> pd.DataFrame:
        """Appearances (minutes, goals, assists, cards) for one player, newest first."""
        self.download()
        con = duckdb.connect(str(self.db_path), read_only=True)
        try:
            return con.execute(
                """
                SELECT a.date, a.competition_id, a.player_name, a.minutes_played, a.goals, a.assists,
                       a.yellow_cards, a.red_cards, g.home_club_name, g.away_club_name, g.home_club_goals, g.away_club_goals
                FROM appearances a JOIN games g USING (game_id)
                WHERE lower(a.player_name) LIKE '%' || lower(?) || '%'
                ORDER BY a.date DESC
                """,
                [player_name],
            ).df()
        finally:
            con.close()

    def substitution_profile(self, min_games: int = 20) -> pd.DataFrame:
        """Per club: average first-substitution minute and subs per game — a tempo/fatigue proxy."""
        self.download()
        con = duckdb.connect(str(self.db_path), read_only=True)
        try:
            return con.execute(
                """
                WITH subs AS (
                  SELECT game_id, club_id, club_name, min(minute) AS first_sub_minute, count(*) AS n_subs
                  FROM game_events WHERE type = 'Substitutions' GROUP BY 1,2,3)
                SELECT club_id, any_value(club_name) AS club_name, count(*) AS games,
                       avg(first_sub_minute) AS avg_first_sub_minute, avg(n_subs) AS avg_subs
                FROM subs GROUP BY 1 HAVING count(*) >= ? ORDER BY avg_first_sub_minute
                """,
                [min_games],
            ).df()
        finally:
            con.close()
