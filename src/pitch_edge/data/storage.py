"""Local-first warehouse: DuckDB for tables, Parquet for the lake export.

One `Warehouse` object owns every table the platform writes. Tables are
created lazily from the first DataFrame written and widened automatically
when a later write carries new columns (sources add odds columns over the
years), so schema evolution never silently drops data. All writes are
idempotent upserts keyed on natural keys — re-running ingestion is safe.

Phase 3 mirrors these tables to BigQuery/GCS via `pitch_edge.cloud`; the
Parquet export here is the hand-off format.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pandas as pd

from pitch_edge.config import get_settings

logger = logging.getLogger(__name__)

TABLE_KEYS: dict[str, list[str]] = {
    "matches": ["match_id"],
    "odds": ["match_id", "bookmaker", "market", "side", "is_closing", "snapshot_ts"],
    "team_ratings": ["team", "date", "source"],
    "venues": ["team"],
    "weather": ["match_id"],
    "features": ["match_id"],
    "news_items": ["item_id"],
    "market_snapshots": ["market_id", "outcome", "snapshot_ts"],
    "pipeline_runs": ["run_id"],
    "model_predictions": ["match_id", "model_name", "run_id"],
    "backtest_bets": ["backtest_id", "match_id", "outcome"],
    "backtest_summaries": ["backtest_id"],
    "statsbomb_matches": ["match_id"],
    "player_embeddings": ["player_id", "model_version"],
    "tm_games": ["game_id"],
    "tm_game_events": ["game_event_id"],
    "tm_appearances": ["appearance_id"],
    "tm_players": ["player_id"],
    "tm_player_valuations": ["player_id", "date"],
    "tm_clubs": ["club_id"],
    "tm_game_lineups": ["game_lineups_id"],
    "tsdb_events": ["event_id"],
    "tsdb_players": ["player_id"],
    "tsdb_teams": ["team_id"],
    # Phase 5
    "wiki_articles": ["team", "lang"],
    "wiki_pageviews": ["team", "date"],  # two spine names can share one article (renames)
    "referee_announcements": ["fixture_key", "referee", "source"],
    "referee_announcement_moves": ["fixture_key", "referee", "venue", "outcome"],
    "api_football_injuries": ["league_code", "season", "fixture_id", "team", "player"],
    "api_football_lineups": ["fixture_id", "team", "player", "slot"],
    "fixture_predictions": ["fixture_key", "model_name", "version"],
    "dossiers": ["fixture_key", "version"],
    "live_odds": ["match_id", "bookmaker", "market", "side", "snapshot_ts"],
    "dim_soccer_markets": ["condition_id"],
    "pmxt_orderbook": ["condition_id", "asset_id", "timestamp_received", "side", "price", "size"],
    "pmxt_days_scanned": ["day"],
    "pmxt_match_map": ["condition_id"],
}


def _duck_type(dtype) -> str:
    kind = getattr(dtype, "kind", "O")
    if kind == "b":
        return "BOOLEAN"
    if kind in ("i", "u"):
        return "BIGINT"
    if kind == "f":
        return "DOUBLE"
    if kind == "M":
        return "TIMESTAMP"
    return "VARCHAR"


class Warehouse:
    def __init__(self, db_path: str | Path | None = None, read_only: bool = False):
        self.db_path = Path(db_path) if db_path else get_settings().db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = duckdb.connect(str(self.db_path), read_only=read_only)

    # ----------------------------------------------------------------- schema
    def table_exists(self, table: str) -> bool:
        row = self._conn.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_name = ?", [table]
        ).fetchone()
        return bool(row and row[0])

    def columns(self, table: str) -> list[str]:
        if not self.table_exists(table):
            return []
        return [r[0] for r in self._conn.execute(f"DESCRIBE {table}").fetchall()]

    def _ensure_table(self, table: str, df: pd.DataFrame) -> None:
        if not self.table_exists(table):
            cols = ", ".join(f'"{c}" {_duck_type(df[c].dtype)}' for c in df.columns)
            self._conn.execute(f"CREATE TABLE {table} ({cols})")
            return
        existing = set(self.columns(table))
        for c in df.columns:
            if c not in existing:
                self._conn.execute(f'ALTER TABLE {table} ADD COLUMN "{c}" {_duck_type(df[c].dtype)}')

    # ----------------------------------------------------------------- writes
    def upsert(self, table: str, df: pd.DataFrame, keys: list[str] | None = None) -> int:
        """Insert rows whose natural key is not yet present. Returns rows inserted."""
        if df is None or df.empty:
            return 0
        keys = keys or TABLE_KEYS.get(table)
        if not keys:
            raise ValueError(f"No natural key known for table {table!r}; pass keys=")
        df = _clean_frame(df)
        missing_keys = [k for k in keys if k not in df.columns]
        if missing_keys:
            raise ValueError(f"upsert({table}): frame is missing key columns {missing_keys}")
        df = df.drop_duplicates(subset=keys)
        self._ensure_table(table, df)
        self._conn.register("incoming_df", df)
        before = self.count(table)
        key_expr = " AND ".join(f'(t."{k}" = i."{k}" OR (t."{k}" IS NULL AND i."{k}" IS NULL))' for k in keys)
        self._conn.execute(
            f"""
            INSERT INTO {table} BY NAME
            SELECT i.* FROM incoming_df i
            WHERE NOT EXISTS (SELECT 1 FROM {table} t WHERE {key_expr})
            """
        )
        self._conn.unregister("incoming_df")
        return self.count(table) - before

    def replace(self, table: str, df: pd.DataFrame) -> int:
        """Drop-and-recreate a table (features, summaries) from a frame."""
        df = _clean_frame(df)
        self._conn.execute(f"DROP TABLE IF EXISTS {table}")
        if df.empty:
            return 0
        self._conn.register("incoming_df", df)
        self._conn.execute(f"CREATE TABLE {table} AS SELECT * FROM incoming_df")
        self._conn.unregister("incoming_df")
        return len(df)

    # ------------------------------------------------------------------ reads
    def read(self, table: str, where: str | None = None, params: list | None = None) -> pd.DataFrame:
        if not self.table_exists(table):
            return pd.DataFrame()
        sql = f"SELECT * FROM {table}" + (f" WHERE {where}" if where else "")
        return self._conn.execute(sql, params or []).df()

    def query(self, sql: str, params: list | None = None) -> pd.DataFrame:
        return self._conn.execute(sql, params or []).df()

    def count(self, table: str) -> int:
        if not self.table_exists(table):
            return 0
        row = self._conn.execute(f"SELECT count(*) FROM {table}").fetchone()
        return int(row[0]) if row else 0

    def matches_with_closing_odds(self, bookmakers: tuple[str, ...] = ("PS", "B365", "Avg", "Max")) -> pd.DataFrame:
        """Matches joined with wide early/closing 1X2 odds per bookmaker prefix."""
        matches = self.read("matches")
        if matches.empty or not self.table_exists("odds"):
            return matches
        odds = self.read("odds", "market = '1x2'")
        if odds.empty:
            return matches
        odds = odds[odds["bookmaker"].isin(bookmakers)]
        side_code = odds["side"].map({"home": "H", "draw": "D", "away": "A"})
        odds["col"] = odds["bookmaker"] + odds["is_closing"].map({True: "C", False: ""}) + side_code
        wide = odds.pivot_table(index="match_id", columns="col", values="price", aggfunc="first")
        return matches.merge(wide, left_on="match_id", right_index=True, how="left")

    # ----------------------------------------------------------- ops logging
    def log_run(
        self, source: str, status: str, rows: int = 0, error: str | None = None, started_at: datetime | None = None
    ) -> str:
        run_id = uuid.uuid4().hex
        now = datetime.now(UTC).replace(tzinfo=None)
        df = pd.DataFrame(
            [
                {
                    "run_id": run_id,
                    "source": source,
                    "started_at": started_at or now,
                    "finished_at": now,
                    "status": status,
                    "rows": int(rows),
                    "error": (error or "")[:2000],
                }
            ]
        )
        self.upsert("pipeline_runs", df)
        return run_id

    def health(self) -> pd.DataFrame:
        runs = self.read("pipeline_runs")
        if runs.empty:
            return pd.DataFrame(columns=["source", "last_success", "last_status", "runs", "errors", "rows_last"])
        runs = runs.sort_values("finished_at")
        grouped = runs.groupby("source")
        out = pd.DataFrame(
            {
                "last_success": grouped.apply(
                    lambda g: g.loc[g["status"] == "ok", "finished_at"].max() if (g["status"] == "ok").any() else pd.NaT
                ),
                "last_status": grouped["status"].last(),
                "runs": grouped.size(),
                "errors": grouped["status"].apply(lambda s: int((s != "ok").sum())),
                "rows_last": grouped["rows"].last(),
            }
        ).reset_index()
        return out

    # ---------------------------------------------------------------- export
    def export_parquet(self, out_dir: str | Path | None = None) -> list[Path]:
        out_dir = Path(out_dir) if out_dir else get_settings().data_dir / "lake"
        out_dir.mkdir(parents=True, exist_ok=True)
        written = []
        for (name,) in self._conn.execute("SELECT table_name FROM information_schema.tables").fetchall():
            path = out_dir / f"{name}.parquet"
            self._conn.execute(f"COPY {name} TO '{path}' (FORMAT PARQUET)")
            written.append(path)
        return written

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Warehouse:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c) for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object:
            # DuckDB needs homogeneous columns; object columns with mixed types -> str
            df[c] = df[c].where(df[c].isna(), df[c].astype(str))
        elif str(df[c].dtype).startswith("datetime64[ns,"):
            df[c] = df[c].dt.tz_convert(None)
    return df.reset_index(drop=True)


# Backwards-compatible alias used by early tests/docs.
MatchStore = Warehouse
