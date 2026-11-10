"""Point-in-time-safe mirrors of lakehouse tables inside the pitch-edge warehouse.

Each `MirrorSpec` names one lake query, the warehouse table it lands in (`lake_<name>`), the natural
key, and the *known-at column*: the mirrored column that says when a row's information became
available. Studies must filter on that column (`known_at_column <= decision time`, plus the
`known_at_lag` documented per spec) and must treat every other timing-dependent column as hindsight:
results, closing odds, resolutions and spell end dates are targets, never features.

POINT-IN-TIME WARNING: a mirrored row is not "known" as a whole. In `lake_fd_matches` every `*_close`
column and every result/statistic column is hindsight relative to `open_known_by` (open odds are known
before `match_date` 00:00 UTC; the exact capture is the Tue/Fri afternoon snapshot, so `open_known_by`
is a conservative bound). Where the lake only gives a period start, the spec adds a derived `known_at`
column (`hour_start + 1h`, `date + 1 day`) and `known_at_column` points at it. `kickoff_time` in
fd_matches is local UK time text, null for most rows before 2019 (about 64% in 2017-18, 31% in 2019,
none since 2020), so decide no later than 00:00 UTC on match day. `lake_fd_matches` starts at 2005-01-01
by default (1993-2004 are not mirrored; pass `--since 1993-01-01` to extend). `lake_tm_manager_tenures.left`
is hindsight until it has passed: use `tenures_as_of`.

Syncing is idempotent. Large tables are pulled in date windows (`window_column`, `step_days`); each window is
written in one transaction that deletes the stored rows whose window column falls in `[start, end)` and
inserts the lake's rows, so re-running a sync over the same window yields identical rows (apart from
`synced_at`) and rows removed or re-keyed upstream disappear. A failed window leaves the table unchanged. Every mirrored
row keeps the lake's own `scraped_at` and gains `synced_at` (UTC, when this pull wrote it).

The lake is read only through its HTTP API (`pitch_edge.lake.client.Lake`). A sync is strictly sequential
(mirrors one after another, windows one after another, no threads), so it holds at most one of the API's
query slots at any time.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any, Protocol

import httpx
import pyarrow as pa

from pitch_edge.data.storage import TABLE_KEYS, Warehouse
from pitch_edge.lake.client import LakehouseError

logger = logging.getLogger(__name__)

# market types whose Polymarket markets are mirrored (the lake map also holds ~370k "other" rows)
PM_MARKET_TYPES = ("win_home", "draw", "win_away", "total", "spread_home", "spread_away", "btts")
_PM_TYPES_SQL = ", ".join(f"'{t}'" for t in PM_MARKET_TYPES)
_PM_CONDITIONS = f"SELECT condition_id FROM pm_soccer_match_map WHERE market_type IN ({_PM_TYPES_SQL})"

_RETRIES = 8
_BACKOFF_S = 5.0


class LakeLike(Protocol):
    """The slice of `Lake` the mirrors use."""

    def sql_batches(self, query: str, params: list[Any] | None = None) -> Iterable[pa.RecordBatch]: ...


@dataclass(frozen=True)
class MirrorSpec:
    """One lake-to-warehouse mirror.

    sql: the lake SELECT. If `window_column` is set it must contain the literal `{window}`, which is
        replaced by a half-open predicate `window_column >= ? AND window_column < ?`.
    table: warehouse table, `lake_<name>`.
    keys: natural key columns; also registered in `TABLE_KEYS`.
    known_at_column: mirrored column giving when the row's information became public (see module doc).
    known_at_note: how to apply it, and which columns are hindsight.
    window_column / window_type / step_days: SQL expression to window on, its cast type
        (`DATE` or `TIMESTAMPTZ`), and the window length in days. None means one query for the table.
    default_start: first window start when no `--since` is given.
    """

    name: str
    sql: str
    keys: tuple[str, ...]
    known_at_column: str | None
    known_at_note: str
    window_column: str | None = None
    window_type: str = "DATE"
    step_days: int = 365
    default_start: str = "2005-01-01"

    @property
    def table(self) -> str:
        return f"lake_{self.name}"


def _spec_list() -> list[MirrorSpec]:
    drop = 'EXCLUDE (dt, "table")'
    return [
        MirrorSpec(
            "fd_matches",
            f"SELECT * {drop}, (match_date::TIMESTAMP AT TIME ZONE 'UTC') AS open_known_by "
            "FROM fd_matches WHERE {window}",
            ("division", "match_date", "home_team", "away_team"),
            "open_known_by",
            "open_known_by = match_date 00:00 UTC: *_open odds are known before it (exact capture is the "
            "Tue/Fri afternoon snapshot). HINDSIGHT in the same row: every *_close column, results, "
            "stats, referee. kickoff_time is local text, null for most rows before 2019.",
            window_column="match_date",
            step_days=365,
            default_start="2005-01-01",
        ),
        MirrorSpec(
            "pm_markets",
            'SELECT m.* EXCLUDE (dt, "table"), p.fd_match_key, p.match_id, '
            "p.market_type AS map_market_type, p.market_line AS map_market_line, p.outcome_side, "
            "p.orientation, p.match_method "
            "FROM pm_soccer_markets m JOIN pm_soccer_match_map p USING (condition_id) "
            f"WHERE p.market_type IN ({_PM_TYPES_SQL}) AND {{window}}",
            ("condition_id",),
            "opens_at",
            "tradable from opens_at; resolved_at / resolution_status / winning_outcome_label are hindsight; "
            "the market->fixture mapping was built after the fact (check match_method).",
            window_column="m.match_date",
            step_days=365,
            default_start="2023-01-01",
        ),
        MirrorSpec(
            "pm_bars_1h",
            f"SELECT * {drop}, hour_start + INTERVAL 1 HOUR AS known_at FROM pm_soccer_bars_1h WHERE condition_id IN ({_PM_CONDITIONS}) AND {{window}}",
            ("asset_id", "hour_start"),
            "known_at",
            "known_at = hour_start + 1h (the bar's close); use only bars with known_at <= decision time.",
            window_column="hour_start",
            window_type="TIMESTAMPTZ",
            step_days=30,
            default_start="2023-09-01",
        ),
        MirrorSpec(
            "kalshi_soccer_markets",
            f"SELECT * {drop} FROM kalshi_markets WHERE category = 'soccer'",
            ("ticker",),
            "open_time",
            "tradable from open_time; status / result / last_price / volume are as of the scrape (hindsight).",
        ),
        MirrorSpec(
            "kalshi_soccer_candles_1h",
            'SELECT * EXCLUDE (dt, "table") FROM kalshi_candles_1h WHERE ticker IN '
            "(SELECT ticker FROM kalshi_markets WHERE category = 'soccer') AND {window}",
            ("ticker", "end_period_ts"),
            "end_period_ts",
            "a candle is known at its period end (end_period_ts).",
            window_column="end_period_ts",
            window_type="TIMESTAMPTZ",
            step_days=30,
            default_start="2025-02-01",
        ),
        MirrorSpec(
            "us_team_matches",
            f"SELECT * {drop}, date + INTERVAL 1 DAY AS known_at FROM us_team_matches WHERE {{window}}",
            ("match_id",),
            "known_at",
            "known_at = date + 1 day (post-match goals/xG; `date` is the kickoff).",
            window_column="date",
            window_type="TIMESTAMPTZ",
            step_days=1500,
            default_start="2014-01-01",
        ),
        MirrorSpec(
            "clubelo",
            f"SELECT * {drop}, date + INTERVAL 1 DAY AS known_at FROM clubelo_ratings WHERE {{window}}",
            ("club_slug", "date"),
            "known_at",
            "`date` is always 00:00 UTC (rating valid from that day); known_at = date + 1 day is the "
            "conservative bound: use the latest row with known_at <= decision time.",
            window_column="date",
            window_type="TIMESTAMPTZ",
            step_days=1500,
            default_start="2000-01-01",
        ),
        MirrorSpec(
            "tm_managers",
            f"SELECT * {drop} FROM tm_managers",
            ("coach_id",),
            None,
            "static id -> name dimension; no timing.",
        ),
        MirrorSpec(
            "tm_manager_tenures",
            f"SELECT * {drop} FROM tm_manager_tenures",
            ("club_id", "coach_id", "appointed"),
            "appointed",
            "appointment is public on `appointed`; `left` is HINDSIGHT until it has passed - read tenures "
            "through `tenures_as_of(conn, t)`, which masks `left` > t to NULL.",
        ),
        MirrorSpec(
            "tm_injuries",
            'SELECT i.* EXCLUDE (dt, "table"), l.first_seen_at FROM tm_injuries i LEFT JOIN '
            "(SELECT player_id, injury, from_date, min(scraped_at) AS first_seen_at FROM tm_injuries_log "
            "GROUP BY ALL) l USING (player_id, injury, from_date)",
            ("player_id", "injury", "from_date"),
            "first_seen_at",
            "a spell is known from first_seen_at (first scrape that contained it; spells present at the "
            "first scrape carry that scrape time, not their true entry time); until_date, days_out and "
            "games_missed are hindsight.",
        ),
    ]


MIRRORS: dict[str, MirrorSpec] = {s.name: s for s in _spec_list()}

for _s in MIRRORS.values():
    TABLE_KEYS.setdefault(_s.table, list(_s.keys))


def windows(spec: MirrorSpec, since: str | date | None = None, until: date | None = None) -> list[tuple[str, str]]:
    """Half-open `[start, end)` ISO-date windows covering `since` (default: the spec's start) through
    one year past `until` (default: today), so fixtures already scheduled are included. A spec with no
    window column yields a single `("", "")` placeholder window."""
    if not spec.window_column:
        return [("", "")]
    start = date.fromisoformat(str(since)[:10]) if since else date.fromisoformat(spec.default_start)
    stop = (until or datetime.now(UTC).date()) + timedelta(days=366)
    out: list[tuple[str, str]] = []
    while start < stop:
        end = start + timedelta(days=spec.step_days)
        out.append((start.isoformat(), end.isoformat()))
        start = end
    return out


def bound(window_type: str) -> str:
    """SQL for one `?` window boundary as a typed value. TIMESTAMPTZ boundaries are read as UTC wall
    time whatever the session TimeZone is; a plain cast would use the session zone."""
    if window_type.upper() == "TIMESTAMPTZ":
        return "(CAST(? AS TIMESTAMP) AT TIME ZONE 'UTC')"
    return f"CAST(? AS {window_type})"


def render_query(spec: MirrorSpec, window: tuple[str, str]) -> tuple[str, list[Any]]:
    """The SQL and parameters for one window of `spec`."""
    if not spec.window_column:
        return spec.sql, []
    col, typ = spec.window_column, spec.window_type
    b = bound(typ)
    predicate = f"{col} >= {b} AND {col} < {b}"
    return spec.sql.replace("{window}", predicate), [window[0], window[1]]


def is_retryable(exc: BaseException) -> bool:
    """True for transient lake failures: transport errors and timeouts, HTTP 502/503/504, and a
    response cut off mid-stream. Every other error (bad SQL, 4xx) is final."""
    if isinstance(exc, httpx.TransportError | httpx.TimeoutException):
        return True
    if isinstance(exc, LakehouseError):
        text = str(exc)
        return text.startswith(("HTTP 502", "HTTP 503", "HTTP 504")) or "cut off" in text
    return False


def _fetch(lake: LakeLike, query: str, params: list[Any], sleep: Callable[[float], None]) -> pa.Table | None:
    """Run one lake query, retrying transient failures (see `is_retryable`) with exponential backoff."""
    for attempt in range(_RETRIES):
        try:
            batches = list(lake.sql_batches(query, params))
            return pa.Table.from_batches(batches) if batches else None
        except Exception as exc:
            if not is_retryable(exc) or attempt == _RETRIES - 1:
                raise
            delay = _BACKOFF_S * 2**attempt
            logger.warning("lake query retry %d/%d in %.0fs (%s)", attempt + 1, _RETRIES, delay, str(exc)[:120])
            sleep(delay)
    return None  # pragma: no cover


def _window_target(spec: MirrorSpec) -> str | None:
    """The window column as named in the warehouse table (table alias stripped)."""
    return spec.window_column.split(".")[-1] if spec.window_column else None


_RUNS_DDL = (
    "CREATE TABLE IF NOT EXISTS lake_sync_runs (run_id VARCHAR PRIMARY KEY, table_name VARCHAR, "
    "full_refresh BOOLEAN, since VARCHAR, synced_at TIMESTAMP, started_at TIMESTAMP, finished_at TIMESTAMP)"
)
_STATE_DDL = (
    "CREATE TABLE IF NOT EXISTS lake_sync_state (run_id VARCHAR, table_name VARCHAR, window_start VARCHAR, "
    "window_end VARCHAR, rows BIGINT, completed_at TIMESTAMP, lake_version VARCHAR, "
    "PRIMARY KEY (run_id, window_start, window_end))"
)


def _open_run(
    wh: Warehouse, spec: MirrorSpec, full_refresh: bool, since: str | None
) -> tuple[str, datetime, set[tuple[str, str]]]:
    """Find or start the sync run for `spec.table` with these parameters.

    An unfinished run with the same `full_refresh` and `since` is resumed: returns its run id, its
    `synced_at` (so resumed windows carry the same stamp) and the windows it already completed. Otherwise a
    new run is recorded and returned with no completed windows."""
    conn = wh._conn  # noqa: SLF001
    conn.execute(_RUNS_DDL)
    conn.execute(_STATE_DDL)
    conn.execute("ALTER TABLE lake_sync_state ADD COLUMN IF NOT EXISTS lake_version VARCHAR")
    row = conn.execute(
        "SELECT run_id, synced_at FROM lake_sync_runs WHERE table_name = ? AND finished_at IS NULL "
        "AND full_refresh = ? AND since IS NOT DISTINCT FROM ? ORDER BY started_at DESC LIMIT 1",
        [spec.table, full_refresh, since],
    ).fetchone()
    if row:
        done = conn.execute(
            "SELECT window_start, window_end FROM lake_sync_state WHERE run_id = ?", [row[0]]
        ).fetchall()
        logger.info("%s: resuming sync run %s (%d windows already done)", spec.table, row[0], len(done))
        return str(row[0]), row[1], {(str(a), str(b)) for a, b in done}
    run_id = uuid.uuid4().hex
    synced_at = datetime.now(UTC).replace(tzinfo=None, microsecond=0)
    conn.execute(
        "INSERT INTO lake_sync_runs VALUES (?, ?, ?, ?, ?, ?, NULL)",
        [run_id, spec.table, full_refresh, since, synced_at, synced_at],
    )
    return run_id, synced_at, set()


def _checkpoint_sql(
    run_id: str, spec: MirrorSpec, window: tuple[str, str], rows: int, lake_version: str | None
) -> tuple[str, list[Any]]:
    """The statement that marks one window of a sync run as completed, with the lake build that served it."""
    return (
        "INSERT OR REPLACE INTO lake_sync_state (run_id, table_name, window_start, window_end, rows, completed_at, "
        "lake_version) VALUES (?, ?, ?, ?, ?, now()::TIMESTAMP, ?)",
        [run_id, spec.table, window[0], window[1], rows, lake_version],
    )


def _dedupe_order(spec: MirrorSpec, columns: list[str]) -> str:
    """The ` ORDER BY ...` clause (with a leading space) that ranks a batch's rows within one key: latest
    `scraped_at` first (NULLs last) when the batch has it, then every other non-key column in `columns`
    order, ascending with NULLs last. Empty when the batch holds only key columns (the rows of a key are
    then identical)."""
    terms = ['"scraped_at" DESC NULLS LAST'] if "scraped_at" in columns else []
    quoted = ['"' + c.replace('"', '""') + '"' for c in columns if c != "scraped_at" and c not in spec.keys]
    terms += [f"{c} ASC NULLS LAST" for c in quoted]
    return f" ORDER BY {', '.join(terms)}" if terms else ""


def _store(
    wh: Warehouse,
    spec: MirrorSpec,
    data: pa.Table | None,
    synced_at: datetime,
    window: tuple[str, str] = ("", ""),
    purge_all: bool = False,
    run_id: str | None = None,
    lake_version: str | None = None,
) -> int:
    """Atomically replace one window of `spec.table` with `data`; returns rows written.

    In a single transaction: stored rows whose window column lies in `[window[0], window[1])` (every
    row if `purge_all`, or for an unwindowed mirror) are deleted, rows sharing a key with the incoming
    batch are deleted, the batch (one row per key, see below) is inserted with `synced_at`, and, when
    `run_id` is given, the window is checkpointed in `lake_sync_state` with `lake_version`, the lake build
    that served it — so a checkpoint exists exactly when the window's data was committed. Any exception
    rolls the transaction back and leaves the table unchanged. `data` of None means the lake returned no
    rows, which still clears the window.

    The row kept per key is the first under a total order: latest `scraped_at` first (NULLs last) when the
    batch has that column, then every other non-key column in the batch's column order, ascending with
    NULLs last. Two different rows under one key and one `scraped_at` therefore always resolve to the same
    row whatever order the lake sent them in; only exactly identical rows remain interchangeable."""
    conn = wh._conn  # noqa: SLF001 - the warehouse exposes no raw execute
    keys = ", ".join(f'"{k}"' for k in spec.keys)
    ts = synced_at.strftime("%Y-%m-%d %H:%M:%S")
    registered = False
    began = False
    try:
        if data is not None:
            conn.register("lake_incoming", data)
            registered = True
            conn.execute("DROP TABLE IF EXISTS lake_stage")
            conn.execute(
                f"CREATE TEMP TABLE lake_stage AS SELECT *, TIMESTAMP '{ts}' AS synced_at FROM lake_incoming "
                f"QUALIFY row_number() OVER (PARTITION BY {keys}{_dedupe_order(spec, data.column_names)}) = 1"
            )
        conn.execute("BEGIN TRANSACTION")
        began = True
        if data is not None:
            if not wh.table_exists(spec.table):
                conn.execute(f"CREATE TABLE {spec.table} AS SELECT * FROM lake_stage LIMIT 0")
            else:
                existing = set(wh.columns(spec.table))
                for name, typ, *_ in conn.execute("DESCRIBE lake_stage").fetchall():
                    if name not in existing:
                        conn.execute(f'ALTER TABLE {spec.table} ADD COLUMN "{name}" {typ}')
        if wh.table_exists(spec.table):
            target = _window_target(spec)
            if purge_all or not target:
                conn.execute(f"DELETE FROM {spec.table}")
            else:
                b = bound(spec.window_type)
                conn.execute(
                    f'DELETE FROM {spec.table} WHERE "{target}" >= {b} AND "{target}" < {b}',
                    [window[0], window[1]],
                )
        written = 0
        if data is not None:
            on = " AND ".join(f't."{k}" IS NOT DISTINCT FROM s."{k}"' for k in spec.keys)
            conn.execute(f"DELETE FROM {spec.table} t WHERE EXISTS (SELECT 1 FROM lake_stage s WHERE {on})")
            conn.execute(f"INSERT INTO {spec.table} BY NAME SELECT * FROM lake_stage")
            row = conn.execute("SELECT count(*) FROM lake_stage").fetchone()
            written = int(row[0]) if row else 0
        if run_id is not None:
            conn.execute(*_checkpoint_sql(run_id, spec, window, written, lake_version))
        conn.execute("COMMIT")
        began = False
        return written
    except BaseException:
        if began:
            conn.execute("ROLLBACK")
        raise
    finally:
        if registered:
            conn.unregister("lake_incoming")
        conn.execute("DROP TABLE IF EXISTS lake_stage")


def assert_unique(wh: Warehouse, spec: MirrorSpec) -> None:
    """Raise RuntimeError if `spec.table` holds more than one row for any key (NULL components compare equal)."""
    if not wh.table_exists(spec.table):
        return
    keys = ", ".join(f'"{k}"' for k in spec.keys)
    row = wh._conn.execute(  # noqa: SLF001
        f"SELECT count(*) FROM (SELECT {keys} FROM {spec.table} GROUP BY {keys} HAVING count(*) > 1)"
    ).fetchone()
    if row and row[0]:
        raise RuntimeError(f"{spec.table}: {row[0]} duplicated key(s) on {list(spec.keys)}")


def tenures_as_of(conn: Any, t: datetime | str) -> Any:
    """Manager tenures as known at `t` (a DuckDB connection over the warehouse, and a cutoff).

    Returns a DataFrame of `lake_tm_manager_tenures` rows with `appointed <= t`, with `left` set to NULL
    when `left > t` (the tenure end was not yet known). Takes a duckdb connection and a datetime or ISO string."""
    return conn.execute(
        'SELECT club_id, coach_id, manager_name, appointed, CASE WHEN "left" > CAST(? AS TIMESTAMPTZ) '
        'THEN NULL ELSE "left" END AS "left" FROM lake_tm_manager_tenures '
        "WHERE appointed <= CAST(? AS TIMESTAMPTZ)",
        [str(t), str(t)],
    ).df()


def sync_mirror(
    wh: Warehouse,
    lake: LakeLike,
    spec: MirrorSpec,
    since: str | date | None = None,
    until: date | None = None,
    sleep: Callable[[float], None] = time.sleep,
    full_refresh: bool = False,
) -> int:
    """Pull `spec` window by window from the lake into `spec.table`; returns rows written.

    Takes the warehouse, a lake client, the spec, an optional window start (`since`; ignored with a warning
    for an unwindowed mirror) and end anchor (`until`, default today). Each window is replaced atomically
    (see `_store`). `full_refresh` empties the table in the first window's transaction and starts from the
    spec's default start, so rows outside the pulled range are dropped too.

    Resumable: every committed window is checkpointed with its run. If a run is interrupted (crash, kill,
    lid closed), calling again with the same `full_refresh`/`since` resumes it — completed windows are
    skipped, a full refresh never re-empties a table it has already started refilling — and the run is
    marked finished only after the last window. Returns rows written in this call. Ends by asserting key
    uniqueness."""
    if since and not spec.window_column:
        logger.warning("%s has no date window; --since %s ignored, pulling the whole table", spec.table, since)
    since_key = None if full_refresh or not since else str(since)[:10]
    run_id, synced_at, done = _open_run(wh, spec, full_refresh, since_key)
    total = 0
    for i, win in enumerate(windows(spec, since_key, until)):
        if win in done:
            continue
        query, params = render_query(spec, win)
        data = _fetch(lake, query, params, sleep)
        version = getattr(lake, "version", None)
        if data is not None and data.num_rows == 0:
            data = None
        if data is None and not wh.table_exists(spec.table):
            wh._conn.execute(*_checkpoint_sql(run_id, spec, win, 0, version))  # noqa: SLF001
            continue
        purge = full_refresh and i == 0
        total += _store(wh, spec, data, synced_at, win, purge_all=purge, run_id=run_id, lake_version=version)
        if data is not None:
            logger.info("%s window %s..%s: %d rows (total %d)", spec.table, win[0], win[1], data.num_rows, total)
    assert_unique(wh, spec)
    wh._conn.execute(  # noqa: SLF001
        "UPDATE lake_sync_runs SET finished_at = now()::TIMESTAMP WHERE run_id = ?", [run_id]
    )
    return total


def run_lake_sync(
    wh: Warehouse,
    lake: LakeLike,
    only: list[str] | None = None,
    since: str | None = None,
    sleep: Callable[[float], None] = time.sleep,
    full_refresh: bool = False,
) -> dict[str, int | None]:
    """Sync the named mirrors (default all), one after another, each logged to `pipeline_runs`.

    Returns rows written per mirror name; a mirror that raised maps to None (the error is in the health
    log, and the remaining mirrors still run). Unknown names raise `KeyError` before anything is pulled."""
    from pitch_edge.data.ingest import _run

    names = list(only) if only else list(MIRRORS)
    unknown = [n for n in names if n not in MIRRORS]
    if unknown:
        raise KeyError(f"unknown mirror(s) {unknown}; available: {sorted(MIRRORS)}")
    out: dict[str, int | None] = {}
    for name in names:
        spec = MIRRORS[name]
        failed = False

        def job(spec: MirrorSpec = spec) -> int:
            nonlocal failed
            try:
                return sync_mirror(wh, lake, spec, since=since, sleep=sleep, full_refresh=full_refresh)
            except Exception:
                failed = True
                raise

        rows = _run(wh, f"lake:{name}", job)
        out[name] = None if failed else rows
    return out
