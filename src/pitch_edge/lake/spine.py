"""The warehouse spine's football-data rows, built from the `lake_fd_matches` mirror.

`matches` (source `football_data_co_uk`) and the long `odds` table are what `ask`, the dashboard, the feature
store and the backtests read. They are derived from the mirror, never fetched: `build_fd_spine` turns mirror
rows into the same columns, ids and stamps the football-data CSV ingestion produced.

Conventions kept from the CSV spine:

* `match_id` is `fd_<division>_<season code>_<YYYYMMDD>_<home slug>_<away slug>` for the 22 main divisions
  and `fdx_<division>_<YYYYMMDD>_<home slug>_<away slug>` for the 16 extra-league divisions, so rows keyed on an
  id (features, weather, backtest bets) stay joinable. Main divisions start with season 2005/06; only completed
  matches (both full-time goals present) are kept.
* `odds` rows come from the mirror's `odds_all` JSON, which holds football-data's own column names (`PSH`,
  `PSCH`, `B365>2.5`, ...), through `fd_vocab.odds_wide_to_long`. Known-at semantics: an open price
  (`is_closing` False) was public before the match date; a close (`is_closing` True), the result and the match
  statistics are hindsight, targets only. `snapshot_ts` is the match date at 00:00 UTC for both, the same
  instant as the mirror's `open_known_by`, so no stamp claims knowledge earlier than the data had.

The build is incremental and resumable. The mirror is cut into calendar-year windows by `match_date`; each
window is replaced in one transaction together with its checkpoint in `lake_sync_state` (table name
`spine:football_data`), which stores a fingerprint of the mirror rows it was built from. A window whose
fingerprint is unchanged is skipped, so re-running after a crash, or after a sync that touched only recent
matches, redoes only what is missing or changed. Rebuilding a window from the same mirror rows yields identical
rows.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date

import pandas as pd

from pitch_edge.data.fd_vocab import (
    EXTRA_LEAGUE_FILES,
    LEAGUE_CODES,
    ODDS_1X2_RE,
    ODDS_OU_RE,
    season_code,
    season_label,
    slug,
)
from pitch_edge.data.storage import Warehouse

logger = logging.getLogger(__name__)

SPINE_SOURCE = "football_data_co_uk"
MIRROR_TABLE = "lake_fd_matches"
STATE_TABLE_NAME = "spine:football_data"
STATE_RUN_ID = "spine:football_data"
# Bump when the transformation changes so every window is rebuilt once.
SPINE_VERSION = "1"
FIRST_MAIN_SEASON = 2005

_RENAME = {
    "ft_home_goals": "home_goals",
    "ft_away_goals": "away_goals",
    "home_yellow": "home_yellows",
    "away_yellow": "away_yellows",
    "home_red": "home_reds",
    "away_red": "away_reds",
}
_NUMERIC = (
    "home_goals",
    "away_goals",
    "ht_home_goals",
    "ht_away_goals",
    "home_shots",
    "away_shots",
    "home_shots_on_target",
    "away_shots_on_target",
    "home_fouls",
    "away_fouls",
    "home_corners",
    "away_corners",
    "home_yellows",
    "away_yellows",
    "home_reds",
    "away_reds",
    "attendance",
)
_MIRROR_COLUMNS = (
    "division",
    "country",
    "league",
    "season",
    "match_date",
    "kickoff_time",
    "home_team",
    "away_team",
    "ft_home_goals",
    "ft_away_goals",
    "ht_home_goals",
    "ht_away_goals",
    "referee",
    "home_shots",
    "away_shots",
    "home_shots_on_target",
    "away_shots_on_target",
    "home_fouls",
    "away_fouls",
    "home_corners",
    "away_corners",
    "home_yellow",
    "away_yellow",
    "home_red",
    "away_red",
    "attendance",
    "odds_all",
)


def _start_year(season: object, day: pd.Timestamp) -> int:
    """Start year of a season label such as `2023-2024` or `2023/24`; falls back to the 1 July rule on the match date."""
    text = str(season or "")
    if len(text) >= 4 and text[:4].isdigit():
        return int(text[:4])
    return day.year if day.month >= 7 else day.year - 1


def _extra_season(season: object) -> str:
    """Extra-league season label: `2019-2020` -> `2019/20`, a calendar-year season (`2023`) unchanged."""
    text = str(season or "").strip().replace("-", "/")
    if len(text) == 9 and text[4] == "/" and text[5:].isdigit():
        return f"{text[:4]}/{text[7:]}"
    return text


def odds_columns(odds_all: object) -> dict[str, float]:
    """The 1X2 and 2.5-goal totals columns of one `odds_all` JSON string, keyed by football-data column name.
    Takes the JSON text (or None); columns outside those families and non-numeric values are dropped."""
    if not isinstance(odds_all, str) or not odds_all:
        return {}
    try:
        raw = json.loads(odds_all)
    except ValueError:
        return {}
    out: dict[str, float] = {}
    for col, value in raw.items():
        if (ODDS_1X2_RE.match(col) or ODDS_OU_RE.match(col)) and isinstance(value, int | float):
            out[col] = float(value)
    return out


def spine_frame(rows: pd.DataFrame) -> pd.DataFrame:
    """Wide match frame in the `ingest` storage shape from mirror rows.

    Takes `lake_fd_matches` rows with the columns in `_MIRROR_COLUMNS`. Returns one row per completed match of
    a known division (match columns of `MATCH_CORE_COLUMNS` plus one numeric column per football-data odds
    column), ready for `store_matches`. Rows without a date, team or both full-time goals, main-division rows
    before season 2005/06, and repeated ids are dropped."""
    df = rows.copy()
    df = df[df["division"].isin(set(LEAGUE_CODES) | set(EXTRA_LEAGUE_FILES))]
    df = df.dropna(subset=["match_date", "home_team", "away_team", "ft_home_goals", "ft_away_goals"])
    if df.empty:
        return pd.DataFrame()
    df = df.rename(columns=_RENAME)
    df["date"] = pd.to_datetime(df["match_date"])
    df["home_team"] = df["home_team"].astype(str).str.strip()
    df["away_team"] = df["away_team"].astype(str).str.strip()
    main = df["division"].isin(LEAGUE_CODES)
    years = pd.Series([_start_year(s, d) for s, d in zip(df["season"], df["date"], strict=True)], index=df.index)
    df = df[~main | (years >= FIRST_MAIN_SEASON)].copy()
    years, main = years[df.index], main[df.index]
    day = df["date"].dt.strftime("%Y%m%d")
    names = df["home_team"].map(slug) + "_" + df["away_team"].map(slug)
    df["league_code"] = df["division"]
    df["source"] = SPINE_SOURCE
    df["country"] = [
        LEAGUE_CODES[d][0] if m else EXTRA_LEAGUE_FILES.get(d) for d, m in zip(df["division"], main, strict=True)
    ]
    df["league"] = [
        f"{LEAGUE_CODES[d][0]} - {LEAGUE_CODES[d][1]}" if m else f"{EXTRA_LEAGUE_FILES[d]} - {str(lg).strip()}"
        for d, lg, m in zip(df["division"], df["league"], main, strict=True)
    ]
    df["season"] = [
        season_label(int(y)) if m else _extra_season(s) for y, s, m in zip(years, df["season"], main, strict=True)
    ]
    ids_main = "fd_" + df["division"] + "_" + years.map(lambda y: season_code(int(y))) + "_" + day + "_" + names
    ids_extra = "fdx_" + df["division"] + "_" + day + "_" + names
    df["match_id"] = ids_main.where(main, ids_extra)
    for col in _NUMERIC:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "attendance" in df and df["attendance"].isna().all():
        df = df.drop(columns="attendance")
    odds = pd.DataFrame([odds_columns(o) for o in df["odds_all"]], index=df.index)
    df = df.drop(columns=["odds_all", "match_date", "division"])
    df = pd.concat([df, odds], axis=1)
    return df.drop_duplicates(subset="match_id").reset_index(drop=True)


@dataclass(frozen=True)
class SpineWindow:
    """One calendar year of the mirror: `[start, end)` ISO dates, the mirror rows in it and their fingerprint."""

    start: str
    end: str
    rows: int
    fingerprint: str


def _ensure_state(wh: Warehouse) -> None:
    from pitch_edge.lake.mirror import _STATE_DDL  # noqa: PLC0415

    conn = wh._conn  # noqa: SLF001
    conn.execute(_STATE_DDL)
    conn.execute("ALTER TABLE lake_sync_state ADD COLUMN IF NOT EXISTS fingerprint VARCHAR")


def mirror_windows(wh: Warehouse) -> list[SpineWindow]:
    """Calendar-year windows over `lake_fd_matches`, each with its row count and content fingerprint.

    The fingerprint covers the count and an order-independent hash of the columns the spine reads plus the
    lake's `scraped_at`, prefixed with `SPINE_VERSION`; it does not change when only `synced_at` does.
    Empty when the mirror is missing or has no dated rows."""
    if not wh.table_exists(MIRROR_TABLE):
        return []
    df = wh.query(
        "SELECT year(match_date) AS y, count(*) AS n, "
        "bit_xor(hash(concat_ws('|', division, match_date, home_team, away_team, ft_home_goals, ft_away_goals, "
        "kickoff_time, referee, scraped_at, odds_all))) AS h "
        f"FROM {MIRROR_TABLE} WHERE match_date IS NOT NULL GROUP BY 1 ORDER BY 1"
    )
    return [
        SpineWindow(f"{int(r.y)}-01-01", f"{int(r.y) + 1}-01-01", int(r.n), f"{SPINE_VERSION}:{int(r.n)}:{int(r.h):x}")
        for r in df.itertuples()
    ]


def _done_fingerprints(wh: Warehouse) -> dict[tuple[str, str], str | None]:
    _ensure_state(wh)
    rows = wh._conn.execute(  # noqa: SLF001
        "SELECT window_start, window_end, fingerprint FROM lake_sync_state WHERE run_id = ?", [STATE_RUN_ID]
    ).fetchall()
    return {(str(a), str(b)): c for a, b, c in rows}


def _rebuild_window(wh: Warehouse, win: SpineWindow) -> tuple[int, int]:
    """Replace the spine rows of one window from the mirror in one transaction; returns (matches, odds rows)
    written. The window's `football_data_co_uk` matches and their odds are deleted first, so rows the lake
    dropped or re-keyed disappear. A failure rolls everything back, checkpoint included."""
    from pitch_edge.data.ingest import prepare_match_tables, store_matches  # noqa: PLC0415

    cols = ", ".join(_MIRROR_COLUMNS)
    rows = wh.query(
        f"SELECT {cols} FROM {MIRROR_TABLE} WHERE match_date >= CAST(? AS DATE) AND match_date < CAST(? AS DATE)",
        [win.start, win.end],
    )
    wide = spine_frame(rows)
    if not wide.empty:
        prepare_match_tables(wh, wide)
    conn = wh._conn  # noqa: SLF001
    conn.execute("BEGIN TRANSACTION")
    try:
        if wh.table_exists("matches"):
            where = "source = ? AND date >= CAST(? AS TIMESTAMP) AND date < CAST(? AS TIMESTAMP)"
            args = [SPINE_SOURCE, win.start, win.end]
            if wh.table_exists("odds"):
                conn.execute(f"DELETE FROM odds WHERE match_id IN (SELECT match_id FROM matches WHERE {where})", args)
            conn.execute(f"DELETE FROM matches WHERE {where}", args)
        odds_before = wh.count("odds")
        written = store_matches(wh, wide) if not wide.empty else 0
        odds_written = wh.count("odds") - odds_before
        conn.execute(
            "INSERT OR REPLACE INTO lake_sync_state (run_id, table_name, window_start, window_end, rows, "
            "completed_at, fingerprint) VALUES (?, ?, ?, ?, ?, now()::TIMESTAMP, ?)",
            [STATE_RUN_ID, STATE_TABLE_NAME, win.start, win.end, written, win.fingerprint],
        )
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    return written, odds_written


def build_fd_spine(wh: Warehouse, force: bool = False, progress: Callable[[str], None] | None = None) -> dict[str, int]:
    """Build `matches` and `odds` rows for source `football_data_co_uk` from `lake_fd_matches`.

    Takes the warehouse, `force` (rebuild every window even when its fingerprint is unchanged) and an optional
    progress callback. Returns `{"windows": total, "rebuilt": n, "skipped": n, "matches": rows written,
    "odds": rows written}`. Resumable and idempotent (see the module docstring); an empty or missing mirror
    returns zeros and touches nothing."""
    windows = mirror_windows(wh)
    done = _done_fingerprints(wh)
    report = {"windows": len(windows), "rebuilt": 0, "skipped": 0, "matches": 0, "odds": 0}
    for win in windows:
        if not force and done.get((win.start, win.end)) == win.fingerprint:
            report["skipped"] += 1
            continue
        matches, odds = _rebuild_window(wh, win)
        report["rebuilt"] += 1
        report["matches"] += matches
        report["odds"] += odds
        msg = f"spine {win.start}..{win.end}: {matches} matches, {odds} odds rows from {win.rows} mirror rows"
        logger.info(msg)
        if progress:
            progress(msg)
    return report


def spine_newest_dates(wh: Warehouse, leagues: list[str]) -> dict[str, date]:
    """Newest `matches.date` per league code for source `football_data_co_uk` (leagues with no rows are absent)."""
    if not wh.table_exists("matches") or not leagues:
        return {}
    marks = ",".join("?" * len(leagues))
    df = wh.query(
        f"SELECT league_code, CAST(max(date) AS DATE) AS d FROM matches WHERE source = ? AND league_code IN ({marks}) "
        "GROUP BY 1",
        [SPINE_SOURCE, *leagues],
    )
    return {str(r.league_code): pd.Timestamp(r.d).date() for r in df.itertuples()}
