"""How far the football-data copy in the warehouse lags the lake, for `pitch-edge health`.

Three layers hold football-data: the lake's `fd_matches` (fetched daily), the warehouse mirror
`lake_fd_matches` (refreshed by `lake-sync`) and the spine `matches` rows built from the mirror
(`pitch_edge.lake.spine`). `football_data_lag` compares the newest completed match date of each layer for the
backtest leagues and says when the gap or the age of the last sync is too large.

Thresholds: a league warns when the lake has a completed match more than `MAX_LAG_DAYS` days newer than the
mirror (a daily schedule that missed a run stays under it; a stopped schedule does not), when the mirror was
last synced more than `MAX_SYNC_AGE_HOURS` hours ago, or when the spine is behind the mirror at all.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any

import pandas as pd

from pitch_edge.data.storage import Warehouse
from pitch_edge.lake.spine import MIRROR_TABLE, STATE_RUN_ID, spine_newest_dates

MAX_LAG_DAYS = 3
MAX_SYNC_AGE_HOURS = 48


@dataclass
class FdLag:
    """Football-data lag report.

    last_sync: newest `synced_at` in the mirror (UTC, naive), None when the mirror is empty or missing.
    last_spine_build: completion time of the newest spine window, None when the spine was never built.
    table: one row per league with mirror_newest, spine_newest, lake_newest (each a date or None) and
        lake_gap_days / spine_gap_days (None when either side is unknown).
    warnings: human-readable lines, empty when everything is within the thresholds.
    """

    last_sync: datetime | None
    last_spine_build: datetime | None
    table: pd.DataFrame
    warnings: list[str] = field(default_factory=list)


def lake_newest_dates(lake: Any, leagues: Sequence[str]) -> dict[str, date]:
    """Newest completed match date per division in the lake's `fd_matches`, in one aggregate query.

    Takes a `Lake` (anything with `sql(query, params)` returning a DataFrame) and the division codes."""
    marks = ",".join("?" * len(leagues))
    df = lake.sql(
        "SELECT division, max(match_date) AS d FROM fd_matches "
        f"WHERE division IN ({marks}) AND ft_home_goals IS NOT NULL AND ft_away_goals IS NOT NULL GROUP BY 1",
        list(leagues),
    )
    return {str(r.division): pd.Timestamp(r.d).date() for r in df.itertuples() if pd.notna(r.d)}


def _mirror_newest(wh: Warehouse, leagues: Sequence[str]) -> dict[str, date]:
    if not wh.table_exists(MIRROR_TABLE):
        return {}
    marks = ",".join("?" * len(leagues))
    df = wh.query(
        f"SELECT division, max(match_date) AS d FROM {MIRROR_TABLE} WHERE division IN ({marks}) "
        "AND ft_home_goals IS NOT NULL AND ft_away_goals IS NOT NULL GROUP BY 1",
        list(leagues),
    )
    return {str(r.division): pd.Timestamp(r.d).date() for r in df.itertuples() if pd.notna(r.d)}


def _scalar(wh: Warehouse, sql: str) -> datetime | None:
    try:
        row = wh._conn.execute(sql).fetchone()  # noqa: SLF001
    except Exception:  # noqa: BLE001 - a missing table means "never"
        return None
    return pd.Timestamp(row[0]).to_pydatetime() if row and row[0] is not None else None


def _gap(newer: date | None, older: date | None) -> int | None:
    return (newer - older).days if newer is not None and older is not None else None


def football_data_lag(
    wh: Warehouse,
    lake_newest: Mapping[str, date] | None = None,
    leagues: Sequence[str] | None = None,
    now: datetime | None = None,
) -> FdLag:
    """Compare the lake, the mirror and the spine for `leagues` (default: the backtest leagues).

    `lake_newest` is `lake_newest_dates(...)` output, or None when the lake was not queried (the lake column is
    then empty and only the sync age and the spine-versus-mirror gap can warn). `now` is the clock (UTC,
    naive) used for the sync age."""
    if leagues is None:
        from pitch_edge.pipeline import DEFAULT_BACKTEST_LEAGUES  # noqa: PLC0415

        leagues = DEFAULT_BACKTEST_LEAGUES
    now = now or datetime.now(UTC).replace(tzinfo=None)
    mirror = _mirror_newest(wh, leagues)
    spine = spine_newest_dates(wh, list(leagues))
    rows: list[dict[str, Any]] = []
    for lg in leagues:
        m, s, k = mirror.get(lg), spine.get(lg), (lake_newest or {}).get(lg)
        rows.append(
            {
                "league": lg,
                "mirror_newest": m,
                "spine_newest": s,
                "lake_newest": k,
                "lake_gap_days": _gap(k, m),
                "spine_gap_days": _gap(m, s),
            }
        )
    table = pd.DataFrame(rows)
    last_sync = _scalar(wh, f"SELECT max(synced_at) FROM {MIRROR_TABLE}")
    last_spine = _scalar(wh, f"SELECT max(completed_at) FROM lake_sync_state WHERE run_id = '{STATE_RUN_ID}'")
    warnings: list[str] = []
    if last_sync is None:
        warnings.append(f"{MIRROR_TABLE} is empty or missing: run `pitch-edge lake-sync --only fd_matches`")
    elif (now - last_sync).total_seconds() > MAX_SYNC_AGE_HOURS * 3600:
        age = (now - last_sync).total_seconds() / 3600
        warnings.append(f"{MIRROR_TABLE} last synced {age:.0f} h ago (limit {MAX_SYNC_AGE_HOURS} h)")
    for r in rows:
        if r["lake_gap_days"] is not None and r["lake_gap_days"] > MAX_LAG_DAYS:
            warnings.append(
                f"{r['league']}: mirror ends {r['mirror_newest']}, lake {r['lake_newest']} "
                f"({r['lake_gap_days']} d behind, limit {MAX_LAG_DAYS} d)"
            )
        if r["spine_gap_days"] is not None and r["spine_gap_days"] > 0:
            warnings.append(
                f"{r['league']}: spine ends {r['spine_newest']}, mirror {r['mirror_newest']} "
                f"({r['spine_gap_days']} d behind; rebuild with `pitch-edge lake-sync --only fd_matches`)"
            )
        if r["mirror_newest"] is not None and r["spine_newest"] is None:
            warnings.append(f"{r['league']}: mirror has matches but the spine has none")
    return FdLag(last_sync, last_spine, table, warnings)
