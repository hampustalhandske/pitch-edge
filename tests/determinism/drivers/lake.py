"""Driver: `run_lake_sync` (fd_matches and tm_managers mirrors) and the football-data spine against a fake lake.

The fake lake serves fixed rows: Premier League and Allsvenskan matches from 2021 to 2023 in the lake's
`fd_matches` columns (odds in `odds_all`), split over several Arrow batches, and a small `tm_managers` table. Every
eighth match key is also sent with an older scrape (latest `scraped_at` must win) and every eleventh with a second
row of the same `scraped_at` that differs in its referee (the mirror's total order must pick one regardless of the
batch order). The schedule, each step recorded in `progress.json` once it finished: `run_lake_sync` of both mirrors
from 2021-01-01 (one-year windows up to a year past today), then `build_football_data_spine`.

Resume: an interrupted step is run again; the mirror resumes its unfinished sync run (completed windows skipped)
and the spine skips windows whose fingerprint checkpoint matches.

Output: every table of the warehouse. Named volatile fields: `synced_at` of each `lake_*` table, and the
`run_id`, `synced_at`, `started_at`, `finished_at` of `lake_sync_runs` (a uuid4 and wall clock),
`completed_at` of `lake_sync_state`, and `run_id`, `started_at`, `finished_at` of `pipeline_runs`. A mirror run id
in `lake_sync_state` is replaced by `run:<table name>` from `lake_sync_runs`, so a resumed run must reuse its run
(a second run of one table would show up as a duplicate label).

Defects: `random-tiebreak` makes the mirror keep a random row per key (`ORDER BY random()`, the pattern of a
row pick without a total order); `checkpoint-first` makes the killed unit write its checkpoint without its rows
before the kill, so the resumed run skips it.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import pyarrow as pa  # noqa: E402
from canon import driver_args, dump_duckdb, finish, kill_after, write_canonical  # noqa: E402

from pitch_edge.data.ingest import build_football_data_spine  # noqa: E402
from pitch_edge.data.storage import Warehouse  # noqa: E402
from pitch_edge.lake import mirror, spine  # noqa: E402

E0_TEAMS = ["Arsenal", "Chelsea", "Liverpool", "Everton", "Fulham", "Brentford"]
SWE_TEAMS = ["Malmo FF", "AIK", "Hammarby", "Djurgarden"]
SCRAPED = datetime(2024, 3, 1, 6, 0, tzinfo=UTC)
STATS = ("ht_home_goals", "ht_away_goals", "home_shots", "away_shots", "home_shots_on_target", "away_shots_on_target",
         "home_fouls", "away_fouls", "home_corners", "away_corners", "home_yellow", "away_yellow", "home_red",
         "away_red")  # fmt: skip
BATCH_ROWS = 7


def _odds(k: int) -> dict[str, float]:
    home = 1.6 + 0.15 * (k % 9)
    draw = 3.2 + 0.1 * (k % 4)
    away = round(1.0 / max(0.12, 1.04 - 1.0 / home - 1.0 / draw), 2)
    return {"B365H": round(home + 0.05, 2), "B365D": round(draw, 2), "B365A": round(away - 0.1, 2),
            "PSH": round(home, 2), "PSD": round(draw, 2), "PSA": away,
            "PSCH": round(home * 0.98, 2), "PSCD": round(draw * 1.01, 2), "PSCA": round(away * 1.02, 2)}  # fmt: skip


def fd_rows() -> list[dict[str, Any]]:
    """Every fixture of three seasons per league, one match day a week, with the tie and staleness duplicates."""
    rows: list[dict[str, Any]] = []
    k = 0
    for division, country, league, teams, first in (
        ("E0", "England", "Premier League", E0_TEAMS, date(2021, 8, 14)),
        ("SWE", "Sweden", "Allsvenskan", SWE_TEAMS, date(2021, 4, 3)),
    ):
        pairs = [(h, a) for h in teams for a in teams if h != a]
        for season_index in range(3):
            start = first.replace(year=first.year + season_index)
            season = f"{start.year}-{start.year + 1}" if division == "E0" else str(start.year)
            for j, (home, away) in enumerate(pairs):
                day = start + timedelta(days=7 * (j // 3))
                row = {
                    "division": division, "country": country, "league": league, "season": season,
                    "match_date": day, "kickoff_time": "15:00" if division == "E0" else None,
                    "home_team": home, "away_team": away, "ft_home_goals": k % 4, "ft_away_goals": (k * 7) % 3,
                    "referee": f"Ref {'ABCDE'[k % 5]}", **{s: (k + i) % 9 for i, s in enumerate(STATS)},
                    "odds_all": json.dumps(_odds(k), sort_keys=True), "ht_result": None, "ft_result": None,
                    "source_file": "fixture", "scraped_at": SCRAPED, "attendance": None,
                    "open_known_by": datetime(day.year, day.month, day.day, tzinfo=UTC),
                }  # fmt: skip
                rows.append(row)
                if k % 8 == 3:  # an older scrape of the same key with another score: must lose
                    rows.append({**row, "ft_home_goals": 9, "scraped_at": SCRAPED - timedelta(days=30)})
                if k % 11 == 5:  # a tie on scraped_at that differs in a non-key column: the total order decides
                    rows.insert(len(rows) - 1, {**row, "referee": "Ref Z"})
                k += 1
    return rows


TM_MANAGERS = [{"coach_id": i, "name": f"Coach {i}", "scraped_at": SCRAPED} for i in range(1, 6)]


class FakeLake:
    """Answers the two mirror queries from the fixed rows: fd_matches by `[start, end)` window on `match_date`,
    tm_managers whole, both in several record batches."""

    version = "fixture-build"

    def __init__(self) -> None:
        self.fd = fd_rows()
        self.fd_schema = pa.Table.from_pylist(self.fd).schema
        self.tm_schema = pa.Table.from_pylist(TM_MANAGERS).schema

    def sql_batches(self, query: str, params: list[Any] | None = None) -> Any:
        if "FROM fd_matches" in query:
            lo, hi = date.fromisoformat(params[0]), date.fromisoformat(params[1])  # type: ignore[index]
            rows, schema = [r for r in self.fd if lo <= r["match_date"] < hi], self.fd_schema
        elif "FROM tm_managers" in query:
            rows, schema = TM_MANAGERS, self.tm_schema
        else:
            raise AssertionError(f"unexpected lake query: {query[:80]}")
        for i in range(0, len(rows), BATCH_ROWS):
            yield pa.RecordBatch.from_pylist(rows[i : i + BATCH_ROWS], schema=schema)


def plant(args: Any, wh: Warehouse) -> None:
    """Install the kill and the planted defects named on the command line."""
    if args.defect == "random-tiebreak":
        mirror._dedupe_order = lambda spec, columns: " ORDER BY random()"  # noqa: SLF001
    if not args.kill_after:
        return

    def mirror_checkpoint_only(wh_: Warehouse, spec: Any, data: Any, synced_at: Any, window: Any, **kw: Any) -> int:
        wh_._conn.execute(*mirror._checkpoint_sql(kw["run_id"], spec, window, 0, kw.get("lake_version")))  # noqa: SLF001
        return 0

    def spine_checkpoint_only(wh_: Warehouse, win: Any) -> tuple[int, int]:
        wh_._conn.execute(  # noqa: SLF001
            "INSERT OR REPLACE INTO lake_sync_state (run_id, table_name, window_start, window_end, rows, "
            "completed_at, fingerprint) VALUES (?, ?, ?, ?, 0, now()::TIMESTAMP, ?)",
            [spine.STATE_RUN_ID, spine.STATE_TABLE_NAME, win.start, win.end, win.fingerprint],
        )
        return 0, 0

    first = args.defect == "checkpoint-first"
    if args.kill_at == "mirror-window":
        kill_after(mirror, "_store", args.kill_after, instead=mirror_checkpoint_only if first else None)
    elif args.kill_at == "spine-window":
        kill_after(spine, "_rebuild_window", args.kill_after, instead=spine_checkpoint_only if first else None)
    else:
        raise ValueError(args.kill_at)


def main() -> None:
    args = driver_args()
    work: Path = args.work
    db = work / "data" / "pitch_edge.duckdb"
    progress_path = work / "progress.json"
    progress: list[str] = json.loads(progress_path.read_text()) if progress_path.exists() else []
    wh = Warehouse(db)
    plant(args, wh)
    steps = [
        (
            "lake-sync",
            lambda: run_and_check(
                mirror.run_lake_sync(
                    wh, FakeLake(), only=["fd_matches", "tm_managers"], since="2021-01-01", sleep=lambda s: None
                )  # fmt: skip
            ),
        ),
        ("spine", lambda: build_football_data_spine(wh)),
    ]
    for name, step in steps:
        if name in progress:
            continue
        step()
        progress.append(name)
        progress_path.write_text(json.dumps(progress))
    runs = wh.query("SELECT run_id, table_name FROM lake_sync_runs")
    wh.close()
    run_label = {str(r.run_id): f"run:{r.table_name}" for r in runs.itertuples()}
    by_run = {"run_id": lambda v: run_label.get(v, v)}
    tables = dump_duckdb(
        db,
        volatile={
            "lake_fd_matches": ["synced_at"],
            "lake_tm_managers": ["synced_at"],
            "lake_sync_runs": ["run_id", "synced_at", "started_at", "finished_at"],
            "lake_sync_state": ["completed_at"],
            "pipeline_runs": ["run_id", "started_at", "finished_at"],
        },
        relabel={"lake_sync_state": by_run},
    )
    write_canonical(args.out, {"tables": tables})


def run_and_check(result: dict[str, int | None]) -> None:
    failed = [name for name, rows in result.items() if rows is None]
    if failed:
        raise RuntimeError(f"mirrors failed: {failed} (see pipeline_runs)")


if __name__ == "__main__":
    main()
    finish()
