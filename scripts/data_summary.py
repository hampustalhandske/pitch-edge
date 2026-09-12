"""Reads every table actually in the warehouse right now and prints a grouped summary — row
counts, column counts, and a date range where a table has an obvious date-like column. Nothing
here is hand-maintained: it queries `information_schema.tables` directly, so it can't drift the
way a written-down number in a README can, and it will show tables nobody remembered were there.

Usage:
    uv run python scripts/data_summary.py             # print the summary
    uv run python scripts/data_summary.py --json out.json   # also write it as JSON
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
from rich import print as rprint  # noqa: E402
from rich.table import Table  # noqa: E402

from pitch_edge.config import get_settings  # noqa: E402
from pitch_edge.data.storage import Warehouse  # noqa: E402

DATE_COLUMN_CANDIDATES = ("date", "published_at", "kickoff_time", "checkpoint_date", "created_at", "started_at")


def _group(table: str) -> str:
    """Groups a table name into the same categories `features/README.md` uses, so this script's
    output can be diffed against that doc by eye."""
    prefixes = {
        "tm_": "Transfermarkt",
        "espn_": "ESPN",
        "statsbomb_": "StatsBomb",
        "wiki_": "Wikipedia attention",
        "tsdb_": "TheSportsDB",
        "backtest_": "Backtest output",
        "player_": "Player graph (research)",
        "inplay_": "In-play (research)",
    }
    for prefix, name in prefixes.items():
        if table.startswith(prefix):
            return name
    if table in {"matches", "odds"}:
        return "Match spine + odds"
    if table == "features":
        return "Feature store"
    if table in {"weather", "venues"}:
        return "Weather / venues"
    if table == "referee_tendency":
        return "Referee"
    if table in {"live_odds", "market_snapshots"}:
        return "Live / alt odds"
    if table == "news_items":
        return "News"
    if table == "pipeline_runs":
        return "Pipeline health"
    return "Other"


def _date_range(wh: Warehouse, table: str, columns: list[str]) -> str:
    col = next((c for c in DATE_COLUMN_CANDIDATES if c in columns), None)
    if col is None:
        return ""
    try:
        row = wh.query(f'SELECT min("{col}") AS lo, max("{col}") AS hi FROM "{table}"').iloc[0]
    except Exception:  # noqa: BLE001 - a malformed date column must not crash the summary
        return ""
    lo, hi = row.get("lo"), row.get("hi")
    if pd.isna(lo) or pd.isna(hi):
        return ""
    return f"{lo} to {hi}"


def summarize(db_path) -> pd.DataFrame:
    rows = []
    with Warehouse(db_path, read_only=True) as wh:
        tables = wh.query(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main' ORDER BY 1"
        )["table_name"].tolist()
        for table in tables:
            n = wh.count(table)
            columns = wh.query(f'SELECT * FROM "{table}" LIMIT 0').columns.tolist()
            rows.append(
                {
                    "group": _group(table),
                    "table": table,
                    "rows": n,
                    "columns": len(columns),
                    "date_range": _date_range(wh, table, columns),
                }
            )
    return pd.DataFrame(rows).sort_values(["group", "table"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", metavar="PATH", help="also write the summary to this JSON file")
    args = parser.parse_args()

    settings = get_settings()
    df = summarize(settings.db_path)

    for group, g in df.groupby("group", sort=False):
        t = Table(title=group)
        for c in ("table", "rows", "columns", "date_range"):
            t.add_column(c)
        for r in g.itertuples(index=False):
            t.add_row(r.table, f"{r.rows:,}", str(r.columns), r.date_range)
        rprint(t)

    rprint(f"\n[bold]{len(df)} tables, {df['rows'].sum():,} total rows[/bold] in {settings.db_path}")

    if args.json:
        Path(args.json).write_text(json.dumps(df.to_dict(orient="records"), indent=2, default=str))
        rprint(f"[dim]wrote {args.json}[/dim]")


if __name__ == "__main__":
    main()
