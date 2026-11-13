"""Pull the H4 OddsPapi slices into data/research/soft-open-vs-betfair-exchange-lag-v1/ (git-ignored).

Writes, through `pitch_edge.lake.client.Lake`:
  paths/paths_<lo>_<hi>.parquet  pinnacle and betfair-ex 1x2 rows stamped at or before kickoff, one file per 7-day
                                 kickoff chunk, built per registered window (2026-01-13..05-31, 2026-06-01..10-19) so
                                 no chunk straddles the boundary; columns and types of `H.PATH_SCHEMA`, also when empty
  fixtures.parquet               oddspapi_fixtures for the kickoff range (`H.FIXTURE_SCHEMA`)
  fetch_status.parquet           oddspapi_fetch_status (`H.FETCH_STATUS_SCHEMA`): what the lake's last read found
  manifest.json                  per finished chunk: rows, sha, lake build, rows and invalid prices by book and window
  snapshot.json                  written once every chunk is done and every fixture in range has been read by the lake
                                 (fetch_status ok or no_history): pull time, Lake.version, totals and invalid-price
                                 rows (price <= 1.0) by window and book, fetch_status counts, the file shas

Resumable: a chunk listed in manifest.json whose file still matches its sha is skipped. Once snapshot.json exists the
pull is final and rerunning is a no-op, so the snapshot label never changes; `--restart` moves the pull aside (never
deleted) and starts over. The pull stops if the lake sends no build (X-Lakehouse-Version) or the build changes
mid-pull, since train and test must come from one pull.

Run: arch -arm64 uv run python research/studies/soft-open-vs-betfair-exchange-lag/pull.py [--restart]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import exchange_lag as H
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import sobf


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ns(col: pa.ChunkedArray) -> np.ndarray:
    return pc.cast(pc.cast(col, pa.timestamp("ns", tz="UTC")), pa.int64()).to_numpy(zero_copy_only=False)


def chunk_stats(table: pa.Table) -> dict[str, Any]:
    """Rows and invalid-price rows (price null or <= MIN_VALID_PRICE) by window and book of one path table."""
    if table.num_rows == 0:
        return {"rows": 0, "by_window_book": {}}
    win = H.window_of(_ns(table["kickoff_utc"]))
    book = np.asarray(table["bookmaker"].to_pylist(), dtype=object)
    price = table["price"].to_numpy(zero_copy_only=False).astype(float)
    bad = ~H.valid_price(price)
    out: dict[str, dict[str, int]] = {}
    for w in sorted(set(win)):
        for b in sorted(set(book)):
            m = (win == w) & (book == b)
            if m.any():
                out[f"{w or 'outside'}|{b}"] = {"rows": int(m.sum()), "invalid_price": int((bad & m).sum())}
    return {"rows": int(table.num_rows), "by_window_book": out}


def fetch_status_check(fixtures: pa.Table, status: pa.Table) -> dict[str, Any]:
    """fetch_status of the fixtures in range: counts by status and the number not read by the lake (no fetch_status
    row, or a status outside `H.FETCH_STATUS_READ`)."""
    ids = fixtures["fixture_id"].to_pylist()
    st = dict(zip(status["fixture_id"].to_pylist(), status["status"].to_pylist(), strict=True))
    seen = [st.get(i) or "missing" for i in ids]
    counts = {k: seen.count(k) for k in sorted(set(seen))}
    unread = sum(v for k, v in counts.items() if k not in H.FETCH_STATUS_READ)
    return {"fetch_status_counts": counts, "fetch_status_unread": int(unread)}


def fetch(lake: Any, sql: str, schema: pa.Schema) -> tuple[pa.Table, str]:
    """One query as a table conformed to `schema` (an empty answer keeps the schema) and the lake build that served
    it; exits when the lake sends no build."""
    batches = list(lake.sql_batches(sql))
    if not lake.version:
        sys.exit("the lake sent no X-Lakehouse-Version: a pull must record its build")
    table = pa.Table.from_batches(batches) if batches else pa.table({})
    return H.conform(table, schema), str(lake.version)


def write(table: pa.Table, path: Path) -> str:
    """Write `table` to `path` atomically (temp file, then rename) and return its sha."""
    tmp = path.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp)
    tmp.replace(path)
    return sobf.file_sha(path)


def _save_json(path: Path, obj: dict[str, Any]) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)


def restart(out: Path) -> None:
    """Move an existing pull aside (never deleted) so the next run starts from an empty directory."""
    if out.exists() and any(out.iterdir()):
        dest = out.with_name(f"{out.name}.superseded-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}")
        out.rename(dest)
        print("moved the previous pull to", dest)


def pull(lake: Any, out: Path = H.OUT) -> dict[str, Any]:
    """Pull every chunk not yet done, then fixtures and fetch_status, and write snapshot.json. With a snapshot already
    on disk nothing is fetched and the existing snapshot is returned."""
    snap_path, man_path = out / "snapshot.json", out / "manifest.json"
    if snap_path.exists():
        print("the pull is final (snapshot.json exists); nothing fetched. Use --restart for a new pull.")
        return dict(json.loads(snap_path.read_text()))
    (out / "paths").mkdir(parents=True, exist_ok=True)
    m = json.loads(man_path.read_text()) if man_path.exists() else {"started_at": _now(), "chunks": {}}

    def check(version: str) -> None:
        first = next(iter(m["chunks"].values()), None)
        if first and first["lake_version"] != version:
            sys.exit(
                f"lake build changed mid-pull ({first['lake_version']} -> {version}); train and test must come "
                "from one pull: rerun with --restart"
            )

    for lo, hi in H.chunks():
        stem = H.chunk_stem(lo, hi)
        path = out / "paths" / f"{stem}.parquet"
        done = m["chunks"].get(stem)
        if done and path.exists() and sobf.file_sha(path) == done["sha"]:
            continue
        table, version = fetch(lake, H.paths_sql(lo, hi), H.PATH_SCHEMA)
        check(version)
        sha = write(table, path)
        m["chunks"][stem] = {"lo": lo, "hi_exclusive": hi, "sha": sha, "lake_version": version, **chunk_stats(table)}
        _save_json(man_path, m)
        print(stem, table.num_rows, version, flush=True)
    files, tables = {}, {}
    for stem, sql, schema in (
        ("fixtures", H.fixtures_sql(), H.FIXTURE_SCHEMA),
        ("fetch_status", H.fetch_status_sql(), H.FETCH_STATUS_SCHEMA),
    ):
        table, version = fetch(lake, sql, schema)
        check(version)
        tables[stem] = table
        files[stem] = {
            "rows": int(table.num_rows),
            "sha": write(table, out / f"{stem}.parquet"),
            "lake_version": version,
        }
        print(stem, table.num_rows, version, flush=True)
    fs = fetch_status_check(tables["fixtures"], tables["fetch_status"])
    if fs["fetch_status_unread"]:
        _save_json(out / "fetch_status_check.json", fs)
        sys.exit(
            f"{fs['fetch_status_unread']} fixtures in range are not read by the lake ({fs['fetch_status_counts']}); "
            "no snapshot written. Rerun with --restart once the lake has read them."
        )
    totals: dict[str, dict[str, int]] = {}
    for c in m["chunks"].values():
        for key, v in c["by_window_book"].items():
            t = totals.setdefault(key, {"rows": 0, "invalid_price": 0})
            t["rows"] += v["rows"]
            t["invalid_price"] += v["invalid_price"]
    stems = [H.chunk_stem(lo, hi) for lo, hi in H.chunks()]
    version = next(iter(m["chunks"].values()))["lake_version"]
    snap = {
        "id": H.REG_ID,
        "started_at": m["started_at"],
        "pulled_at": _now(),
        "lake_version": version,
        "consistent_lake_version": True,
        "complete": set(stems) == set(m["chunks"]),
        "kickoff_range": list(H.KICKOFF_RANGE),
        "n_chunks": len(stems),
        "paths_sha": H.combined_sha(m["chunks"][k]["sha"] for k in stems),
        "path_rows": sum(c["rows"] for c in m["chunks"].values()),
        "rows_by_window_book": dict(sorted(totals.items())),
        **fs,
        "files": files,
        "paths_sql_template": H.paths_sql("<lo>", "<hi>"),
    }
    _save_json(snap_path, snap)
    return snap


if __name__ == "__main__":
    from pitch_edge.lake.client import Lake

    ap = argparse.ArgumentParser()
    ap.add_argument("--restart", action="store_true", help="move the existing pull aside and start over")
    a = ap.parse_args()
    if a.restart:
        restart(H.OUT)
    sobf.load_env()
    with Lake(timeout=600.0) as client:
        s = pull(client, H.OUT)
    print(json.dumps({k: v for k, v in s.items() if k != "paths_sql_template"}, indent=1))
    print(H.snapshot_label(H.OUT))
