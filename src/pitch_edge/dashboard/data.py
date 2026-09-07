"""Read-only data access for the dashboard.

Every function here opens a short-lived read-only DuckDB connection (or reads a JSON/CSV
artifact written by the pipeline) and returns a plain pandas DataFrame / dict. The dashboard
never trains a model, never fetches odds, and never computes a number that isn't already sitting
in a warehouse table or a `reports/` / `data/artifacts/` file — this module is the only place that
touches those paths, so that guarantee is easy to audit.

A tiny TTL cache (replacement for Streamlit's `st.cache_data`) keeps repeated page renders from
re-opening DuckDB on every callback; the cache key always includes the resolved db path so a
different `PITCH_EDGE_DATA_DIR` never serves stale frames.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from pitch_edge.config import Settings, get_settings
from pitch_edge.data.storage import Warehouse

OUTCOMES = ("home", "draw", "away")
BIG5 = ["E0", "D1", "SP1", "I1", "F1"]

_TTL_S = 60.0
_cache: dict[tuple, tuple[float, Any]] = {}


class _SettingsProxy:
    """Re-resolves `get_settings()` on every attribute access instead of once at import time, so a
    changed `PITCH_EDGE_DATA_DIR` (a different warehouse, or a test's isolated tmp dir) is always
    picked up — this module is imported once per process, but the env it should read can change."""

    def __getattr__(self, name: str) -> Any:
        return getattr(get_settings(), name)


SETTINGS: Settings = _SettingsProxy()  # type: ignore[assignment]


def _db() -> str:
    settings = get_settings()
    settings.ensure_dirs()
    return str(settings.db_path)


def _cached(key: tuple, fn):
    now = time.monotonic()
    hit = _cache.get(key)
    if hit is not None and now - hit[0] < _TTL_S:
        return hit[1]
    value = fn()
    _cache[key] = (now, value)
    return value


def clear_cache() -> None:
    _cache.clear()


def read(table: str, where: str | None = None) -> pd.DataFrame:
    db = _db()

    def _do() -> pd.DataFrame:
        try:
            with Warehouse(db, read_only=True) as wh:
                return wh.read(table, where)
        except Exception:  # noqa: BLE001 - table may not exist yet / writer holds the lock
            return pd.DataFrame()

    return _cached(("read", db, table, where), _do)


def query(sql: str) -> pd.DataFrame:
    db = _db()

    def _do() -> pd.DataFrame:
        try:
            with Warehouse(db, read_only=True) as wh:
                return wh.query(sql)
        except Exception:  # noqa: BLE001
            return pd.DataFrame()

    return _cached(("query", db, sql), _do)


def artifact(name: str) -> dict:
    p = SETTINGS.artifacts_dir / name
    return json.loads(p.read_text()) if p.exists() else {}


def pending_signals() -> dict:
    p = SETTINGS.artifacts_dir / "pending_signals.json"
    return json.loads(p.read_text()) if p.exists() else {"thread_id": None, "proposals": []}


def clear_pending_signals(thread_id: str | None) -> None:
    p = SETTINGS.artifacts_dir / "pending_signals.json"
    p.write_text(json.dumps({"thread_id": thread_id, "proposals": []}))


def record_decisions(proposals: list[dict], approved_keys: set[str], approver: str, thread_id: str | None) -> int:
    """Write one paper_trades row per proposal — approved ones as `approved_paper`, the rest
    `rejected` — exactly the shape the LangGraph pipeline / old dashboard used. Never touches a
    bookmaker: this only ever appends rows to a local warehouse table."""
    now = datetime.now(UTC).replace(tzinfo=None).isoformat(timespec="seconds")
    rows = []
    for p in proposals:
        key = f"{p['match_id']}|{p['outcome']}"
        rows.append(
            {
                **p,
                "trade_id": f"{thread_id}:{key}",
                "status": "approved_paper" if key in approved_keys else "rejected",
                "approved_by": approver,
                "approved_at": now,
                "run_id": thread_id,
            }
        )
    wh = Warehouse(SETTINGS.db_path)
    try:
        wh.upsert("paper_trades", pd.DataFrame(rows))
    finally:
        wh.close()
    clear_cache()
    return len(approved_keys)


def latest_run(summ: pd.DataFrame) -> str | None:
    rid_path = SETTINGS.artifacts_dir / "run_id.txt"
    if rid_path.exists():
        rid = rid_path.read_text().strip()
        if not summ.empty and (summ["run_id"] == rid).any():
            return rid
    return summ.sort_values("run_id")["run_id"].iloc[-1] if not summ.empty else None


def fmt(n, digits: int = 0) -> str:
    if n is None or (isinstance(n, float) and np.isnan(n)):
        return "—"
    return f"{n:,.{digits}f}"


def bits(model_ll, market_ll):
    return (market_ll - model_ll) / np.log(2)


def overview_data() -> dict:
    universe = artifact("data_universe.json")
    summ_all = read("backtest_summaries")
    run = latest_run(summ_all)
    summ_run = summ_all[summ_all["run_id"] == run] if run else summ_all
    pending = pending_signals()
    return {"universe": universe, "summ_run": summ_run, "run": run, "pending": pending}
