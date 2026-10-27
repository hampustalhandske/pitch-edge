"""Read-only lakehouse access for the R3 `closing-line-prediction` studies.

`q(sql, params)` runs one SELECT through `pitch_edge.lake.client.Lake` and returns a pandas DataFrame,
retrying when the API answers 503 (the lake allows two concurrent queries shared with other users).
Credentials come from LAKEHOUSE_* in the repo's .env, loaded here into the process environment without
printing them.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import pandas as pd

from pitch_edge.lake.client import Lake, LakehouseError

REPO = Path(__file__).resolve().parents[3]


def _load_env() -> None:
    for line in (REPO / ".env").read_text().splitlines():
        if line.startswith("LAKEHOUSE_") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _with_retry(fn, *args, attempts: int = 20):  # type: ignore[no-untyped-def]
    _load_env()
    for i in range(attempts):
        try:
            with Lake(timeout=600.0) as lake:
                return fn(lake, *args)
        except LakehouseError as exc:
            msg = str(exc)
            if "503" in msg or "busy" in msg.lower() or "concurren" in msg.lower():
                time.sleep(min(5 * 2**i, 60))
                continue
            raise
    raise RuntimeError("lake: gave up after repeated 503")


def q(sql: str, params: list[Any] | None = None) -> pd.DataFrame:
    """One read-only query -> DataFrame (all rows)."""
    return _with_retry(lambda lake, s, p: lake.sql(s, p), sql, params)
