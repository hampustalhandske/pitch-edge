"""Minimal HTTP client for the lakehouse SQL API, used by the R1 study scripts.

`sql(query, params, fmt)` posts one read-only SELECT to `POST /v1/sql` and returns a pandas DataFrame
(fetched as parquet). Credentials come from `LAKEHOUSE_API_URL` / `LAKEHOUSE_API_TOKEN` in the repo's
`.env` (or the environment). A 503 (the lake's two-concurrent-query limit) is retried with backoff.
"""

from __future__ import annotations

import io
import os
import sys
import time
from pathlib import Path

import httpx
import pandas as pd

REPO = Path(__file__).resolve().parents[3]


def _env() -> tuple[str, str]:
    url = os.environ.get("LAKEHOUSE_API_URL")
    tok = os.environ.get("LAKEHOUSE_API_TOKEN")
    if not (url and tok):
        for line in (REPO / ".env").read_text().splitlines():
            if line.startswith("LAKEHOUSE_API_URL="):
                url = url or line.split("=", 1)[1].strip().strip('"')
            if line.startswith("LAKEHOUSE_API_TOKEN="):
                tok = tok or line.split("=", 1)[1].strip().strip('"')
    return (url or "http://127.0.0.1:8600").replace("localhost", "127.0.0.1"), tok or ""


def sql(query: str, params: list | None = None, retries: int = 12) -> pd.DataFrame:
    """Run one SELECT on the lake and return the result as a DataFrame."""
    url, tok = _env()
    body = {"sql": query, "params": params or [], "format": "parquet"}
    delay = 5.0
    for _attempt in range(retries):
        try:
            r = httpx.post(f"{url}/v1/sql", json=body, headers={"Authorization": f"Bearer {tok}"}, timeout=360)
        except httpx.TransportError as exc:
            print(f"lake transport error ({exc}), retry in {delay:.0f}s", file=sys.stderr)
            time.sleep(delay)
            delay = min(delay * 1.5, 60)
            continue
        if r.status_code == 503:
            wait = float(r.headers.get("Retry-After", delay))
            print(f"lake busy (503), retry in {wait:.0f}s", file=sys.stderr)
            time.sleep(max(wait, delay))
            delay = min(delay * 1.5, 60)
            continue
        if r.status_code != 200:
            raise RuntimeError(f"lake error {r.status_code}: {r.text[:500]}")
        return pd.read_parquet(io.BytesIO(r.content))
    raise RuntimeError("lake busy: retries exhausted")


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_rows", 200)
    print(sql(sys.argv[1]))
