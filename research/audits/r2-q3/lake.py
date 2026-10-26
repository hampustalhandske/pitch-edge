"""Minimal lakehouse SQL client for the R2-Q3 audit (independent of the study code).

`q(sql, fmt)` posts one read-only query to the lake API and returns a pandas DataFrame (format parquet)
or the parsed JSON. Reads LAKEHOUSE_API_URL / LAKEHOUSE_API_TOKEN from the repo's .env; retries 503.
"""

from __future__ import annotations

import io
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]


def _env() -> dict[str, str]:
    out = {}
    for line in (REPO / ".env").read_text().splitlines():
        if line.startswith("LAKEHOUSE_") and "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def q(sql: str, params: list | None = None) -> pd.DataFrame:
    env = _env()
    url = env.get("LAKEHOUSE_API_URL", "http://127.0.0.1:8600").rstrip("/") + "/v1/sql"
    body = json.dumps({"sql": sql, "params": params or [], "format": "parquet"}).encode()
    hdr = {"Authorization": f"Bearer {env['LAKEHOUSE_API_TOKEN']}", "Content-Type": "application/json"}
    for attempt in range(15):
        req = urllib.request.Request(url, data=body, headers=hdr, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=330) as r:
                return pd.read_parquet(io.BytesIO(r.read()))
        except urllib.error.HTTPError as e:
            if e.code == 503:
                time.sleep(int(e.headers.get("Retry-After") or 0) or min(5 * 2**attempt, 60))
                continue
            raise RuntimeError(f"HTTP {e.code}: {e.read()[:400]!r}") from e
    raise RuntimeError("gave up after repeated 503")
