"""Run one read-only SQL query against the lakehouse HTTP API.

Usage: lake_sql.py "<sql>" [--parquet OUT]   (token from LAKEHOUSE_API_TOKEN in the environment)
Prints JSON rows, or writes parquet to OUT. Retries 503 (concurrency limit) with backoff.
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request


def run(sql: str, fmt: str = "json") -> bytes:
    url = os.environ.get("LAKEHOUSE_API_URL", "http://127.0.0.1:8600").rstrip("/") + "/v1/sql"
    body = json.dumps({"sql": sql, "params": [], "format": fmt}).encode()
    headers = {"Authorization": f"Bearer {os.environ['LAKEHOUSE_API_TOKEN']}", "Content-Type": "application/json"}
    for attempt in range(12):
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=320) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 503:
                wait = int(e.headers.get("Retry-After") or 0) or min(5 * 2**attempt, 60)
                print(f"503, retrying in {wait}s", file=sys.stderr)
                time.sleep(wait)
                continue
            raise SystemExit(f"HTTP {e.code}: {e.read()[:500]!r}") from e
    raise SystemExit("gave up after repeated 503")


if __name__ == "__main__":
    sql = sys.argv[1]
    if len(sys.argv) > 3 and sys.argv[2] == "--parquet":
        data = run(sql, "parquet")
        with open(sys.argv[3], "wb") as f:
            f.write(data)
        print(f"wrote {len(data)} bytes to {sys.argv[3]}")
    else:
        out = json.loads(run(sql, "json"))
        rows = out.get("rows", out) if isinstance(out, dict) else out
        if isinstance(out, dict) and "columns" in out:
            print(out["columns"])
        for row in rows if isinstance(rows, list) else [rows]:
            print(row)
