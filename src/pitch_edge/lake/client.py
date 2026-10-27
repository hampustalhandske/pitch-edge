"""HTTP client for the lake: the external, read-only data service pitch-edge mirrors and studies from.

The service is not part of this repository. It is reached at `LAKEHOUSE_API_URL` with the bearer token
`LAKEHOUSE_API_TOKEN` (both read from the environment) and is expected to expose:

- `POST /v1/sql` with JSON `{"sql": str, "params": list, "format": "arrow"}`, answering with an Arrow IPC
  stream of the result rows;
- `GET /v1/health`, answering with a JSON status object.

    from pitch_edge.lake.client import Lake

    with Lake() as lake:
        df = lake.sql("SELECT * FROM fd_matches WHERE match_date >= ?", ["2024-08-01"])
        for batch in lake.sql_batches("SELECT * FROM pm_soccer_match_map"):  # one Arrow batch at a time
            ...
"""

from __future__ import annotations

import io
import logging
import os
import time
from collections.abc import Callable, Iterator
from typing import Any

import httpx
import pandas as pd
import pyarrow as pa

logger = logging.getLogger(__name__)

DEFAULT_URL = "http://127.0.0.1:8600"

# The service admits a few concurrent queries; an extra one waits ~30 s, then gets HTTP 503 with `Retry-After`.
BUSY_RETRIES = 4  # re-sends after a 503 before the error is raised
BUSY_FALLBACK_S = 5.0  # wait when a 503 carries no usable `Retry-After`
BUSY_MAX_WAIT_S = 60.0  # upper bound on any single wait


class LakehouseError(RuntimeError):
    """The lake service answered with an HTTP error status."""


class _ResponseReader(io.RawIOBase):
    """Read-only file object over a streamed HTTP body, so pyarrow can parse it as it arrives."""

    def __init__(self, response: httpx.Response) -> None:
        self._chunks = response.iter_bytes()
        self._pending = b""

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: Any) -> int:
        while not self._pending:
            try:
                self._pending = next(self._chunks)
            except StopIteration:
                return 0
        n = min(len(buffer), len(self._pending))
        buffer[:n] = self._pending[:n]
        self._pending = self._pending[n:]
        return n


class Lake:
    """Read-only SQL access to the lake service.

    Args:
        base_url: service root; defaults to `LAKEHOUSE_API_URL`, then `DEFAULT_URL`.
        token: bearer token; defaults to `LAKEHOUSE_API_TOKEN`. Never logged or stored outside the
            HTTP client's headers.
        timeout: read timeout in seconds for one request (connecting times out after 10 s).
        sleep: waiting function used between retries of a busy (HTTP 503) answer.

    A `Lake` issues one query at a time and callers must not share it across threads, so one client
    holds at most one of the service's query slots. When the service is busy (HTTP 503), `sql_batches`
    waits for `Retry-After` seconds (`BUSY_FALLBACK_S` if absent, at most `BUSY_MAX_WAIT_S`) and re-sends,
    up to `BUSY_RETRIES` times, then raises `LakehouseError`.

    `version` is the lake build (`X-Lakehouse-Version` header) that answered the last successful query, or
    None before the first one or when the service sends no version; record it with whatever a pull writes.
    """

    def __init__(
        self,
        base_url: str | None = None,
        token: str | None = None,
        timeout: float = 600.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._sleep = sleep
        self.version: str | None = None
        self.base_url = (base_url or os.getenv("LAKEHOUSE_API_URL") or DEFAULT_URL).rstrip("/")
        self._http = httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {token or os.getenv('LAKEHOUSE_API_TOKEN', '')}"},
            timeout=httpx.Timeout(timeout, connect=10.0),
        )

    def sql(self, query: str, params: list[Any] | None = None) -> pd.DataFrame:
        """Run one SELECT and return every row as a DataFrame (dates as `datetime64`)."""
        batches = list(self.sql_batches(query, params))
        if not batches:
            return pd.DataFrame()
        return pa.Table.from_batches(batches).to_pandas(date_as_object=False)

    def sql_batches(self, query: str, params: list[Any] | None = None) -> Iterator[pa.RecordBatch]:
        """Run one SELECT and yield the result as Arrow record batches while it streams in."""
        body = {"sql": query, "params": params or [], "format": "arrow"}
        for attempt in range(BUSY_RETRIES + 1):
            with self._http.stream("POST", "/v1/sql", json=body) as response:
                if response.status_code == 503 and attempt < BUSY_RETRIES:
                    delay = _retry_after(response)
                    logger.warning("lake busy (HTTP 503); retry %d/%d in %.0fs", attempt + 1, BUSY_RETRIES, delay)
                else:
                    _raise_for_status(response)
                    self.version = response.headers.get("X-Lakehouse-Version")
                    yield from pa.ipc.open_stream(io.BufferedReader(_ResponseReader(response)))
                    return
            self._sleep(delay)

    def health(self) -> dict[str, Any]:
        """The service's health report."""
        response = self._http.get("/v1/health")
        _raise_for_status(response)
        return response.json()

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> Lake:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _retry_after(response: httpx.Response) -> float:
    """Seconds to wait before re-sending after a 503: the `Retry-After` value, else `BUSY_FALLBACK_S`,
    clamped to `[0, BUSY_MAX_WAIT_S]`. An HTTP-date value is treated as absent."""
    try:
        seconds = float(response.headers.get("Retry-After", ""))
    except ValueError:
        seconds = BUSY_FALLBACK_S
    return min(max(seconds, 0.0), BUSY_MAX_WAIT_S)


def _raise_for_status(response: httpx.Response) -> None:
    """Raise `LakehouseError` with the service's error detail for any 4xx/5xx response."""
    if response.status_code < 400:
        return
    response.read()
    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        detail = response.text
    raise LakehouseError(f"HTTP {response.status_code}: {detail}")
