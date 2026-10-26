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
import os
from collections.abc import Iterator
from typing import Any

import httpx
import pandas as pd
import pyarrow as pa

DEFAULT_URL = "http://127.0.0.1:8600"


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
    """

    def __init__(self, base_url: str | None = None, token: str | None = None, timeout: float = 600.0) -> None:
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
        with self._http.stream("POST", "/v1/sql", json=body) as response:
            _raise_for_status(response)
            yield from pa.ipc.open_stream(io.BufferedReader(_ResponseReader(response)))

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
