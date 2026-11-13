"""HTTP client for the lake: the external, read-only data service pitch-edge mirrors and studies from.

The service is not part of this repository. It is reached at `LAKEHOUSE_API_URL` with the bearer token
`LAKEHOUSE_API_TOKEN` (both read from the environment) and is expected to expose:

- `POST /v1/sql` with JSON `{"sql": str, "params": list, "format": "arrow"}`, answering with an Arrow IPC
  stream of the result rows and the lake build in the `X-Lakehouse-Version` header;
- `GET /v1/health`, answering with a JSON status object.

    from pitch_edge.lake.client import Lake

    with Lake() as lake:
        df = lake.sql("SELECT * FROM fd_matches WHERE match_date >= ?", ["2024-08-01"])
        for batch in lake.sql_batches("SELECT * FROM pm_soccer_match_map"):  # one Arrow batch at a time
            ...

Retry rule for a query. The lake restarts on demand and admits a few concurrent queries, so three answers are
transient and re-sent after a logged wait: HTTP 503 (busy; the wait is its `Retry-After`, clamped to
`[0, RETRY_MAX_WAIT_S]`), a refused connection (`httpx.ConnectError`) and a connection reset before any row
arrived (`httpx.RemoteProtocolError`, `httpx.ReadError`). Without a `Retry-After` the waits grow from
`RETRY_FIRST_WAIT_S`, doubling, capped at `RETRY_MAX_WAIT_S` each. Retrying stops when the next wait would take
the total waited for that query past `RETRY_BUDGET_S` (about 6.5 minutes) or after `RETRY_MAX_ATTEMPTS`
attempts; the `LakehouseError` raised then names the last cause and the number of attempts. Every other failure
raises `LakehouseError` at once: any 4xx, a 5xx other than 503, a read timeout, an answer without a lake build.
A response cut after rows have started to arrive is never returned as a complete result: `sql_batches` raises
`LakehouseError("stream cut after N batches ...")` (it has already yielded those batches), while `sql`, which
holds every batch until the end, re-sends the whole query within the same budget.

A `Lake` cannot be constructed while a lake freeze is recorded (`pitch_edge.control`); it raises
`LakeFrozenError` instead, so no sync, settle or study pull reads the lake during one.
"""

from __future__ import annotations

import io
import logging
import os
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import httpx
import pandas as pd
import pyarrow as pa

from pitch_edge.control import FreezeLogError, active_freeze_event

logger = logging.getLogger(__name__)

DEFAULT_URL = "http://127.0.0.1:8600"

RETRY_FIRST_WAIT_S = 5.0  # first wait after a transient failure without `Retry-After`; doubles per retry
RETRY_MAX_WAIT_S = 60.0  # upper bound on any single wait (also clamps `Retry-After`)
RETRY_BUDGET_S = 390.0  # total waiting allowed for one query before giving up
RETRY_MAX_ATTEMPTS = 20  # bound on attempts for one query, also when the waits are zero

VERSION_HEADER = "X-Lakehouse-Version"

# Exceptions raised before any row arrived that mean the lake is restarting or dropped the connection.
_RESET_ERRORS = (httpx.ConnectError, httpx.RemoteProtocolError, httpx.ReadError)


class LakehouseError(RuntimeError):
    """The lake service failed to answer a query: an HTTP error status, a transport failure, a response cut off
    mid-stream, an answer without a lake build, or transient failures that outlasted the retry budget."""


class LakeFrozenError(LakehouseError):
    """The lake must not be read now: a freeze is recorded here (`pitch-edge control freeze`, until a thaw), or the
    lake answered HTTP 423 because its own read freeze does not allow this client. Never retried."""


class _StreamCutError(LakehouseError):
    """The response stopped after some batches had been read; the result is incomplete."""


class _TransientError(Exception):
    """A failure worth re-sending the query for; `wait` is the server's requested wait, if it gave one."""

    def __init__(self, cause: str, wait: float | None = None) -> None:
        super().__init__(cause)
        self.cause = cause
        self.wait = wait


@dataclass
class _Backoff:
    """Waiting state for one query: how many attempts were made and how long was waited so far."""

    sleep: Callable[[float], None]
    attempts: int = 0
    waited: float = 0.0

    def wait_or_give_up(self, cause: str, requested: float | None) -> None:
        """Sleep before the next attempt, or raise `LakehouseError` when the budget or attempt cap is spent.
        `requested` is the server's `Retry-After` (already clamped), None for the exponential wait."""
        retries = self.attempts - 1
        wait = requested if requested is not None else min(RETRY_FIRST_WAIT_S * 2**retries, RETRY_MAX_WAIT_S)
        if self.attempts >= RETRY_MAX_ATTEMPTS or self.waited + wait > RETRY_BUDGET_S:
            raise LakehouseError(f"{cause} (gave up after {self.attempts} attempts and {self.waited:.0f}s of waiting)")
        logger.warning("lake query attempt %d failed (%s); retrying in %.0fs", self.attempts, cause, wait)
        self.sleep(wait)
        self.waited += wait


ARROW_EOS = b"\xff\xff\xff\xff\x00\x00\x00\x00"
"""The Arrow IPC end-of-stream marker. The lake ends every complete answer with it and never ends an aborted one
with it, so a body that stops without it was cut, even when it stops on a message boundary."""


class _ResponseReader(io.RawIOBase):
    """Read-only file object over a streamed HTTP body, so pyarrow can parse it as it arrives. `tail` holds the
    last bytes read so far (up to the length of `ARROW_EOS`)."""

    def __init__(self, response: httpx.Response) -> None:
        self._chunks = response.iter_bytes()
        self._pending = b""
        self.tail = b""

    def drain(self) -> None:
        """Read the rest of the body, so `tail` ends where the body ends."""
        for chunk in self._chunks:
            self.tail = (self.tail + chunk)[-len(ARROW_EOS) :]
        if self._pending:
            self.tail = (self.tail + self._pending)[-len(ARROW_EOS) :]
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
        self.tail = (self.tail + self._pending[:n])[-len(ARROW_EOS) :]
        self._pending = self._pending[n:]
        return n


class Lake:
    """Read-only SQL access to the lake service.

    Args:
        base_url: service root; defaults to `LAKEHOUSE_API_URL`, then `DEFAULT_URL`.
        token: bearer token; defaults to `LAKEHOUSE_API_TOKEN`. Never logged or stored outside the
            HTTP client's headers.
        timeout: read timeout in seconds for one request (connecting times out after 10 s).
        sleep: waiting function used between retries (tests pass one that records the waits).
        require_version: when True (the default), a query answer without the `X-Lakehouse-Version` header
            raises `LakehouseError("the lake sent no build version")`, so no result is used without its build.

    Raises:
        LakeFrozenError: a lake freeze is active now, or the freeze log cannot be read (treated as frozen).

    A `Lake` issues one query at a time and callers must not share it across threads, so one client
    holds at most one of the service's query slots. Transient failures are retried as described in the
    module docstring: HTTP 503, a refused connection and a reset connection, with logged waits totalling at
    most `RETRY_BUDGET_S` per query; anything else raises `LakehouseError` at once.

    `version` is the lake build (`X-Lakehouse-Version` header) that answered the last successful query, or
    None before the first one (or, with `require_version=False`, when the service sent none); record it
    with whatever a pull writes.
    """

    def __init__(
        self,
        base_url: str | None = None,
        token: str | None = None,
        timeout: float = 600.0,
        sleep: Callable[[float], None] = time.sleep,
        require_version: bool = True,
    ) -> None:
        _refuse_if_frozen()
        self._sleep = sleep
        self.require_version = require_version
        self.version: str | None = None
        self.base_url = (base_url or os.getenv("LAKEHOUSE_API_URL") or DEFAULT_URL).rstrip("/")
        self._http = httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {token or os.getenv('LAKEHOUSE_API_TOKEN', '')}"},
            timeout=httpx.Timeout(timeout, connect=10.0),
        )

    def sql(self, query: str, params: list[Any] | None = None) -> pd.DataFrame:
        """Run one SELECT and return every row as a DataFrame (dates as `datetime64`). A response cut
        mid-stream is discarded and the query re-sent, within the same retry budget as the other retries."""
        backoff = _Backoff(self._sleep)
        while True:
            try:
                batches = list(self._query(query, params, backoff))
                break
            except _StreamCutError as exc:
                backoff.wait_or_give_up(str(exc), None)
        if not batches:
            return pd.DataFrame()
        return pa.Table.from_batches(batches).to_pandas(date_as_object=False)

    def sql_batches(self, query: str, params: list[Any] | None = None) -> Iterator[pa.RecordBatch]:
        """Run one SELECT and yield the result as Arrow record batches while it streams in. Raises
        `LakehouseError("stream cut after N batches ...")` when the response stops after N batches were read,
        so the caller must discard what it received."""
        yield from self._query(query, params, _Backoff(self._sleep))

    def _query(self, query: str, params: list[Any] | None, backoff: _Backoff) -> Iterator[pa.RecordBatch]:
        """Send the query, re-sending after transient failures (see the module docstring), and yield its batches."""
        body = {"sql": query, "params": params or [], "format": "arrow"}
        while True:
            backoff.attempts += 1
            try:
                yield from self._attempt(body)
                return
            except _TransientError as exc:
                backoff.wait_or_give_up(exc.cause, exc.wait)

    def _attempt(self, body: dict[str, Any]) -> Iterator[pa.RecordBatch]:
        """One request. Raises `_TransientError` for a retryable failure before any batch was yielded,
        `LakehouseError` for everything else that went wrong."""
        try:
            with self._http.stream("POST", "/v1/sql", json=body) as response:
                if response.status_code == 503:
                    raise _TransientError(_error_text(response), _retry_after(response))
                if response.status_code == 423:
                    raise LakeFrozenError(f"the lake's read freeze refuses this client ({_error_text(response)})")
                _raise_for_status(response)
                version = response.headers.get(VERSION_HEADER)
                if self.require_version and not version:
                    raise LakehouseError(f"the lake sent no build version ({VERSION_HEADER} header missing)")
                self.version = version
                yield from _read_batches(response)
        except _RESET_ERRORS as exc:
            raise _TransientError(f"{type(exc).__name__}: {exc}") from exc
        except httpx.HTTPError as exc:
            raise LakehouseError(f"{type(exc).__name__}: {exc}") from exc

    def health(self) -> dict[str, Any]:
        """The service's health report (one request, not retried)."""
        response = self._http.get("/v1/health")
        _raise_for_status(response)
        return response.json()

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> Lake:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _refuse_if_frozen() -> None:
    """Raise `LakeFrozenError` while a lake freeze is active, or when the freeze log cannot be read."""
    try:
        freeze = active_freeze_event()
    except FreezeLogError as exc:
        raise LakeFrozenError(f"the lake freeze log is unreadable, so the lake is treated as frozen: {exc}") from exc
    if freeze is not None:
        raise LakeFrozenError(
            f"lake freeze since {freeze.at:%Y-%m-%d %H:%M:%S} UTC ({freeze.reason}); "
            "record `pitch-edge control thaw` to read the lake again"
        )


def _read_batches(response: httpx.Response) -> Iterator[pa.RecordBatch]:
    """Yield the Arrow batches of a successful response. A connection reset before the first batch is
    `_TransientError` (safe to re-send); a read timeout is a plain `LakehouseError` (not retried); any other
    transport or Arrow failure, and a body that ends without `ARROW_EOS`, is `_StreamCutError`."""
    n = 0
    raw = _ResponseReader(response)
    try:
        for batch in pa.ipc.open_stream(io.BufferedReader(raw)):
            yield batch
            n += 1
        raw.drain()
    except httpx.TimeoutException as exc:
        raise LakehouseError(f"{type(exc).__name__} after {n} batches: {exc}") from exc
    except (httpx.TransportError, pa.ArrowInvalid) as exc:
        cause = f"{type(exc).__name__}: {exc}"
        if n == 0 and isinstance(exc, _RESET_ERRORS):
            raise _TransientError(cause) from exc
        raise _StreamCutError(f"stream cut after {n} batches: the response was cut off ({cause})") from exc
    if raw.tail != ARROW_EOS:
        raise _StreamCutError(
            f"stream cut after {n} batches: the response was cut off (it ended without the Arrow end-of-stream marker)"
        )


def _retry_after(response: httpx.Response) -> float | None:
    """Seconds the server asked to wait before re-sending (`Retry-After`), clamped to `[0, RETRY_MAX_WAIT_S]`;
    None when the header is absent or not a number (an HTTP-date value counts as absent)."""
    try:
        seconds = float(response.headers.get("Retry-After", ""))
    except ValueError:
        return None
    return min(max(seconds, 0.0), RETRY_MAX_WAIT_S)


def _error_text(response: httpx.Response) -> str:
    """`HTTP <status>: <detail>` for an error response, the detail taken from its JSON body when present."""
    response.read()
    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        detail = response.text
    return f"HTTP {response.status_code}: {detail}"


def _raise_for_status(response: httpx.Response) -> None:
    """Raise `LakehouseError` with the service's error detail for any 4xx/5xx response."""
    if response.status_code >= 400:
        raise LakehouseError(_error_text(response))
