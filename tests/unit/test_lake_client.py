"""Mocked tests for the lake HTTP client's busy handling and the sync's one-query-at-a-time rule."""

from __future__ import annotations

import io
from datetime import date

import httpx
import pyarrow as pa
import pytest

from pitch_edge.data.storage import Warehouse
from pitch_edge.lake import client as client_mod
from pitch_edge.lake.client import Lake, LakehouseError
from pitch_edge.lake.mirror import MIRRORS, run_lake_sync


def _arrow_body() -> bytes:
    table = pa.table({"x": [1, 2]})
    sink = io.BytesIO()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue()


def _lake(handler, sleeps: list[float]) -> Lake:
    lake = Lake(base_url="http://lake.test", token="t", sleep=sleeps.append)
    lake._http = httpx.Client(base_url="http://lake.test", transport=httpx.MockTransport(handler))
    return lake


@pytest.mark.unit
def test_503_then_200_waits_retry_after_and_succeeds():
    responses = [
        httpx.Response(503, headers={"Retry-After": "7"}, json={"detail": "busy"}),
        httpx.Response(200, content=_arrow_body()),
    ]
    sleeps: list[float] = []
    lake = _lake(lambda request: responses.pop(0), sleeps)
    assert lake.sql("SELECT 1")["x"].tolist() == [1, 2]
    assert sleeps == [7.0]


@pytest.mark.unit
def test_repeated_503_raises_after_the_cap():
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(503, headers={"Retry-After": "1000"}, json={"detail": "all slots busy"})

    sleeps: list[float] = []
    lake = _lake(handler, sleeps)
    with pytest.raises(LakehouseError, match="HTTP 503: all slots busy"):
        lake.sql("SELECT 1")
    assert len(calls) == client_mod.BUSY_RETRIES + 1
    assert sleeps == [client_mod.BUSY_MAX_WAIT_S] * client_mod.BUSY_RETRIES


@pytest.mark.unit
def test_missing_retry_after_uses_fallback():
    responses = [httpx.Response(503, json={"detail": "busy"}), httpx.Response(200, content=_arrow_body())]
    sleeps: list[float] = []
    _lake(lambda request: responses.pop(0), sleeps).sql("SELECT 1")
    assert sleeps == [client_mod.BUSY_FALLBACK_S]


@pytest.mark.unit
def test_non_busy_errors_are_not_retried():
    sleeps: list[float] = []
    lake = _lake(lambda request: httpx.Response(400, json={"detail": "bad sql"}), sleeps)
    with pytest.raises(LakehouseError, match="HTTP 400"):
        lake.sql("SELECT nope")
    assert sleeps == []


class _OverlapDetectingLake:
    """Fails the test if a query starts before the previous one's batches were fully consumed."""

    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0
        self.queries = 0

    def sql_batches(self, query, params=None):
        self.active += 1
        self.queries += 1
        self.max_active = max(self.max_active, self.active)
        assert self.active == 1, "overlapping lake queries"
        try:
            yield pa.RecordBatch.from_pylist([{"coach_id": 1, "name": "x"}])
        finally:
            self.active -= 1


@pytest.mark.unit
def test_sync_never_overlaps_lake_queries(tmp_path):
    lake = _OverlapDetectingLake()
    names = ["fd_matches", "clubelo", "tm_managers"]
    with Warehouse(tmp_path / "w.duckdb") as wh:
        run_lake_sync(wh, lake, only=names, since=date(2024, 1, 1).isoformat(), sleep=lambda s: None)
    assert lake.queries >= len(names)
    assert lake.max_active == 1
    assert set(names) <= set(MIRRORS)
