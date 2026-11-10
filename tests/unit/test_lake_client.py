"""Mocked tests for the lake HTTP client (retries of transient failures within a waiting budget, no retry of
anything else, cut streams, the required build version) and the sync (one query at a time, lake build recorded per
window)."""

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


VERSION = {"X-Lakehouse-Version": "856ab2f64e97"}


def _ok(content: bytes | None = None) -> httpx.Response:
    """A successful answer carrying a lake build, as the service sends it."""
    return httpx.Response(200, headers=VERSION, content=_arrow_body() if content is None else content)


def _lake(handler, sleeps: list[float], **kw) -> Lake:
    lake = Lake(base_url="http://lake.test", token="t", sleep=sleeps.append, **kw)
    lake._http = httpx.Client(base_url="http://lake.test", transport=httpx.MockTransport(handler))
    return lake


@pytest.mark.unit
def test_503_then_200_waits_retry_after_and_succeeds():
    responses = [
        httpx.Response(503, headers={"Retry-After": "7"}, json={"detail": "busy"}),
        _ok(),
    ]
    sleeps: list[float] = []
    lake = _lake(lambda request: responses.pop(0), sleeps)
    assert lake.sql("SELECT 1")["x"].tolist() == [1, 2]
    assert sleeps == [7.0]


@pytest.mark.unit
def test_version_is_the_build_that_answered_the_last_query():
    responses = [
        httpx.Response(200, headers={"X-Lakehouse-Version": "856ab2f64e97"}, content=_arrow_body()),
        httpx.Response(200, content=_arrow_body()),
    ]
    lake = _lake(lambda request: responses.pop(0), [], require_version=False)
    assert lake.version is None
    lake.sql("SELECT 1")
    assert lake.version == "856ab2f64e97"
    lake.sql("SELECT 1")
    assert lake.version is None


@pytest.mark.unit
def test_repeated_503_raises_once_the_waiting_budget_is_spent():
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(503, headers={"Retry-After": "1000"}, json={"detail": "all slots busy"})

    sleeps: list[float] = []
    lake = _lake(handler, sleeps)
    with pytest.raises(LakehouseError, match=r"HTTP 503: all slots busy \(gave up after 7 attempts"):
        lake.sql("SELECT 1")
    # Retry-After 1000 is clamped to 60 s; a seventh wait would take the total past the 390 s budget
    assert sleeps == [client_mod.RETRY_MAX_WAIT_S] * 6
    assert len(calls) == 7


@pytest.mark.unit
def test_missing_retry_after_uses_fallback():
    responses = [httpx.Response(503, json={"detail": "busy"}), _ok()]
    sleeps: list[float] = []
    _lake(lambda request: responses.pop(0), sleeps).sql("SELECT 1")
    assert sleeps == [client_mod.RETRY_FIRST_WAIT_S]


@pytest.mark.unit
def test_non_busy_errors_are_not_retried():
    sleeps: list[float] = []
    lake = _lake(lambda request: httpx.Response(400, json={"detail": "bad sql"}), sleeps)
    with pytest.raises(LakehouseError, match="HTTP 400"):
        lake.sql("SELECT nope")
    assert sleeps == []


def _batched_body(n_batches: int = 4) -> tuple[bytes, list[int]]:
    """An Arrow stream of `n_batches` two-row batches, and the byte offset at which each batch ends."""
    table = pa.table({"x": list(range(2 * n_batches))})
    sink = io.BytesIO()
    ends = []
    with pa.ipc.new_stream(sink, table.schema) as writer:
        for batch in table.to_batches(max_chunksize=2):
            writer.write_batch(batch)
            ends.append(sink.tell())
    return sink.getvalue(), ends


class _CutStream(httpx.SyncByteStream):
    """A response body that sends `body[:cut]` and then fails as a reset connection does."""

    def __init__(self, body: bytes, cut: int) -> None:
        self.body, self.cut = body, cut

    def __iter__(self):
        yield self.body[: self.cut]
        raise httpx.ReadError("Connection reset by peer")


def _refused(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("[Errno 61] Connection refused", request=request)


def _scripted(*steps):
    """Handler that plays `steps` in order: an exception class is raised, a response is returned."""
    queue = list(steps)
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        step = queue.pop(0)
        if isinstance(step, type) and issubclass(step, Exception):
            raise step("transient", request=request)
        return step

    return handler, calls


@pytest.mark.unit
@pytest.mark.parametrize(
    "first",
    [
        httpx.Response(503, json={"detail": "lake restarting"}),
        httpx.ConnectError,  # lake process down: connection refused
        httpx.RemoteProtocolError,  # server disconnected without sending a response
        httpx.ReadError,  # connection reset before the answer
    ],
    ids=["503", "connect-refused", "disconnected", "reset"],
)
def test_each_transient_cause_is_retried_until_the_lake_answers(first, caplog):
    handler, calls = _scripted(first, first, _ok())
    sleeps: list[float] = []
    with caplog.at_level("WARNING", logger="pitch_edge.lake.client"):
        df = _lake(handler, sleeps).sql("SELECT 1")
    assert df["x"].tolist() == [1, 2]
    assert len(calls) == 3
    assert sleeps == [5.0, 10.0]  # exponential from RETRY_FIRST_WAIT_S
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 2
    assert "attempt 1 failed" in warnings[0] and "retrying in 5s" in warnings[0]
    assert "attempt 2 failed" in warnings[1] and "retrying in 10s" in warnings[1]


@pytest.mark.unit
def test_a_reset_before_the_first_batch_is_retried_through_sql_batches():
    body, ends = _batched_body()
    handler, calls = _scripted(httpx.Response(200, headers=VERSION, stream=_CutStream(body, 10)), _ok(body))
    sleeps: list[float] = []
    batches = list(_lake(handler, sleeps).sql_batches("SELECT 1"))
    assert sum(b.num_rows for b in batches) == 8
    assert len(calls) == 2 and sleeps == [5.0]


@pytest.mark.unit
def test_a_refused_lake_is_given_up_once_the_budget_is_spent(caplog):
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return _refused(request)

    sleeps: list[float] = []
    with caplog.at_level("WARNING", logger="pitch_edge.lake.client"):
        with pytest.raises(LakehouseError, match=r"ConnectError: .*Connection refused \(gave up after 10 attempts"):
            _lake(handler, sleeps).sql("SELECT 1")
    assert sleeps == [5.0, 10.0, 20.0, 40.0, 60.0, 60.0, 60.0, 60.0, 60.0]
    assert sum(sleeps) <= client_mod.RETRY_BUDGET_S < sum(sleeps) + client_mod.RETRY_MAX_WAIT_S
    assert len(calls) == 10
    assert len([r for r in caplog.records if r.levelname == "WARNING"]) == 9


@pytest.mark.unit
def test_zero_retry_after_waits_are_still_bounded_by_the_attempt_cap():
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(503, headers={"Retry-After": "0"}, json={"detail": "busy"})

    with pytest.raises(LakehouseError, match="gave up after 20 attempts"):
        _lake(handler, []).sql("SELECT 1")
    assert len(calls) == client_mod.RETRY_MAX_ATTEMPTS


@pytest.mark.unit
@pytest.mark.parametrize(
    "answer,match",
    [
        (httpx.Response(400, json={"detail": "bad sql"}), "HTTP 400: bad sql"),
        (httpx.Response(500, json={"detail": "internal error"}), "HTTP 500: internal error"),
        (httpx.Response(502, json={"detail": "bad gateway"}), "HTTP 502: bad gateway"),
        (httpx.ReadTimeout, "ReadTimeout"),
    ],
    ids=["400", "500", "502", "read-timeout"],
)
def test_non_transient_failures_are_raised_at_once(answer, match):
    handler, calls = _scripted(answer, _ok())
    sleeps: list[float] = []
    with pytest.raises(LakehouseError, match=match):
        _lake(handler, sleeps).sql("SELECT 1")
    assert len(calls) == 1 and sleeps == []


@pytest.mark.unit
def test_a_stream_cut_after_rows_arrived_raises_from_sql_batches():
    body, ends = _batched_body()
    handler, calls = _scripted(httpx.Response(200, headers=VERSION, stream=_CutStream(body, ends[1])), _ok(body))
    sleeps: list[float] = []
    received = []
    with pytest.raises(LakehouseError, match="stream cut after 2 batches"):
        for batch in _lake(handler, sleeps).sql_batches("SELECT 1"):
            received.append(batch)
    assert len(received) == 2  # what arrived was yielded, then the cut is raised; never a silent short result
    assert len(calls) == 1 and sleeps == []


@pytest.mark.unit
def test_a_stream_cut_mid_batch_raises_from_sql_batches():
    body, ends = _batched_body()
    handler, _ = _scripted(httpx.Response(200, headers=VERSION, stream=_CutStream(body, ends[0] + 5)))
    with pytest.raises(LakehouseError, match="stream cut after 1 batches"):
        list(_lake(handler, []).sql_batches("SELECT 1"))


@pytest.mark.unit
def test_sql_re_sends_a_cut_query_and_returns_only_the_complete_answer():
    body, ends = _batched_body()
    handler, calls = _scripted(httpx.Response(200, headers=VERSION, stream=_CutStream(body, ends[1])), _ok(body))
    sleeps: list[float] = []
    df = _lake(handler, sleeps).sql("SELECT 1")
    assert df["x"].tolist() == list(range(8))  # the two batches of the cut attempt are not duplicated
    assert len(calls) == 2 and sleeps == [5.0]


@pytest.mark.unit
def test_sql_gives_up_on_a_stream_that_is_always_cut():
    body, ends = _batched_body()
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(200, headers=VERSION, stream=_CutStream(body, ends[1]))

    sleeps: list[float] = []
    with pytest.raises(LakehouseError, match=r"stream cut after 2 batches.*gave up after 10 attempts"):
        _lake(handler, sleeps).sql("SELECT 1")
    assert sum(sleeps) <= client_mod.RETRY_BUDGET_S and len(calls) == 10


@pytest.mark.unit
def test_an_answer_without_a_build_version_is_refused():
    handler, _ = _scripted(httpx.Response(200, content=_arrow_body()))
    lake = _lake(handler, [])
    with pytest.raises(LakehouseError, match="the lake sent no build version"):
        lake.sql("SELECT 1")
    assert lake.version is None


@pytest.mark.unit
def test_require_version_false_accepts_an_answer_without_a_build_version():
    handler, _ = _scripted(httpx.Response(200, content=_arrow_body()))
    lake = _lake(handler, [], require_version=False)
    assert lake.sql("SELECT 1")["x"].tolist() == [1, 2]
    assert lake.version is None


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


class _VersionedLake:
    """Answers every query with one row and reports a fixed lake build, as `Lake.version` does."""

    version = "856ab2f64e97"

    def sql_batches(self, query, params=None):
        yield pa.RecordBatch.from_pylist([{"coach_id": 1, "name": "x"}])


@pytest.mark.unit
def test_sync_records_the_lake_build_per_window(tmp_path):
    with Warehouse(tmp_path / "w.duckdb") as wh:
        run_lake_sync(wh, _VersionedLake(), only=["tm_managers"], sleep=lambda s: None)
        versions = wh._conn.execute("SELECT DISTINCT lake_version FROM lake_sync_state").fetchall()
    assert versions == [("856ab2f64e97",)]


@pytest.mark.unit
def test_sync_adds_the_version_column_to_an_existing_state_table(tmp_path):
    with Warehouse(tmp_path / "w.duckdb") as wh:
        wh._conn.execute(
            "CREATE TABLE lake_sync_state (run_id VARCHAR, table_name VARCHAR, window_start VARCHAR, "
            "window_end VARCHAR, rows BIGINT, completed_at TIMESTAMP, PRIMARY KEY (run_id, window_start, window_end))"
        )
        run_lake_sync(wh, _VersionedLake(), only=["tm_managers"], sleep=lambda s: None)
        versions = wh._conn.execute("SELECT DISTINCT lake_version FROM lake_sync_state").fetchall()
    assert versions == [("856ab2f64e97",)]


@pytest.mark.unit
@pytest.mark.parametrize("at", ["after 2 batches", "after the last batch"])
def test_a_body_that_ends_cleanly_without_the_end_of_stream_marker_is_a_cut(at):
    body, ends = _batched_body()
    assert body.endswith(client_mod.ARROW_EOS) and len(body) == ends[-1] + len(client_mod.ARROW_EOS)
    cut = ends[1] if at == "after 2 batches" else ends[-1]
    handler, calls = _scripted(httpx.Response(200, headers=VERSION, content=body[:cut]), _ok(body))
    with pytest.raises(LakehouseError, match="ended without the Arrow end-of-stream marker"):
        list(_lake(handler, []).sql_batches("SELECT 1"))
    handler, calls = _scripted(httpx.Response(200, headers=VERSION, content=body[:cut]), _ok(body))
    sleeps: list[float] = []
    assert _lake(handler, sleeps).sql("SELECT 1")["x"].tolist() == list(range(8))  # sql() re-sends it
    assert len(calls) == 2 and sleeps == [5.0]


@pytest.mark.unit
def test_a_complete_answer_with_no_rows_is_accepted():
    schema = pa.schema([("x", pa.int64())])
    sink = io.BytesIO()
    with pa.ipc.new_stream(sink, schema):
        pass
    handler, calls = _scripted(_ok(sink.getvalue()))
    assert list(_lake(handler, []).sql_batches("SELECT 1")) == []
    assert len(calls) == 1
