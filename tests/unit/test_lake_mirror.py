"""Mocked tests for the lakehouse mirrors (no network)."""

from __future__ import annotations

from datetime import date

import httpx
import pyarrow as pa
import pytest

from pitch_edge.data.storage import TABLE_KEYS, Warehouse
from pitch_edge.lake import mirror
from pitch_edge.lake.client import Lake, LakehouseError
from pitch_edge.lake.mirror import (
    MIRRORS,
    assert_unique,
    is_retryable,
    run_lake_sync,
    sync_mirror,
    tenures_as_of,
    windows,
)


class FakeLake:
    """Returns canned rows per query and records every (sql, params) call."""

    def __init__(self, rows: list[dict] | None = None, fail_first: int = 0):
        self.rows = rows if rows is not None else []
        self.calls: list[tuple[str, list]] = []
        self.fail_first = fail_first

    def sql_batches(self, query, params=None):
        self.calls.append((query, params or []))
        if self.fail_first > 0:
            self.fail_first -= 1
            raise LakehouseError("HTTP 503: busy")
        if not self.rows or (params and params[0] != "2024-01-01" and params[0] != "2025-01-01"):
            return
        yield pa.RecordBatch.from_pylist(self.rows)


def _fd_rows(goals: int = 1) -> list[dict]:
    return [
        {
            "division": "E0",
            "match_date": date(2024, 8, 17),
            "home_team": "A",
            "away_team": "B",
            "ft_home_goals": goals,
            "scraped_at": None,
        },
        {
            "division": "E0",
            "match_date": date(2024, 8, 17),
            "home_team": "C",
            "away_team": "D",
            "ft_home_goals": 2,
            "scraped_at": None,
        },
    ]


@pytest.fixture
def wh(tmp_path):
    with Warehouse(tmp_path / "w.duckdb") as w:
        yield w


@pytest.mark.unit
def test_every_mirror_registered_in_table_keys():
    for spec in MIRRORS.values():
        assert spec.table.startswith("lake_")
        assert TABLE_KEYS[spec.table] == list(spec.keys)


@pytest.mark.unit
def test_resync_is_idempotent_and_updates_in_place(wh):
    spec = MIRRORS["fd_matches"]
    lake = FakeLake(_fd_rows(1))
    first = sync_mirror(wh, lake, spec, since="2024-01-01", until=date(2024, 12, 31), sleep=lambda s: None)
    assert first == 2
    sync_mirror(wh, lake, spec, since="2024-01-01", until=date(2024, 12, 31), sleep=lambda s: None)
    assert wh.count("lake_fd_matches") == 2
    assert "synced_at" in wh.columns("lake_fd_matches")
    lake.rows = _fd_rows(5)
    sync_mirror(wh, lake, spec, since="2024-01-01", until=date(2024, 12, 31), sleep=lambda s: None)
    df = wh.read("lake_fd_matches", "home_team = 'A'")
    assert wh.count("lake_fd_matches") == 2 and int(df["ft_home_goals"].iloc[0]) == 5


@pytest.mark.unit
def test_duplicate_keys_in_a_batch_collapse(wh):
    rows = _fd_rows() + _fd_rows()
    assert (
        sync_mirror(
            wh, FakeLake(rows), MIRRORS["fd_matches"], since="2024-01-01", until=date(2024, 1, 2), sleep=lambda s: None
        )
        == 2
    )
    assert wh.count("lake_fd_matches") == 2


@pytest.mark.unit
def test_window_parameters_are_used():
    spec = MIRRORS["pm_bars_1h"]
    wins = windows(spec, since="2025-01-01", until=date(2025, 3, 1))
    assert wins[0] == ("2025-01-01", "2025-01-31")
    assert all(a < b for a, b in wins) and wins[-1][1] > "2026-03-01"
    lake = FakeLake([])
    mirror.sync_mirror(
        Warehouse(":memory:"), lake, spec, since="2025-01-01", until=date(2025, 1, 5), sleep=lambda s: None
    )
    query, params = lake.calls[0]
    assert "{window}" not in query and query.count("?") == 2 and params == ["2025-01-01", "2025-01-31"]
    assert "pm_soccer_match_map" in query


@pytest.mark.unit
def test_unwindowed_mirror_runs_one_query(wh):
    lake = FakeLake([])
    sync_mirror(wh, lake, MIRRORS["tm_managers"])
    assert len(lake.calls) == 1 and lake.calls[0][1] == []


@pytest.mark.unit
def test_503_is_retried_with_backoff(wh):
    sleeps: list[float] = []
    lake = FakeLake([{"coach_id": 1, "name": "X"}, {"coach_id": 2, "name": "Y"}], fail_first=2)
    n = sync_mirror(wh, lake, MIRRORS["tm_managers"], sleep=sleeps.append)
    assert n == 2 and len(sleeps) == 2 and sleeps[1] > sleeps[0]


@pytest.mark.unit
def test_run_logs_health_and_isolates_failures(wh):
    class Boom(FakeLake):
        def sql_batches(self, query, params=None):
            raise ValueError("bad sql")
            yield  # pragma: no cover

    result = run_lake_sync(wh, Boom(), only=["tm_managers"])
    assert result == {"tm_managers": None}
    assert wh.read("pipeline_runs")["status"].tolist() == ["error"]
    with pytest.raises(KeyError):
        run_lake_sync(wh, FakeLake(), only=["nope"])


@pytest.mark.unit
def test_token_never_logged(monkeypatch, caplog):
    secret = "tok-SECRET-123"
    lake = Lake(base_url="http://127.0.0.1:1", token=secret)
    assert secret not in repr(lake.__dict__.get("base_url"))
    with caplog.at_level("DEBUG"):
        with pytest.raises(Exception):  # noqa: B017,PT011 - connection refused
            lake.health()
    assert secret not in caplog.text
    assert secret not in "".join(str(MIRRORS[n].sql) + MIRRORS[n].known_at_note for n in MIRRORS)


@pytest.mark.unit
@pytest.mark.parametrize(
    "exc,expected",
    [
        (httpx.ConnectTimeout("t"), True),
        (httpx.ReadError("r"), True),
        (LakehouseError("HTTP 503: all 2 query slots are busy"), True),
        (LakehouseError("HTTP 502: bad gateway"), True),
        (LakehouseError("HTTP 504: timeout"), True),
        (LakehouseError("the response was cut off: the query failed"), True),
        (LakehouseError("HTTP 400: Conversion Error near 503 rows"), False),
        (ValueError("503 in the message"), False),
        (LakehouseError("HTTP 401: bad token"), False),
    ],
)
def test_retry_classification(exc, expected):
    assert is_retryable(exc) is expected


@pytest.mark.unit
def test_non_retryable_error_is_not_retried(wh):
    class Bad(FakeLake):
        def sql_batches(self, query, params=None):
            self.calls.append((query, params or []))
            raise ValueError("limit 503 exceeded")
            yield  # pragma: no cover

    lake = Bad()
    sleeps: list[float] = []
    with pytest.raises(ValueError):
        sync_mirror(wh, lake, MIRRORS["tm_managers"], sleep=sleeps.append)
    assert len(lake.calls) == 1 and not sleeps


@pytest.mark.unit
def test_known_at_column_is_selected_by_every_spec():
    for spec in MIRRORS.values():
        if spec.known_at_column in ("known_at", "open_known_by"):
            assert f"AS {spec.known_at_column}" in spec.sql, spec.name
    assert MIRRORS["pm_bars_1h"].known_at_column == "known_at"
    assert "INTERVAL 1 HOUR" in MIRRORS["pm_bars_1h"].sql
    assert "INTERVAL 1 DAY" in MIRRORS["us_team_matches"].sql
    assert "INTERVAL 1 DAY" in MIRRORS["clubelo"].sql
    assert MIRRORS["fd_matches"].known_at_column == "open_known_by"
    assert "hindsight" in MIRRORS["fd_matches"].known_at_note.lower()
    assert "HINDSIGHT" in MIRRORS["tm_manager_tenures"].known_at_note


@pytest.mark.network
@pytest.mark.parametrize("name", list(MIRRORS))
def test_known_at_column_exists_in_live_query(name):
    spec = MIRRORS[name]
    if not spec.known_at_column:
        pytest.skip("no known-at column")
    from pitch_edge.lake.mirror import render_query

    query, params = render_query(spec, windows(spec, since=spec.default_start)[0])
    with Lake() as lake:
        cols = lake.sql(f"SELECT * FROM ({query}) LIMIT 0", params).columns
    assert spec.known_at_column in cols


@pytest.mark.unit
def test_tenures_as_of_masks_future_left(wh):
    conn = wh._conn
    conn.execute(
        "CREATE TABLE lake_tm_manager_tenures (club_id BIGINT, coach_id BIGINT, manager_name VARCHAR, "
        'appointed TIMESTAMPTZ, "left" TIMESTAMPTZ)'
    )
    conn.execute(
        "INSERT INTO lake_tm_manager_tenures VALUES "
        "(1, 1, 'A', '2020-01-01 00:00:00+00', '2022-01-01 00:00:00+00'),"
        "(1, 2, 'B', '2022-01-01 00:00:00+00', NULL),"
        "(1, 3, 'C', '2024-01-01 00:00:00+00', NULL)"
    )
    df = tenures_as_of(conn, "2021-06-01 00:00:00+00")
    assert df["manager_name"].tolist() == ["A"]
    assert df["left"].isna().all()
    df = tenures_as_of(conn, "2023-01-01 00:00:00+00")
    assert df.sort_values("manager_name")["left"].notna().tolist() == [True, False]


def _tm_rows(*names: str, scraped=None) -> list[dict]:
    return [{"coach_id": i, "name": n, "scraped_at": scraped} for i, n in enumerate(names, 1)]


@pytest.mark.unit
def test_insert_failure_rolls_back_and_leaves_table_unchanged(wh):
    spec = MIRRORS["tm_managers"]
    sync_mirror(wh, FakeLake(_tm_rows("X", "Y")), spec)
    before = wh.read("lake_tm_managers").sort_values("coach_id").reset_index(drop=True)
    wh._conn.execute("ALTER TABLE lake_tm_managers ALTER name SET DATA TYPE BIGINT USING 0")
    with pytest.raises(Exception):  # noqa: B017,PT011 - string cannot be inserted into BIGINT column
        sync_mirror(wh, FakeLake(_tm_rows("P", "Q", "R")), spec)
    assert wh.count("lake_tm_managers") == 2
    assert wh.read("lake_tm_managers")["coach_id"].sort_values().tolist() == before["coach_id"].tolist()
    # connection is usable and not stuck in a transaction
    wh._conn.execute("SELECT 1")
    assert "lake_incoming" not in [r[0] for r in wh._conn.execute("SHOW TABLES").fetchall()]


@pytest.mark.unit
def test_resync_window_removes_stale_and_rekeyed_rows(wh):
    spec = MIRRORS["fd_matches"]
    kw = {"since": "2024-01-01", "until": date(2024, 1, 2), "sleep": lambda s: None}
    sync_mirror(wh, FakeLake(_fd_rows()), spec, **kw)
    rekeyed = [{**_fd_rows()[0], "home_team": "A2"}]
    sync_mirror(wh, FakeLake(rekeyed), spec, **kw)
    assert sorted(wh.read("lake_fd_matches")["home_team"]) == ["A2"]
    sync_mirror(wh, FakeLake([]), spec, **kw)
    assert wh.count("lake_fd_matches") == 0


@pytest.mark.unit
def test_window_delete_leaves_other_windows_alone(wh):
    class ByWindow(FakeLake):
        def sql_batches(self, query, params=None):
            self.calls.append((query, params or []))
            rows = [r for r in self.rows if params[0] <= r["match_date"].isoformat() < params[1]]
            if rows:
                yield pa.RecordBatch.from_pylist(rows)

    spec = MIRRORS["fd_matches"]
    old = {**_fd_rows()[0], "match_date": date(2010, 5, 1), "home_team": "OLD"}
    kw = {"until": date(2024, 1, 2), "sleep": lambda s: None}
    sync_mirror(wh, ByWindow([old]), spec, since="2005-01-01", **kw)
    assert wh.count("lake_fd_matches") == 1
    sync_mirror(wh, ByWindow(_fd_rows()), spec, since="2024-01-01", **kw)
    assert sorted(wh.read("lake_fd_matches")["home_team"]) == ["A", "C", "OLD"]


@pytest.mark.unit
def test_full_refresh_clears_rows_outside_pulled_range(wh):
    spec = MIRRORS["tm_managers"]
    sync_mirror(wh, FakeLake(_tm_rows("X", "Y", "Z")), spec)
    sync_mirror(wh, FakeLake(_tm_rows("X")), spec, full_refresh=True)
    assert wh.count("lake_tm_managers") == 1


@pytest.mark.unit
def test_since_on_unwindowed_mirror_warns(wh, caplog):
    with caplog.at_level("WARNING"):
        sync_mirror(wh, FakeLake(_tm_rows("X")), MIRRORS["tm_managers"], since="2020-01-01")
    assert "ignored" in caplog.text


@pytest.mark.unit
def test_null_key_components_replace_in_place(wh):
    spec = MIRRORS["tm_manager_tenures"]
    row = {"club_id": 1, "coach_id": None, "manager_name": "A", "appointed": None, "left": None, "scraped_at": None}
    sync_mirror(wh, FakeLake([row]), spec)
    sync_mirror(wh, FakeLake([{**row, "manager_name": "B"}]), spec)
    df = wh.read("lake_tm_manager_tenures")
    assert len(df) == 1 and df["manager_name"].iloc[0] == "B"


@pytest.mark.unit
def test_latest_scraped_at_wins_and_null_loses(wh):
    from datetime import datetime

    spec = MIRRORS["tm_managers"]
    rows = [
        {"coach_id": 1, "name": "null", "scraped_at": None},
        {"coach_id": 1, "name": "old", "scraped_at": datetime(2026, 1, 1)},
        {"coach_id": 1, "name": "new", "scraped_at": datetime(2026, 2, 1)},
    ]
    sync_mirror(wh, FakeLake(rows), spec)
    assert wh.read("lake_tm_managers")["name"].tolist() == ["new"]
    sync_mirror(wh, FakeLake([rows[0], rows[1]]), spec)
    assert wh.read("lake_tm_managers")["name"].tolist() == ["old"]


@pytest.mark.unit
def test_assert_unique_detects_duplicates(wh):
    spec = MIRRORS["tm_managers"]
    sync_mirror(wh, FakeLake(_tm_rows("X")), spec)
    assert_unique(wh, spec)
    wh._conn.execute("INSERT INTO lake_tm_managers SELECT * FROM lake_tm_managers")
    with pytest.raises(RuntimeError, match="duplicated"):
        assert_unique(wh, spec)


@pytest.mark.unit
def test_cli_requires_token(monkeypatch):
    from typer.testing import CliRunner

    from pitch_edge.cli import app

    monkeypatch.setenv("LAKEHOUSE_API_TOKEN", "")
    result = CliRunner().invoke(app, ["lake-sync", "--only", "tm_managers"])
    assert result.exit_code == 2 and "LAKEHOUSE_API_TOKEN" in result.output


@pytest.mark.unit
@pytest.mark.parametrize("tz", ["Europe/Stockholm", "America/New_York", "UTC"])
def test_timestamptz_windows_ignore_session_timezone(wh, tz):
    """Rows in the last hours of a UTC window survive the next window's delete under any session zone."""
    from datetime import UTC, datetime, timedelta

    class Hours(FakeLake):
        def sql_batches(self, query, params=None):
            self.calls.append((query, params or []))
            lo = datetime.fromisoformat(params[0]).replace(tzinfo=UTC)
            hi = datetime.fromisoformat(params[1]).replace(tzinfo=UTC)
            rows = [
                {"asset_id": "a", "hour_start": t, "condition_id": "c"}
                for t in (lo + timedelta(hours=h) for h in range(0, (hi - lo).days * 24))
            ]
            if rows:
                yield pa.RecordBatch.from_pylist(rows, schema=None)

    wh._conn.execute(f"SET TimeZone='{tz}'")
    spec = MIRRORS["pm_bars_1h"]
    n = sync_mirror(
        wh, Hours(), spec, since="2025-03-01", until=date(2025, 3, 1), sleep=lambda s: None, full_refresh=True
    )
    stored = wh._conn.execute("SELECT count(*) FROM lake_pm_bars_1h").fetchone()[0]
    assert stored == n > 0
    # the hour before the 2025-03-31 window boundary must still be there
    boundary_hour = wh._conn.execute(
        "SELECT count(*) FROM lake_pm_bars_1h WHERE hour_start = TIMESTAMPTZ '2025-03-30 23:00:00+00'"
    ).fetchone()[0]
    assert boundary_hour == 1


class _ByWindowFailing(FakeLake):
    """Serves rows whose match_date falls in the queried window; raises a final error on window `fail_at`."""

    def __init__(self, rows, fail_at: str | None = None):
        super().__init__(rows)
        self.fail_at = fail_at

    def sql_batches(self, query, params=None):
        self.calls.append((query, params or []))
        if self.fail_at is not None and params and params[0] == self.fail_at:
            raise ValueError("simulated interruption")
        rows = [r for r in self.rows if params and params[0] <= r["match_date"].isoformat() < params[1]]
        if rows:
            yield pa.RecordBatch.from_pylist(rows)


def _rows_in_years(*years: int) -> list[dict]:
    return [{**_fd_rows()[0], "match_date": date(y, 3, 1), "home_team": f"H{y}"} for y in years]


@pytest.mark.unit
def test_interrupted_sync_resumes_without_requerying_done_windows(wh):
    spec = MIRRORS["fd_matches"]  # 365-day windows from 2005-01-01
    kw = {"since": "2005-01-01", "until": date(2008, 1, 2), "sleep": lambda s: None}
    rows = _rows_in_years(2005, 2006, 2007)
    first = _ByWindowFailing(rows, fail_at=windows(spec, "2005-01-01")[2][0])
    with pytest.raises(ValueError):
        sync_mirror(wh, first, spec, **kw)
    assert wh.count("lake_fd_matches") == 2  # the two windows before the interruption are committed

    second = _ByWindowFailing(rows)
    sync_mirror(wh, second, spec, **kw)
    queried = {p[0] for _, p in second.calls}
    assert windows(spec, "2005-01-01")[0][0] not in queried and windows(spec, "2005-01-01")[1][0] not in queried
    assert sorted(wh.read("lake_fd_matches")["home_team"]) == ["H2005", "H2006", "H2007"]
    open_runs = wh._conn.execute("SELECT count(*) FROM lake_sync_runs WHERE finished_at IS NULL").fetchone()[0]
    assert open_runs == 0


@pytest.mark.unit
def test_interrupted_full_refresh_is_not_re_emptied_on_resume(wh):
    spec = MIRRORS["fd_matches"]
    kw = {"until": date(2006, 1, 2), "sleep": lambda s: None}
    sync_mirror(wh, _ByWindowFailing(_rows_in_years(2005)), spec, since="2005-01-01", **kw)
    rows = _rows_in_years(2005, 2006)
    with pytest.raises(ValueError):
        sync_mirror(wh, _ByWindowFailing(rows, fail_at=windows(spec)[1][0]), spec, full_refresh=True, **kw)
    resumed = _ByWindowFailing(rows)
    sync_mirror(wh, resumed, spec, full_refresh=True, **kw)
    assert windows(spec)[0][0] not in {p[0] for _, p in resumed.calls}
    assert sorted(wh.read("lake_fd_matches")["home_team"]) == ["H2005", "H2006"]


@pytest.mark.unit
def test_finished_run_is_not_resumed(wh):
    spec = MIRRORS["fd_matches"]
    kw = {"since": "2005-01-01", "until": date(2006, 1, 2), "sleep": lambda s: None}
    sync_mirror(wh, _ByWindowFailing(_rows_in_years(2005)), spec, **kw)
    again = _ByWindowFailing(_rows_in_years(2005))
    sync_mirror(wh, again, spec, **kw)
    assert {p[0] for _, p in again.calls} == {w[0] for w in windows(spec, "2005-01-01", date(2006, 1, 2))}
