"""Loop-cycle gaps and evaluation-window coverage of the paper ledger."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from test_paper_report import HYP, FakeLedger, _add, registry  # noqa: F401
from typer.testing import CliRunner

from pitch_edge.cli import app
from pitch_edge.paper.report import (
    cycle_gaps,
    report_notes,
    summarise,
    window_coverage,
)
from pitch_edge.paper.store import PaperStore

pytestmark = pytest.mark.unit

H = timedelta(hours=1)
T0 = datetime(2026, 10, 21, 0, 0, tzinfo=UTC)


def _hours(*offsets: float) -> list[datetime]:
    return [T0 + timedelta(hours=o) for o in offsets]


def test_no_gap_when_cycles_arrive_about_every_hour():
    starts = _hours(0, 1, 2.2, 3.4, 4.4)  # drift below 1.5 x interval is not a gap
    assert cycle_gaps(starts, H) == []


def test_one_gap_counts_the_missed_cycles():
    starts = _hours(0, 1, 2, 7, 8)  # 5 h between cycles 3 and 4: 4 missed
    gaps = cycle_gaps(starts, H)
    assert len(gaps) == 1
    g = gaps[0]
    assert (g.start, g.end, g.missed) == (starts[2], starts[3], 4)


def test_gap_threshold_is_one_and_a_half_intervals():
    assert cycle_gaps(_hours(0, 1.5), H) == []
    gaps = cycle_gaps(_hours(0, 1.6), H)
    assert [g.missed for g in gaps] == [1]  # round(1.6) - 1


def test_fewer_than_two_cycles_have_no_gaps():
    assert cycle_gaps([], H) == []
    assert cycle_gaps(_hours(0), H) == []


def test_coverage_counts_hours_with_a_cycle():
    starts = _hours(0.1, 1.1, 2.1, 6.1, 7.1)  # hours 3, 4, 5 of the 8-hour window have no cycle
    assert window_coverage(starts, T0, T0 + 8 * H, H) == pytest.approx(5 / 8)


def test_coverage_counts_a_gap_at_the_start_of_the_window():
    starts = _hours(3.5, 4.5, 5.5, 6.5)  # first cycle three hours after the window opens
    assert window_coverage(starts, T0, T0 + 7 * H, H) == pytest.approx(4 / 7)


def test_coverage_ignores_cycles_outside_the_window_and_is_nan_for_an_empty_window():
    import math

    starts = _hours(-5, -4, 0.5)
    assert window_coverage(starts, T0, T0 + 2 * H, H) == pytest.approx(0.5)
    assert math.isnan(window_coverage(starts, T0, T0, H))


def test_store_cycle_starts_are_distinct_decide_times(tmp_path):
    with PaperStore(tmp_path / "p.duckdb") as store:
        for t in _hours(0, 0, 1):
            for sid in ("a", "b"):
                store.record_decide_run(sid, t, n_fixtures_evaluated=1, n_fixtures_qualifying=0, n_new_decisions=0)
        assert store.cycle_starts() == _hours(0, 1)


def test_summarise_reports_coverage_of_the_primary_segment(tmp_path, registry):  # noqa: F811
    with PaperStore(tmp_path / "p.duckdb") as store:
        _add(store, 1, datetime(2026, 10, 22, 9, tzinfo=UTC))
        # evaluation window opens at the later of 2026-10-21 and the ledger registration (2026-10-20 20:54)
        # so it opens 2026-10-21 00:00; cycles cover 4 of the first 6 hours
        for t in _hours(0.2, 1.2, 4.2, 5.2):
            store.record_decide_run(
                "soft-vs-sharp-live", t, n_fixtures_evaluated=1, n_fixtures_qualifying=0, n_new_decisions=0
            )
        rep = summarise(
            store,
            registry_dir=registry,
            ledger=FakeLedger(),
            n_boot=50,
            now=T0 + 6 * H,
        )
    row = rep.set_index("segment").loc["primary"]
    assert row["window_coverage"] == pytest.approx(4 / 6)
    notes = " ".join(report_notes(rep))
    assert "coverage" in notes.lower()
    assert "L5" in notes
    assert "67%" in notes


def test_cli_report_and_status_print_coverage_and_gaps(tmp_path, monkeypatch):
    db = tmp_path / "paper.duckdb"
    monkeypatch.setenv("PITCH_EDGE_PAPER_DB", str(db))
    with PaperStore(db) as store:
        _add(store, 1, datetime(2026, 10, 22, 9, tzinfo=UTC), hyp="control", strategy="control-fav-pinnacle")
        for t in _hours(0, 1, 2, 7, 8):
            store.record_decide_run(
                "control-fav-pinnacle", t, n_fixtures_evaluated=1, n_fixtures_qualifying=0, n_new_decisions=0
            )
    runner = CliRunner()
    res = runner.invoke(app, ["paper", "report", "--n-boot", "50"])
    assert res.exit_code == 0, res.output
    assert "coverage" in res.output.lower()
    assert "L5" in res.output
    res = runner.invoke(app, ["paper", "status"])
    assert res.exit_code == 0, res.output
    assert "cycles observed" in res.output
    assert "1 gap" in res.output
    assert "4 missed" in res.output


def test_a_gap_holding_a_recorded_pause_is_paused_not_missed():
    starts = _hours(0, 1, 2, 7, 8, 12, 13)
    gaps = cycle_gaps(starts, H, pauses=[(T0 + timedelta(hours=2.5), T0 + timedelta(hours=6.5))])
    assert [(g.missed, g.paused) for g in gaps] == [(4, True), (3, False)]


def test_cli_status_shows_pauses_apart_from_missed_cycles(tmp_path, monkeypatch):
    db = tmp_path / "paper.duckdb"
    monkeypatch.setenv("PITCH_EDGE_PAPER_DB", str(db))
    with PaperStore(db) as store:
        for t in _hours(0, 1, 2, 7, 8, 12, 13):
            store.record_decide_run(
                "control-fav-pinnacle", t, n_fixtures_evaluated=1, n_fixtures_qualifying=0, n_new_decisions=0
            )
    runner = CliRunner()
    for since in ("2026-10-21T02:30:00Z", "2026-10-21T13:30:00Z", "2026-10-21T13:30:00Z"):
        res = runner.invoke(app, ["paper", "pause", "--since", since, "--reason", "owner"])
        assert res.exit_code == 0, res.output
        if since.startswith("2026-10-21T02"):  # the owner restarts the loop before its 07:00 cycle
            res = runner.invoke(app, ["paper", "resume", "--at", "2026-10-21T06:30:00Z", "--reason", "owner"])
            assert res.exit_code == 0, res.output
    assert "already recorded" in res.output
    res = runner.invoke(app, ["paper", "status"])
    assert res.exit_code == 0, res.output
    assert "1 gap(s), 3 missed" in res.output
    assert "4 paused cycles" in res.output
    assert "paused since 2026-10-21 13:30" in res.output
