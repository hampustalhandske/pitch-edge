"""A pause recorded in the paper ledger stops the loop and every writing command until an explicit resume, and
coverage counts the paused hours as paused."""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from test_paper_report import FakeLedger, _add, registry  # noqa: F401
from typer.testing import CliRunner

from pitch_edge.cli import app
from pitch_edge.paper.report import cycle_gaps, report_notes, summarise, window_coverage, window_paused
from pitch_edge.paper.store import PaperStore, pause_intervals

pytestmark = pytest.mark.unit

H = timedelta(hours=1)
T0 = datetime(2026, 10, 21, 0, 0, tzinfo=UTC)
REPO = Path(__file__).resolve().parents[2]


def _hours(*offsets: float) -> list[datetime]:
    return [T0 + timedelta(hours=o) for o in offsets]


# --------------------------------------------------------------------------- intervals


def test_a_pause_lasts_until_the_first_resume_after_it():
    p, inner, r, later = _hours(2, 5, 9, 12)
    assert pause_intervals([p, inner], [r]) == [(p, r)]  # a pause inside an active one extends it
    assert pause_intervals([p, inner, later], [r]) == [(p, r), (later, None)]
    assert pause_intervals([], [r]) == []  # a resume with no pause adds nothing
    assert pause_intervals([later], [r]) == [(later, None)]  # a resume before the pause does not end it


def test_store_resume_ends_the_active_pause_and_refuses_without_one(tmp_path):
    with PaperStore(tmp_path / "p.duckdb") as store:
        assert not store.record_resume(T0 + 1 * H, "nothing to resume")
        assert store.record_pause(T0 + 2 * H, "owner")
        assert store.active_pause(T0 + 1 * H) is None
        assert store.active_pause(T0 + 50 * H) == T0 + 2 * H  # no resume: still paused
        assert store.record_resume(T0 + 9 * H, "owner reloaded the loop")
        assert not store.record_resume(T0 + 9 * H, "same resume twice")
        assert store.active_pause(T0 + 8 * H) == T0 + 2 * H
        assert store.active_pause(T0 + 9 * H) is None
        assert store.pause_intervals() == [(T0 + 2 * H, T0 + 9 * H)]


# --------------------------------------------------------------------------- commands during a pause


def _counts(db: Path) -> dict[str, int]:
    with PaperStore(db) as store:
        tables = [t for t in store.tables() if t not in ("pauses", "resumes")]
        return {t: store.con.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in tables}


def test_collect_decide_and_settle_write_nothing_while_paused(tmp_path, monkeypatch):
    db = tmp_path / "paper.duckdb"
    monkeypatch.setenv("PITCH_EDGE_PAPER_DB", str(db))
    with PaperStore(db) as store:
        store.record_pause(datetime(2026, 10, 28, 7, 48, tzinfo=UTC), "owner")

    def no_network(*args, **kwargs):
        raise AssertionError("a paused command must not fetch")

    import pitch_edge.lake.client as lake_client
    import pitch_edge.paper.collectors as collectors

    for name in ("PolymarketCollector", "KalshiCollector", "OddsApiCollector"):
        monkeypatch.setattr(collectors, name, no_network)
    monkeypatch.setattr(lake_client, "Lake", no_network)
    before = _counts(db)
    runner = CliRunner()
    for args in (["paper", "collect"], ["paper", "decide"], ["paper", "settle"]):
        res = runner.invoke(app, args)
        assert res.exit_code == 0, res.output
        assert "paused since 2026-10-28 07:48:00 UTC" in res.output
    assert _counts(db) == before
    res = runner.invoke(app, ["paper", "gate"])
    assert res.exit_code == 10, res.output


def test_resume_lets_the_gate_open_and_refuses_without_a_pause(tmp_path, monkeypatch):
    db = tmp_path / "paper.duckdb"
    monkeypatch.setenv("PITCH_EDGE_PAPER_DB", str(db))
    runner = CliRunner()
    assert runner.invoke(app, ["paper", "gate"]).exit_code == 0  # no ledger yet: not paused
    res = runner.invoke(app, ["paper", "resume", "--at", "2026-10-21T00:00:00Z", "--reason", "owner"])
    assert res.exit_code == 1  # nothing to resume
    assert runner.invoke(app, ["paper", "pause", "--since", "2026-10-21T01:00:00Z", "--reason", "owner"]).exit_code == 0
    assert runner.invoke(app, ["paper", "gate"]).exit_code == 10
    res = runner.invoke(app, ["paper", "resume", "--at", "2026-10-21T05:00:00Z", "--reason", "owner reloaded"])
    assert res.exit_code == 0, res.output
    assert runner.invoke(app, ["paper", "gate"]).exit_code == 0


def _fake_uv(bin_dir: Path, calls: Path, gate_rc: int) -> None:
    """Stub `arch` on PATH: `arch -arm64 uv run pitch-edge <args>` appends <args> to `calls`; `paper gate`
    exits `gate_rc`, every other command 0."""
    arch = bin_dir / "arch"
    arch.write_text(
        "#!/bin/bash\n"
        "shift; shift; shift; shift\n"  # -arm64 uv run pitch-edge
        f'echo "$*" >> "{calls}"\n'
        f'if [ "$*" = "paper gate" ]; then exit {gate_rc}; fi\n'
        "exit 0\n"
    )
    arch.chmod(arch.stat().st_mode | stat.S_IEXEC)


def _run_loop(tmp_path: Path, gate_rc: int, timeout: float = 20) -> tuple[subprocess.CompletedProcess, str, str]:
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    shutil.copy(REPO / "scripts" / "paper_loop.sh", root / "scripts" / "paper_loop.sh")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "calls.txt"
    _fake_uv(bin_dir, calls, gate_rc)
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "PAPER_LOOP_INTERVAL_S": "3600"}
    proc = subprocess.run(
        ["bash", str(root / "scripts" / "paper_loop.sh")], env=env, capture_output=True, text=True, timeout=timeout
    )
    log = (root / "data" / "logs" / "paper_loop.log").read_text()
    return proc, calls.read_text() if calls.exists() else "", log


def test_the_loop_exits_cleanly_at_a_ledger_pause_without_running_a_cycle(tmp_path):
    proc, calls, log = _run_loop(tmp_path, gate_rc=10)
    assert proc.returncode == 0
    assert calls.splitlines() == ["paper gate"]  # no collect, decide or settle
    assert "paused in the ledger" in log
    assert "cycle start" not in log
    assert not (tmp_path / "repo" / "data" / "paper" / "paper_loop.pid").exists()


def test_the_loop_skips_the_cycle_when_the_pause_check_fails(tmp_path):
    with pytest.raises(subprocess.TimeoutExpired):  # it keeps waiting for the next cycle instead of trading
        _run_loop(tmp_path, gate_rc=1, timeout=3)
    calls = (tmp_path / "calls.txt").read_text().splitlines()
    assert calls == ["paper gate"]
    log = (tmp_path / "repo" / "data" / "logs" / "paper_loop.log").read_text()
    assert "pause check failed (exit 1); cycle skipped" in log
    assert "cycle start" not in log


# --------------------------------------------------------------------------- coverage


def test_cycles_inside_a_pause_are_not_coverage_and_its_hours_read_paused():
    starts = _hours(0, 1, 4, 5, 6, 7)
    pauses = [(T0 + 4 * H, T0 + 7 * H)]  # cycles at 4, 5, 6 ran inside the pause
    assert window_coverage(starts, T0, T0 + 8 * H, H) == pytest.approx(6 / 8)
    assert window_coverage(starts, T0, T0 + 8 * H, H, pauses) == pytest.approx(3 / 8)
    assert window_paused(T0, T0 + 8 * H, pauses, H) == pytest.approx(3 / 8)
    assert window_paused(T0, T0 + 8 * H, [(T0 + 6 * H, None)], H) == pytest.approx(2 / 8)  # still active
    assert window_paused(T0, T0 + 8 * H, [], H) == 0


def test_a_gap_overlapping_a_pause_interval_is_paused_not_missed():
    starts = _hours(0, 1, 2, 7, 8, 12, 13)
    assert [(g.missed, g.paused) for g in cycle_gaps(starts, H, [(T0 + 9 * H, T0 + 11 * H)])] == [
        (4, False),
        (3, True),
    ]
    assert [g.paused for g in cycle_gaps(starts, H, [(T0 + 1.5 * H, None)])] == [True, True]


def test_summarise_counts_pause_hours_as_paused_not_covered(tmp_path, registry):  # noqa: F811
    with PaperStore(tmp_path / "p.duckdb") as store:
        _add(store, 1, datetime(2026, 10, 22, 9, tzinfo=UTC))
        for t in _hours(0.2, 1.2, 2.2, 3.2, 4.2, 5.2):
            store.record_decide_run(
                "soft-vs-sharp-live", t, n_fixtures_evaluated=1, n_fixtures_qualifying=0, n_new_decisions=0
            )
        store.record_pause(T0 + 3 * H, "owner")  # no resume: hours 3-5 are paused although cycles ran
        rep = summarise(store, registry_dir=registry, ledger=FakeLedger(), n_boot=50, now=T0 + 6 * H)
    row = rep.set_index("segment").loc["primary"]
    assert row["window_coverage"] == pytest.approx(3 / 6)
    assert row["window_paused"] == pytest.approx(3 / 6)
    assert "50% of evaluation-window hours covered, 50% paused" in " ".join(report_notes(rep))
