"""Controls in code: the lake freeze log and what it stops (`Lake()`, `lake-sync`, `control gate`), the env-file
override, and the scheduled scripts' start-of-run guard (a dirty tree or an unpinned HEAD refuses) and pin script,
run against temporary git repositories with a stubbed `arch` on PATH."""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from pitch_edge import control
from pitch_edge.cli import app
from pitch_edge.config import load_dotenv
from pitch_edge.control import (
    FreezeLogError,
    FreezeStateError,
    active_freeze,
    active_freeze_event,
    freeze_log_path,
    record_freeze,
    record_thaw,
)
from pitch_edge.lake.client import Lake, LakeFrozenError

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts"
H = timedelta(hours=1)
T0 = datetime(2026, 11, 10, 9, 11, tzinfo=UTC)


# --------------------------------------------------------------------------- freeze log semantics


def test_a_freeze_lasts_until_the_first_thaw_after_it():
    record_freeze(T0, "lake rebuild")
    assert active_freeze(T0 - H) is None
    assert active_freeze(T0) == T0
    assert active_freeze(T0 + 50 * H) == T0  # no thaw: still frozen
    record_thaw(T0 + 5 * H, "rebuild done")
    assert active_freeze(T0 + 5 * H - timedelta(seconds=1)) == T0
    assert active_freeze(T0 + 5 * H) is None
    assert active_freeze_event(T0 + H).reason == "lake rebuild"
    lines = freeze_log_path().read_text().splitlines()
    assert [line.split('"event": "')[1].split('"')[0] for line in lines] == ["freeze", "thaw"]  # appended


def test_thaw_refuses_without_an_active_freeze_and_a_second_freeze_is_refused():
    with pytest.raises(FreezeStateError, match="no lake freeze is active"):
        record_thaw(T0, "nothing to thaw")
    record_freeze(T0, "lake rebuild")
    with pytest.raises(FreezeStateError, match="already active since 2026-11-10 09:11:00"):
        record_freeze(T0 + H, "again")
    with pytest.raises(FreezeStateError):
        record_thaw(T0 - H, "a thaw before the freeze ends nothing")
    record_thaw(T0 + 2 * H, "done")
    with pytest.raises(FreezeStateError):
        record_thaw(T0 + 2 * H, "same thaw twice")
    record_freeze(T0 + 3 * H, "second rebuild")
    assert active_freeze(T0 + 4 * H) == T0 + 3 * H


def test_intervals_follow_the_paper_pause_rule():
    """Same rule as `paper.store.pause_intervals`: a thaw before a freeze at the same time, freezes inside an active
    one extend it, thaws without an active freeze add nothing."""
    from pitch_edge.paper.store import pause_intervals

    def ev(kind: str, h: float) -> control.FreezeEvent:
        return control.FreezeEvent(kind, T0 + h * H, "r", T0)

    cases = [
        ([ev("freeze", 2), ev("freeze", 5), ev("thaw", 9)]),
        ([ev("freeze", 2), ev("freeze", 5), ev("freeze", 12), ev("thaw", 9)]),
        ([ev("thaw", 9)]),
        ([ev("freeze", 12), ev("thaw", 9)]),
        ([ev("freeze", 4), ev("thaw", 4)]),
    ]
    for events in cases:
        got = [(start.at, end) for start, end in control.freeze_intervals(events)]
        expected = pause_intervals(
            [e.at for e in events if e.event == "freeze"], [e.at for e in events if e.event == "thaw"]
        )
        assert got == expected


def test_naive_times_are_rejected():
    with pytest.raises(ValueError, match="timezone-aware"):
        record_freeze(datetime(2026, 11, 10, 9, 11), "naive")


@pytest.mark.parametrize(
    "line",
    [
        "not json",
        '["freeze"]',
        '{"event": "freeze", "at": "2026-11-10T09:11:00+00:00", "reason": "x"}',  # no recorded_at
        '{"event": "melt", "at": "2026-11-10T09:11:00+00:00", "reason": "x", "recorded_at": "2026-11-10T09:11:00+00:00"}',
        '{"event": "freeze", "at": "2026-11-10T09:11:00", "reason": "x", "recorded_at": "2026-11-10T09:11:00+00:00"}',
        '{"event": "freeze", "at": "yesterday", "reason": "x", "recorded_at": "2026-11-10T09:11:00+00:00"}',
        '{"event": "freeze", "at": "2026-11-10T09:11:00+00:00", "reason": 3, "recorded_at": "2026-11-10T09:11:00+00:00"}',
    ],
    ids=["not-json", "not-object", "missing-key", "unknown-event", "naive-time", "bad-time", "reason-not-text"],
)
def test_a_corrupt_log_fails_closed(line):
    record_freeze(T0, "lake rebuild")
    record_thaw(T0 + H, "done")  # the readable lines alone say "not frozen"
    with freeze_log_path().open("a") as fh:
        fh.write(line + "\n")
    with pytest.raises(FreezeLogError, match="line 3 is not a valid freeze event"):
        active_freeze(T0 + 2 * H)
    with pytest.raises(LakeFrozenError, match="unreadable, so the lake is treated as frozen"):
        Lake(base_url="http://lake.test", token="t")
    with pytest.raises(FreezeLogError):
        record_freeze(T0 + 3 * H, "cannot append to a corrupt log")
    res = CliRunner().invoke(app, ["control", "gate"])
    assert res.exit_code == 10, res.output
    assert "treated as frozen" in res.output


def test_an_unreadable_log_fails_closed():
    path = freeze_log_path()
    path.mkdir(parents=True)  # a directory where the file should be: exists but cannot be read
    with pytest.raises(FreezeLogError, match="cannot read"):
        active_freeze()
    with pytest.raises(LakeFrozenError):
        Lake(base_url="http://lake.test", token="t")


def test_a_missing_log_means_not_frozen():
    assert not freeze_log_path().exists()
    assert active_freeze() is None


# --------------------------------------------------------------------------- what the freeze stops


def test_lake_refuses_while_frozen_and_reads_again_after_a_thaw():
    now = datetime.now(UTC)
    record_freeze(now - H, "lake rebuild LK-111")
    with pytest.raises(LakeFrozenError, match=r"lake freeze since .* UTC \(lake rebuild LK-111\)"):
        Lake(base_url="http://lake.test", token="t")
    record_thaw(now - timedelta(minutes=1), "rebuild done")
    Lake(base_url="http://lake.test", token="t").close()


def test_a_future_freeze_does_not_stop_the_lake_yet():
    record_freeze(datetime.now(UTC) + 24 * H, "planned rebuild")
    Lake(base_url="http://lake.test", token="t").close()


def test_control_commands_and_gate_exit_codes():
    runner = CliRunner()
    res = runner.invoke(app, ["control", "gate"])
    assert res.exit_code == 0 and "lake not frozen" in res.output
    res = runner.invoke(app, ["control", "thaw", "--reason", "nothing to thaw"])
    assert res.exit_code == 1 and "thaw not recorded" in res.output
    res = runner.invoke(app, ["control", "freeze", "--reason", "lake rebuild", "--at", "2026-11-10T09:11:00Z"])
    assert res.exit_code == 0, res.output
    res = runner.invoke(app, ["control", "freeze", "--reason", "again"])
    assert res.exit_code == 1 and "already active" in res.output
    res = runner.invoke(app, ["control", "gate"])
    assert res.exit_code == 10
    assert "lake freeze since 2026-11-10 09:11:00 UTC (lake rebuild)" in res.output
    res = runner.invoke(app, ["control", "status"])
    assert res.exit_code == 0 and "lake freeze since 2026-11-10 09:11:00 UTC (lake rebuild)" in res.output
    res = runner.invoke(app, ["control", "thaw", "--reason", "rebuild done"])
    assert res.exit_code == 0, res.output
    assert runner.invoke(app, ["control", "gate"]).exit_code == 0
    assert "lake not frozen" in runner.invoke(app, ["control", "status"]).output


def _forbid_lake_and_http(monkeypatch) -> None:
    def no_lake(*args, **kwargs):
        raise AssertionError("lake-sync must not contact the lake during a freeze")

    monkeypatch.setattr("pitch_edge.lake.client.Lake", no_lake)
    monkeypatch.setattr(httpx.Client, "send", no_lake)


def test_lake_sync_skips_without_contacting_the_lake_while_frozen(monkeypatch, tmp_path):
    monkeypatch.setenv("LAKEHOUSE_API_TOKEN", "t")
    _forbid_lake_and_http(monkeypatch)
    record_freeze(datetime(2026, 11, 10, 9, 11, tzinfo=UTC), "lake rebuild")
    res = CliRunner().invoke(app, ["lake-sync", "--only", "fd_matches", "--skip-if-busy"])
    assert res.exit_code == 0, res.output
    assert "skipped: lake freeze since 2026-11-10 09:11:00 UTC (lake rebuild)" in res.output
    assert not (tmp_path / "data" / "pitch_edge.duckdb").exists()  # the warehouse was not opened either


def test_lake_sync_skips_while_the_freeze_log_is_corrupt(monkeypatch):
    monkeypatch.setenv("LAKEHOUSE_API_TOKEN", "t")
    _forbid_lake_and_http(monkeypatch)
    freeze_log_path().parent.mkdir(parents=True)
    freeze_log_path().write_text("{broken\n")
    res = CliRunner().invoke(app, ["lake-sync", "--only", "fd_matches"])
    assert res.exit_code == 0, res.output
    assert "skipped: the lake freeze log is unreadable" in res.output


def test_env_file_comes_from_pitch_edge_env_file(tmp_path, monkeypatch):
    env = tmp_path / "main-checkout.env"
    env.write_text("PE_TEST_ENV_FILE_KEY=from-main\n")
    monkeypatch.delenv("PE_TEST_ENV_FILE_KEY", raising=False)
    monkeypatch.setenv("PITCH_EDGE_ENV_FILE", str(env))
    monkeypatch.chdir(tmp_path)  # no .env in the working directory
    assert load_dotenv() == 1
    assert os.environ["PE_TEST_ENV_FILE_KEY"] == "from-main"


# --------------------------------------------------------------------------- scheduled scripts: helpers

GIT = [
    "git",
    "-c",
    "user.name=test",
    "-c",
    "user.email=test@example.invalid",
    "-c",
    "commit.gpgsign=false",
    "-c",
    "core.hooksPath=/dev/null",
]


def git(root: Path, *args: str) -> str:
    return subprocess.run([*GIT, "-C", str(root), *args], check=True, capture_output=True, text=True).stdout.strip()


def make_scheduled_tree(root: Path, scripts: list[str], pin: bool = True) -> str:
    """A git repository at `root` holding copies of `scripts` (plus the guard) with `data/` ignored, committed
    once; with `pin`, `root/data/control/scheduled_pin.log` names that commit. Returns the commit sha."""
    (root / "scripts").mkdir(parents=True)
    for name in [*scripts, "scheduled_guard.sh"]:
        shutil.copy(SCRIPTS / name, root / "scripts" / name)
    (root / ".gitignore").write_text("data/\n")
    git(root, "init", "-q")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "init")
    sha = git(root, "rev-parse", "HEAD")
    if pin:
        write_pin(root / "data", sha)
    return sha


def write_pin(data_dir: Path, sha: str) -> None:
    log = data_dir / "control" / "scheduled_pin.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as fh:
        fh.write(f"2026-11-10T09:00:00Z {sha}\n")


def fake_arch(bin_dir: Path, calls: Path, gate_rc: dict[str, int] | None = None) -> None:
    """Stub `arch` on PATH. `arch -arm64 uv run pitch-edge <args>` appends <args> to `calls` and exits
    `gate_rc[<args>]` (default 0); any other `arch` call (e.g. `uv sync`) appends `arch <its args>` and exits 0."""
    bin_dir.mkdir(exist_ok=True)
    cases = "".join(f'  "{args}") exit {rc} ;;\n' for args, rc in (gate_rc or {}).items())
    script = bin_dir / "arch"
    script.write_text(
        "#!/bin/bash\n"
        'if [ "$1 $2 $3 $4" = "-arm64 uv run pitch-edge" ]; then\n'
        "  shift 4\n"
        f'  echo "$*" >> "{calls}"\n'
        '  case "$*" in\n'
        f"{cases}"
        "  esac\n"
        "  exit 0\n"
        "fi\n"
        f'echo "arch $*" >> "{calls}"\n'
        "exit 0\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)


def run_script(
    tmp_path: Path, script: Path, gate_rc: dict[str, int] | None = None, data_dir: Path | None = None, timeout=20
) -> tuple[subprocess.CompletedProcess, list[str]]:
    calls = tmp_path / "calls.txt"
    calls.unlink(missing_ok=True)
    fake_arch(tmp_path / "bin", calls, gate_rc)
    env = {**os.environ, "PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}", "PAPER_LOOP_INTERVAL_S": "3600"}
    env["PITCH_EDGE_DATA_DIR"] = str(data_dir if data_dir is not None else script.parents[1] / "data")
    proc = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True, timeout=timeout)
    return proc, calls.read_text().splitlines() if calls.exists() else []


# --------------------------------------------------------------------------- lake_sync_fd.sh


def _sync_tree(tmp_path: Path, pin: bool = True) -> tuple[Path, str]:
    root = tmp_path / "repo"
    return root, make_scheduled_tree(root, ["lake_sync_fd.sh"], pin=pin)


def _sync_log(root: Path) -> str:
    return (root / "data" / "logs" / "lake_sync_fd.log").read_text()


SYNC = "lake-sync --only fd_matches --skip-if-busy"


def test_lake_sync_script_runs_on_a_clean_pinned_tree(tmp_path):
    root, _ = _sync_tree(tmp_path)
    proc, calls = run_script(tmp_path, root / "scripts" / "lake_sync_fd.sh")
    assert proc.returncode == 0, proc.stderr
    assert calls == ["control gate", SYNC]
    assert "lake-sync fd_matches exit 0" in _sync_log(root)


def test_lake_sync_script_exits_0_without_syncing_while_the_gate_says_frozen(tmp_path):
    root, _ = _sync_tree(tmp_path)
    proc, calls = run_script(tmp_path, root / "scripts" / "lake_sync_fd.sh", gate_rc={"control gate": 10})
    assert proc.returncode == 0
    assert calls == ["control gate"]
    assert "skipped: lake frozen" in _sync_log(root)
    assert "lake-sync fd_matches start" not in _sync_log(root)


def test_lake_sync_script_does_not_sync_when_the_gate_fails(tmp_path):
    root, _ = _sync_tree(tmp_path)
    proc, calls = run_script(tmp_path, root / "scripts" / "lake_sync_fd.sh", gate_rc={"control gate": 1})
    assert proc.returncode == 1
    assert calls == ["control gate"]
    assert "freeze check failed (exit 1); no sync" in _sync_log(root)


@pytest.mark.parametrize("change", ["modified", "untracked", "staged"])
def test_lake_sync_script_refuses_a_dirty_tree(tmp_path, change):
    root, _ = _sync_tree(tmp_path)
    if change == "modified":
        with (root / "scripts" / "lake_sync_fd.sh").open("a") as fh:
            fh.write("# half-edited\n")
    elif change == "untracked":
        (root / "scripts" / "new_module.py").write_text("x = 1\n")
    else:
        (root / "staged.txt").write_text("x\n")
        git(root, "add", "staged.txt")
    proc, calls = run_script(tmp_path, root / "scripts" / "lake_sync_fd.sh")
    assert proc.returncode == 0
    assert calls == []  # neither the gate nor the sync ran
    assert "has uncommitted changes" in _sync_log(root)


def test_lake_sync_script_refuses_a_head_that_is_not_the_pin(tmp_path):
    root, pinned = _sync_tree(tmp_path)
    git(root, "commit", "-q", "--allow-empty", "-m", "unpinned commit")
    proc, calls = run_script(tmp_path, root / "scripts" / "lake_sync_fd.sh")
    assert proc.returncode == 0
    assert calls == []
    assert f"is not the pinned sha {pinned}" in _sync_log(root)


def test_lake_sync_script_refuses_without_a_pin(tmp_path):
    root, _ = _sync_tree(tmp_path, pin=False)
    proc, calls = run_script(tmp_path, root / "scripts" / "lake_sync_fd.sh")
    assert proc.returncode == 0
    assert calls == []
    assert "no pinned sha" in _sync_log(root)


def test_lake_sync_script_follows_the_last_pin(tmp_path):
    root, first = _sync_tree(tmp_path)
    git(root, "commit", "-q", "--allow-empty", "-m", "second")
    write_pin(root / "data", git(root, "rev-parse", "HEAD"))  # the newest line wins
    proc, calls = run_script(tmp_path, root / "scripts" / "lake_sync_fd.sh")
    assert proc.returncode == 0 and calls == ["control gate", SYNC]


# --------------------------------------------------------------------------- paper_loop.sh


def _loop_tree(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "repo"
    return root, make_scheduled_tree(root, ["paper_loop.sh"])


def _loop_log(root: Path) -> str:
    return (root / "data" / "logs" / "paper_loop.log").read_text()


def test_paper_loop_runs_on_a_clean_pinned_tree(tmp_path):
    root, _ = _loop_tree(tmp_path)
    proc, calls = run_script(tmp_path, root / "scripts" / "paper_loop.sh", gate_rc={"paper gate": 10})
    assert proc.returncode == 0
    assert calls == ["paper gate"]  # it passed the guard and reached the ledger's pause check
    assert "paused in the ledger" in _loop_log(root)


def test_paper_loop_refuses_a_dirty_tree(tmp_path):
    root, _ = _loop_tree(tmp_path)
    (root / "scripts" / "paper_loop.sh").write_text((root / "scripts" / "paper_loop.sh").read_text() + "# edit\n")
    proc, calls = run_script(tmp_path, root / "scripts" / "paper_loop.sh", gate_rc={"paper gate": 10})
    assert proc.returncode == 0  # a clean exit, so launchd's KeepAlive does not respawn it
    assert calls == []
    assert "has uncommitted changes" in _loop_log(root)
    assert "paper loop started" not in _loop_log(root)
    assert not (root / "data" / "paper" / "paper_loop.pid").exists()


def test_paper_loop_refuses_a_head_that_is_not_the_pin(tmp_path):
    root, pinned = _loop_tree(tmp_path)
    git(root, "commit", "-q", "--allow-empty", "-m", "unpinned commit")
    proc, calls = run_script(tmp_path, root / "scripts" / "paper_loop.sh", gate_rc={"paper gate": 10})
    assert proc.returncode == 0
    assert calls == []
    assert f"is not the pinned sha {pinned}" in _loop_log(root)


def test_scripts_use_pitch_edge_data_dir_for_logs_and_the_pin(tmp_path):
    root, sha = _loop_tree(tmp_path)
    shutil.rmtree(root / "data")
    elsewhere = tmp_path / "main-data"
    write_pin(elsewhere, sha)
    proc, calls = run_script(tmp_path, root / "scripts" / "paper_loop.sh", {"paper gate": 10}, data_dir=elsewhere)
    assert proc.returncode == 0 and calls == ["paper gate"]
    assert "paused in the ledger" in (elsewhere / "logs" / "paper_loop.log").read_text()
    assert not (root / "data").exists()


# --------------------------------------------------------------------------- scheduled_pin.sh


def _pin_repo(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "main"
    sha = make_scheduled_tree(root, ["lake_sync_fd.sh", "scheduled_pin.sh"], pin=False)
    return root, sha


def _pin(tmp_path: Path, root: Path, arg: str) -> tuple[subprocess.CompletedProcess, list[str]]:
    calls = tmp_path / "calls.txt"
    calls.unlink(missing_ok=True)
    fake_arch(tmp_path / "bin", calls)
    env = {k: v for k, v in os.environ.items() if k != "PITCH_EDGE_DATA_DIR"}  # default: <repo>/data
    env["PATH"] = f"{tmp_path / 'bin'}:{os.environ['PATH']}"
    proc = subprocess.run(
        ["bash", str(root / "scripts" / "scheduled_pin.sh"), arg], env=env, capture_output=True, text=True, timeout=60
    )
    return proc, calls.read_text().splitlines() if calls.exists() else []


@pytest.mark.parametrize("arg", ["", "not-a-ref", "HEAD:scripts"], ids=["empty", "unknown", "tree-object"])
def test_pin_refuses_anything_that_is_not_a_commit(tmp_path, arg):
    root, _ = _pin_repo(tmp_path)
    proc, calls = _pin(tmp_path, root, arg)
    assert proc.returncode == 2
    assert "refused" in proc.stdout
    assert not (root / ".worktrees" / "scheduled").exists()
    assert not (root / "data" / "control" / "scheduled_pin.log").exists()
    assert calls == []


def test_pin_creates_and_moves_the_scheduled_worktree_and_the_jobs_run_from_it(tmp_path):
    root, first = _pin_repo(tmp_path)
    proc, calls = _pin(tmp_path, root, first[:10])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    wt = root / ".worktrees" / "scheduled"
    assert git(wt, "rev-parse", "HEAD") == first
    assert calls == ["arch -arm64 uv sync --extra dev --locked"]
    pin_log = root / "data" / "control" / "scheduled_pin.log"
    assert pin_log.read_text().splitlines()[-1].split()[1] == first  # the full sha, not the short form given

    # the scheduled job, run from the pinned worktree with the main checkout's data dir, passes its guard
    proc, calls = run_script(tmp_path, wt / "scripts" / "lake_sync_fd.sh", data_dir=root / "data")
    assert proc.returncode == 0 and calls == ["control gate", SYNC]

    git(root, "commit", "-q", "--allow-empty", "-m", "second")
    second = git(root, "rev-parse", "HEAD")
    proc, _ = _pin(tmp_path, root, second)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert git(wt, "rev-parse", "HEAD") == second
    assert [line.split()[1] for line in pin_log.read_text().splitlines()] == [first, second]


def test_pin_refuses_to_move_a_dirty_scheduled_worktree(tmp_path):
    root, first = _pin_repo(tmp_path)
    assert _pin(tmp_path, root, first)[0].returncode == 0
    wt = root / ".worktrees" / "scheduled"
    (wt / "scripts" / "stray.py").write_text("x = 1\n")
    git(root, "commit", "-q", "--allow-empty", "-m", "second")
    proc, _ = _pin(tmp_path, root, "HEAD")
    assert proc.returncode == 3 and "uncommitted changes" in proc.stdout
    assert git(wt, "rev-parse", "HEAD") == first
    assert len((root / "data" / "control" / "scheduled_pin.log").read_text().splitlines()) == 1


def test_paper_settle_skips_cleanly_while_frozen(monkeypatch, tmp_path):
    monkeypatch.setenv("LAKEHOUSE_API_TOKEN", "t")
    monkeypatch.setenv("PITCH_EDGE_PAPER_DB", str(tmp_path / "paper.duckdb"))
    monkeypatch.setattr(httpx.Client, "send", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no request")))
    record_freeze(datetime(2026, 11, 10, 9, 11, tzinfo=UTC), "lake rebuild")
    res = CliRunner().invoke(app, ["paper", "settle"])
    assert res.exit_code == 0, res.output
    assert "settle skipped: lake freeze since 2026-11-10 09:11:00 UTC (lake rebuild)" in res.output
    assert not (tmp_path / "paper.duckdb").exists()  # the ledger was not opened
