"""The local gate scripts (scripts/ci/): each refuses an argument that is not a commit, and macos_ci.sh reports a
planted failing step as `RESULT rc=1` and `GATE FAIL` with a non-zero exit, and all-green steps as `GATE PASS`.

The macOS runs use a throwaway git repo and a stub `uv` on PATH, so nothing is installed or tested for real; they
need macOS (BSD `/usr/bin/time -l`, `arch -arm64`). linux_ci.sh is checked only up to its argument guard, with
stub `open`/`docker`/`osascript` that record any call, so Docker is never started.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
MACOS = REPO / "scripts" / "ci" / "macos_ci.sh"
LINUX = REPO / "scripts" / "ci" / "linux_ci.sh"

STUB_UV = """#!/bin/bash
# Stub uv: exits 1 when its arguments start with the text in ./fail_on next to it, else 0.
here=$(cd "$(dirname "$0")" && pwd)
echo "stub uv $*"
if [ -s "$here/fail_on" ] && [[ "$*" == "$(cat "$here/fail_on")"* ]]; then echo "1 failed"; exit 1; fi
echo "1 passed"
"""

STUB_RECORDER = """#!/bin/bash
echo "$(basename "$0") $*" >> "$(cd "$(dirname "$0")" && pwd)/calls.log"
"""


def _can_run_macos_gate() -> bool:
    if sys.platform != "darwin" or not Path("/usr/bin/time").exists():
        return False
    return subprocess.run(["arch", "-arm64", "/usr/bin/true"], capture_output=True).returncode == 0


macos_only = pytest.mark.skipif(not _can_run_macos_gate(), reason="needs macOS with arch -arm64 and /usr/bin/time")


def _git(repo: Path, *args: str) -> str:
    env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
    out = subprocess.run(["git", *args], cwd=repo, env=env, capture_output=True, text=True, check=True)
    return out.stdout.strip()


@pytest.fixture
def repo(tmp_path) -> tuple[Path, str]:
    """A git repo with one commit, and that commit's sha."""
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    (root / "README").write_text("x\n")
    _git(root, "add", "README")
    _git(root, "-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-q", "-m", "one")
    return root, _git(root, "rev-parse", "HEAD")


def _stubs(tmp_path: Path, names_and_bodies: dict[str, str]) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    for name, body in names_and_bodies.items():
        f = bin_dir / name
        f.write_text(body)
        f.chmod(0o755)
    return bin_dir


def _run(script: Path, arg: str, cwd: Path, bin_dir: Path) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"}
    return subprocess.run(["bash", str(script), arg], cwd=cwd, env=env, capture_output=True, text=True, timeout=120)


@pytest.mark.parametrize("script", [MACOS, LINUX], ids=["macos", "linux"])
def test_a_non_commit_argument_is_refused_before_anything_runs(script, repo, tmp_path):
    root, _ = repo
    bin_dir = _stubs(tmp_path, dict.fromkeys(("uv", "open", "docker", "osascript"), STUB_RECORDER))
    for arg in ("not-a-commit", ""):
        out = _run(script, arg, root, bin_dir)
        assert out.returncode == 2, out.stdout + out.stderr
        assert "refused" in out.stdout
    assert not (bin_dir / "calls.log").exists()  # no uv step, no Docker start, no Docker quit
    assert not (root / "data").exists() and not (root / ".worktrees").exists()


def test_the_linux_guard_comes_before_docker_is_started():
    lines = LINUX.read_text().splitlines()
    guard = next(i for i, line in enumerate(lines) if line.startswith("SHA=$(git rev-parse --verify"))
    first_docker = next(i for i, line in enumerate(lines) if line.startswith(("open -a", "trap ")))
    assert guard < first_docker


def _gate(repo: tuple[Path, str], tmp_path: Path, fail_on: str) -> tuple[subprocess.CompletedProcess[str], str]:
    root, sha = repo
    bin_dir = _stubs(tmp_path, {"uv": STUB_UV})
    (bin_dir / "fail_on").write_text(fail_on)
    out = _run(MACOS, sha, root, bin_dir)
    log = (root / "data" / "logs" / f"ci-macos-{sha[:7]}.log").read_text()
    return out, log


@macos_only
def test_macos_gate_reports_a_planted_failing_pytest_step(repo, tmp_path):
    out, log = _gate(repo, tmp_path, "run pytest")
    assert out.returncode != 0, out.stdout + out.stderr
    results = [line for line in log.splitlines() if line.startswith("=== RESULT")]
    assert len(results) == 5
    assert [r.split()[2] for r in results] == ["rc=0", "rc=0", "rc=0", "rc=0", "rc=1"]
    assert log.index("=== STEP uv run pytest") < log.index("=== RESULT rc=1")
    assert "=== GATE FAIL (uv run pytest)" in log and "GATE PASS" not in log
    assert "=== GATE FAIL (uv run pytest)" in out.stdout
    assert not (repo[0] / ".worktrees" / f"ci-{repo[1][:7]}").exists()  # the worktree is removed


@macos_only
def test_macos_gate_passes_when_every_step_exits_zero(repo, tmp_path):
    out, log = _gate(repo, tmp_path, "")
    assert out.returncode == 0, out.stdout + out.stderr
    assert [line.split()[2] for line in log.splitlines() if line.startswith("=== RESULT")] == ["rc=0"] * 5
    assert "=== GATE PASS" in log and "GATE FAIL" not in log


@pytest.mark.parametrize("script", [MACOS, LINUX], ids=["macos", "linux"])
def test_the_gate_scripts_hold_no_absolute_home_path(script):
    text = script.read_text()
    assert "/Users/" not in text and "/home/" not in text
