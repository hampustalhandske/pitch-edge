"""Test-side helpers of the determinism tests (engineering rules 1 and 2): run pipeline drivers in fresh
processes and compare their canonical outputs byte for byte.

A *job* is a list of driver invocations run one after another in one working directory (a kill run followed
by its resume, or a single run); `run_jobs` runs the jobs of a test module concurrently, each invocation a new
`sys.executable` process with its own `PITCH_EDGE_DATA_DIR`, `PYTHONHASHSEED` and working directory, so nothing
is shared between runs but the code and the fixed inputs. A run's output is the file its driver wrote with
`canon.write_canonical`; a run that was killed writes none.
"""

from __future__ import annotations

import difflib
import json
import os
import signal
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
DRIVERS = HERE / "drivers"
REPO_ROOT = HERE.parents[1]
SIGKILLED = -signal.SIGKILL
RUN_TIMEOUT_S = 300
# Environment variables a run never inherits: credentials and anything that points pitch-edge at real data.
_DROPPED_ENV = (
    "ANTHROPIC_API_KEY",
    "LAKEHOUSE_API_TOKEN",
    "LAKEHOUSE_API_URL",
    "ODDS_API_KEY",
    "GROQ_CLOUD_API_KEY",
    "PITCH_EDGE_ENV_FILE",
    "PITCH_EDGE_DB_PATH",
)


@dataclass(frozen=True)
class Invocation:
    """One driver run: the driver's file name in `drivers/`, the `PYTHONHASHSEED`, and its extra arguments."""

    driver: str
    hashseed: int
    defect: str = ""
    kill_after: int = 0
    kill_at: str = ""
    extra: tuple[str, ...] = ()


@dataclass
class Run:
    """What one invocation left behind: exit code, stderr tail and the output file (absent when killed)."""

    invocation: Invocation
    work: Path
    out: Path
    returncode: int
    stderr: str

    @property
    def text(self) -> str:
        if not self.out.exists():
            raise AssertionError(
                f"{self.invocation} wrote no output (exit {self.returncode}); stderr tail:\n{self.stderr[-3000:]}"
            )
        return self.out.read_text()


@dataclass
class Job:
    """Invocations run in order in one working directory."""

    name: str
    steps: list[Invocation]
    runs: list[Run] = field(default_factory=list)


def _env(work: Path, hashseed: int) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in _DROPPED_ENV}
    env.update(
        {
            "PYTHONHASHSEED": str(hashseed),
            "PITCH_EDGE_DATA_DIR": str(work / "data"),
            "PITCH_EDGE_REPORTS_DIR": str(work / "reports"),
            "PITCH_EDGE_HTTP_MIN_INTERVAL": "0",
            "PITCH_EDGE_ENV_FILE": str(work / "no.env"),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    return env


def _invoke(inv: Invocation, work: Path, index: int) -> Run:
    out = work / f"output-{index}.json"
    cmd = [sys.executable, str(DRIVERS / inv.driver), "--work", str(work), "--out", str(out)]
    if inv.defect:
        cmd += ["--defect", inv.defect]
    if inv.kill_after:
        cmd += ["--kill-after", str(inv.kill_after), "--kill-at", inv.kill_at]
    cmd += list(inv.extra)
    proc = subprocess.run(
        cmd, cwd=work, env=_env(work, inv.hashseed), capture_output=True, text=True, timeout=RUN_TIMEOUT_S
    )
    return Run(inv, work, out, proc.returncode, proc.stderr)


def _run_job(job: Job, root: Path) -> Job:
    work = root / job.name
    work.mkdir(parents=True, exist_ok=False)
    job.runs = [_invoke(inv, work, i) for i, inv in enumerate(job.steps)]
    return job


def run_jobs(root: Path, jobs: list[Job]) -> dict[str, Job]:
    """Run `jobs` concurrently (each in `root/<job name>`, its steps in order) and return them by name."""
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        done = list(pool.map(lambda j: _run_job(j, root), jobs))
    return {j.name: j for j in done}


def final(job: Job) -> Run:
    """The job's last run, which must have exited 0."""
    run = job.runs[-1]
    assert run.returncode == 0, f"{job.name}: exit {run.returncode}; stderr tail:\n{run.stderr[-3000:]}"
    return run


def assert_killed(job: Job) -> None:
    """Every run of `job` but the last died of SIGKILL and wrote no output."""
    for run in job.runs[:-1]:
        assert run.returncode == SIGKILLED, (
            f"{job.name}: expected SIGKILL, got exit {run.returncode}; stderr tail:\n{run.stderr[-3000:]}"
        )
        assert not run.out.exists(), f"{job.name}: a killed run left an output file"


def diff(a: str, b: str, limit: int = 80) -> str:
    """The first `limit` lines of a unified diff of two canonical outputs."""
    lines = list(difflib.unified_diff(a.splitlines(), b.splitlines(), "first", "second", lineterm="", n=2))
    more = f"\n... {len(lines) - limit} more diff lines" if len(lines) > limit else ""
    return "\n".join(lines[:limit]) + more


def identical(a: Run, b: Run) -> bool:
    """Whether the two runs' output files are byte-identical."""
    return a.out.read_bytes() == b.out.read_bytes()


def without(text: str, paths: tuple[tuple[str, ...], ...]) -> str:
    """The canonical output `text` with the named sections removed (each path a sequence of object keys), written
    back in the same canonical form, so the rest still compares byte for byte. Raises KeyError for a missing path."""
    obj = json.loads(text)
    for path in paths:
        node = obj
        for key in path[:-1]:
            node = node[key]
        del node[path[-1]]
    return json.dumps(obj, sort_keys=True, indent=1, allow_nan=False, ensure_ascii=True) + "\n"


def assert_identical(a: Run, b: Run, *, excluding: tuple[tuple[str, ...], ...] = ()) -> None:
    """The two runs' outputs are byte-identical (apart from the `excluding` sections); otherwise fail with their
    diff."""
    ta, tb = a.text, b.text
    if excluding:
        ta, tb = without(ta, excluding), without(tb, excluding)
        assert ta == tb, f"outputs differ outside {excluding}:\n{diff(ta, tb)}"
        return
    assert identical(a, b), f"outputs differ:\n{diff(ta, tb)}"


def assert_detected(a: Run, b: Run) -> None:
    """Negative control: the comparison tells the two runs' outputs apart (both runs finished)."""
    ta, tb = a.text, b.text
    assert ta != tb and not identical(a, b), "the planted defect left the output unchanged: the check is blind to it"
