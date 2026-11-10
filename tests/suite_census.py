"""Suite-level censuses run at the end of a pytest session (registered by `tests/conftest.py`).

1. Ladder-criteria census. Every `Grade` built by `pitch_edge.research.gates.grade` during the session is
   recorded. When every test of `tests/unit/test_research_gates.py` ran (the whole file was selected and each of
   its tests reached its call phase), every criterion in `gates.CRITERIA` must have been reached as `pass` and as
   `fail` or `not-evaluated` by some test. A criterion that by design can never reach one of those sides is listed
   in `NEVER_PASS` / `NEVER_FAIL` with the reason; an entry there that is reached anyway, or that names no
   criterion, is reported as stale.

2. Pinned skip set. On a full run (see `full_run_blockers`), every skipped test (or module skipped at
   collection) must be listed in `tests/expected_skips.txt`, one `<node id> | <reason>` per line. A listed test
   may run instead of skipping (the skips depend on the machine: data/, EDGELAB_DIR, optional packages, macOS).
   A listed node id that the session does not know at all (not collected, not deselected, not skipped at
   collection) is reported as stale. Expected `xfail`s are not skips.

Either census turns a passing session into a failed one (exit status 1) and prints what is missing.
"""

from __future__ import annotations

import shlex
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

GATES_TEST_FILE = "tests/unit/test_research_gates.py"
EXPECTED_SKIPS = Path(__file__).with_name("expected_skips.txt")
PLUGIN_NAME = "pitch-edge-suite-census"

# (rung, criterion) -> why no ledger state can make it `pass`.
NEVER_PASS: dict[tuple[str, str], str] = {}
# (rung, criterion) -> why no ledger state can make it `fail` or `not-evaluated`.
NEVER_FAIL: dict[tuple[str, str], str] = {}

FAIL_SIDE = frozenset({"fail", "not-evaluated"})


# --------------------------------------------------------------------------- 1. ladder criteria


def criteria_gaps(
    criteria: Mapping[str, Sequence[str]],
    seen: Iterable[tuple[str, str, str]],
    never_pass: Mapping[tuple[str, str], str] = NEVER_PASS,
    never_fail: Mapping[tuple[str, str], str] = NEVER_FAIL,
) -> list[str]:
    """Problems of a criteria census, one message each (empty when complete).

    Takes the ladder (`rung -> criterion names`), the `(rung, criterion, status)` triples the session produced
    and the two allowlists. Reports each criterion never seen as `pass` or never seen as `fail`/`not-evaluated`
    (unless allowlisted on that side), and each allowlist entry that is stale.
    """
    statuses: dict[tuple[str, str], set[str]] = {}
    for rung, name, status in seen:
        statuses.setdefault((rung, name), set()).add(status)
    known = {(rung, name) for rung, names in criteria.items() for name in names}
    problems: list[str] = []
    for rung, names in criteria.items():
        for name in names:
            got = statuses.get((rung, name), set())
            if "pass" not in got and (rung, name) not in never_pass:
                problems.append(f"{rung} '{name}': no test reaches pass")
            if not got & FAIL_SIDE and (rung, name) not in never_fail:
                problems.append(f"{rung} '{name}': no test reaches fail or not-evaluated")
    for side, allow, reached in (("pass", never_pass, {"pass"}), ("fail", never_fail, FAIL_SIDE)):
        for key in allow:
            if key not in known:
                problems.append(f"stale NEVER_{side.upper()} entry {key}: no such criterion")
            elif statuses.get(key, set()) & reached:
                problems.append(f"stale NEVER_{side.upper()} entry {key}: a test reaches {side}")
    return problems


# --------------------------------------------------------------------------- 2. skip set


def load_expected_skips(path: Path = EXPECTED_SKIPS) -> dict[str, str]:
    """Node id -> reason from an expected-skips file; `#` comments and blank lines are ignored.

    Raises ValueError for a line without a ` | <reason>` part.
    """
    out: dict[str, str] = {}
    for n, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        node, sep, reason = (part.strip() for part in line.partition(" | "))
        if not sep or not node or not reason:
            raise ValueError(f"{path.name}:{n}: expected '<node id> | <reason>', got {raw!r}")
        out[node] = reason
    return out


def skip_problems(skipped: Mapping[str, str], expected: Mapping[str, str], known: set[str]) -> list[str]:
    """Problems of a skip census, one message each (empty when clean).

    Takes the skips the session saw (`node id -> skip message`), the expected skips (`node id -> reason`) and
    every node id the session knows. Reports each unlisted skip and each listed node id the session does not know.
    """
    problems = [f"unlisted skip: {node} ({why})" for node, why in skipped.items() if node not in expected]
    problems += [f"stale expected skip (no such test): {node}" for node in expected if node not in known]
    return problems


def _default_markexpr(config: Any) -> str:
    """The `-m` expression the ini `addopts` sets (the last one wins), or "" when it sets none."""
    args = list(config.getini("addopts"))
    if isinstance(args, str):  # pragma: no cover - getini returns a list for `addopts`
        args = shlex.split(args)
    expr = ""
    for i, arg in enumerate(args):
        if arg == "-m" and i + 1 < len(args):
            expr = args[i + 1]
        elif arg.startswith("-m") and len(arg) > 2:
            expr = arg[2:]
    return expr


def full_run_blockers(config: Any) -> list[str]:
    """Why this session is not a full run (empty when it is).

    A full run takes its paths from the ini `testpaths` (no path or node id on the command line), keeps the ini
    `-m` expression, and has no `-k`, `--deselect`, `--lf`, `--sw`, `--collect-only` or `-o` override of
    `testpaths`/`addopts`. Deselection by the ini
    `-m` expression (the `network` tests) is part of a full run.
    """
    out: list[str] = []
    if config.args_source != pytest.Config.ArgsSource.TESTPATHS:
        out.append("paths given on the command line")
    if config.getoption("keyword", ""):
        out.append("-k given")
    if config.getoption("markexpr", "") != _default_markexpr(config):
        out.append("-m differs from the ini addopts")
    if config.getoption("deselect", None):
        out.append("--deselect given")
    if config.getoption("lf", False):
        out.append("--lf given")
    if config.getoption("stepwise", False):
        out.append("--sw given")
    if config.getoption("collectonly", False):
        out.append("--collect-only given")
    overrides = config.getoption("override_ini", None) or []
    if any(o.split("=", 1)[0].strip() in ("testpaths", "addopts") for o in overrides):
        out.append("-o testpaths/addopts given")
    return out


# --------------------------------------------------------------------------- the plugin


class SuiteCensus:
    """Pytest plugin object: records grades, skips and the gates file's tests, and runs both censuses."""

    def __init__(self, config: pytest.Config):
        self.config = config
        self.seen: set[tuple[str, str, str]] = set()
        self.skipped: dict[str, str] = {}
        self.known: set[str] = set()
        self.gate_tests: set[str] = set()
        self.ran: set[str] = set()
        self.problems: list[str] = []
        self._restore: Any = None
        self._record_grades()

    def _record_grades(self) -> None:
        """Wrap `Grade.__init__` so each `Grade` built inside `gates.grade` adds its criteria to `self.seen`."""
        from pitch_edge.research import gates

        original = gates.Grade.__init__
        grade_code = gates.grade.__code__
        seen = self.seen

        def recording_init(grade_self: Any, *args: Any, **kwargs: Any) -> None:
            original(grade_self, *args, **kwargs)
            if sys._getframe(1).f_code is grade_code:
                seen.update((c.rung, c.name, c.status) for c in grade_self.criteria)

        gates.Grade.__init__ = recording_init  # type: ignore[method-assign]
        self._restore = (gates.Grade, original)

    def pytest_unconfigure(self) -> None:
        if self._restore is not None:
            cls, original = self._restore
            cls.__init__ = original
            self._restore = None

    def pytest_collectreport(self, report: pytest.CollectReport) -> None:
        if report.skipped:
            self.skipped[report.nodeid] = _skip_message(report)
            self.known.add(report.nodeid)
        if report.nodeid.split("::", 1)[0] == GATES_TEST_FILE:
            self.gate_tests.update(x.nodeid for x in report.result if isinstance(x, pytest.Item))

    def pytest_deselected(self, items: Sequence[pytest.Item]) -> None:
        self.known.update(item.nodeid for item in items)

    def pytest_collection_finish(self, session: pytest.Session) -> None:
        self.known.update(item.nodeid for item in session.items)

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        if report.skipped and not hasattr(report, "wasxfail"):
            self.skipped[report.nodeid] = _skip_message(report)
        if report.when == "call":
            self.ran.add(report.nodeid)

    def pytest_sessionfinish(self, session: pytest.Session, exitstatus: int) -> None:
        if exitstatus not in (pytest.ExitCode.OK, pytest.ExitCode.TESTS_FAILED):
            return
        problems: list[str] = []
        if self.gate_tests and self.gate_tests <= self.ran:
            from pitch_edge.research.gates import CRITERIA

            problems += [f"criteria census: {p}" for p in criteria_gaps(CRITERIA, self.seen)]
        if not full_run_blockers(self.config):
            expected = load_expected_skips()
            problems += [f"skip census: {p}" for p in skip_problems(self.skipped, expected, self.known)]
        self.problems = problems
        if problems and session.exitstatus == pytest.ExitCode.OK:
            session.exitstatus = pytest.ExitCode.TESTS_FAILED

    def pytest_terminal_summary(self, terminalreporter: Any) -> None:
        if not self.problems:
            return
        terminalreporter.write_sep("=", "suite census FAILED", red=True)
        for problem in self.problems:
            terminalreporter.write_line(problem)


def _skip_message(report: pytest.TestReport | pytest.CollectReport) -> str:
    longrepr = report.longrepr
    if isinstance(longrepr, tuple) and len(longrepr) == 3:
        return str(longrepr[2]).removeprefix("Skipped: ")
    return str(longrepr)
