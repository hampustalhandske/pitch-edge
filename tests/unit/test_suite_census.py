"""Unit tests of the suite censuses in tests/suite_census.py: each check is fed a planted gap and must report it."""

from __future__ import annotations

import pytest
from suite_census import (
    EXPECTED_SKIPS,
    PLUGIN_NAME,
    criteria_gaps,
    full_run_blockers,
    load_expected_skips,
    skip_problems,
)

from pitch_edge.research.gates import CRITERIA, Criterion, Grade, grade
from pitch_edge.research.ledger import Ledger

pytestmark = pytest.mark.unit


def _complete() -> set[tuple[str, str, str]]:
    return {(r, n, s) for r, names in CRITERIA.items() for n in names for s in ("pass", "fail")}


# --------------------------------------------------------------------------- ladder criteria


def test_a_complete_criteria_census_has_no_gaps():
    assert criteria_gaps(CRITERIA, _complete(), {}, {}) == []


def test_a_criterion_without_a_pass_case_is_reported():
    seen = _complete() - {("L3", "PBO < 0.25", "pass")}
    assert criteria_gaps(CRITERIA, seen, {}, {}) == ["L3 'PBO < 0.25': no test reaches pass"]


def test_a_criterion_without_a_fail_or_not_evaluated_case_is_reported():
    seen = _complete() - {("L0", "registered", "fail")}
    assert criteria_gaps(CRITERIA, seen, {}, {}) == ["L0 'registered': no test reaches fail or not-evaluated"]
    assert criteria_gaps(CRITERIA, seen | {("L0", "registered", "not-evaluated")}, {}, {}) == []
    # n/a and partial are neither side
    seen |= {("L0", "registered", "n/a"), ("L0", "registered", "partial")}
    assert criteria_gaps(CRITERIA, seen, {}, {}) == ["L0 'registered': no test reaches fail or not-evaluated"]


def test_allowlisted_criteria_are_exempt_and_stale_allowlist_entries_are_reported():
    seen = _complete() - {("L5", "paper CLV CI lower bound > 0", "pass")}
    never_pass = {("L5", "paper CLV CI lower bound > 0"): "reason"}
    assert criteria_gaps(CRITERIA, seen, never_pass, {}) == []
    assert criteria_gaps(CRITERIA, _complete(), never_pass, {}) == [
        "stale NEVER_PASS entry ('L5', 'paper CLV CI lower bound > 0'): a test reaches pass"
    ]
    assert criteria_gaps(CRITERIA, _complete(), {}, {("L9", "nothing"): "reason"}) == [
        "stale NEVER_FAIL entry ('L9', 'nothing'): no such criterion"
    ]


def test_grades_built_by_grade_are_recorded_and_hand_built_ones_are_not(request, tmp_path):
    census = request.config.pluginmanager.get_plugin(PLUGIN_NAME)
    census.seen.discard(("L0", "registered", "fail"))
    assert grade("never-registered-v1", Ledger(tmp_path / "l.duckdb")).rung == "none"
    assert ("L0", "registered", "fail") in census.seen
    Grade("x", "none", [Criterion("L9", "planted", "pass")])
    assert not any(rung == "L9" for rung, _, _ in census.seen)


# --------------------------------------------------------------------------- skip set


def test_an_unlisted_skip_is_reported_and_listed_skips_may_run():
    expected = {"tests/a.py::test_x": "needs data/", "tests/a.py::test_y": "needs EDGELAB_DIR"}
    known = {"tests/a.py::test_x", "tests/a.py::test_y", "tests/b.py::test_z"}
    assert skip_problems({"tests/a.py::test_x": "no data"}, expected, known) == []
    assert skip_problems({"tests/b.py::test_z": "planted"}, expected, known) == [
        "unlisted skip: tests/b.py::test_z (planted)"
    ]


def test_an_expected_skip_naming_no_test_is_reported():
    assert skip_problems({}, {"tests/gone.py::test_old": "r"}, {"tests/a.py::test_x"}) == [
        "stale expected skip (no such test): tests/gone.py::test_old"
    ]


def test_the_expected_skips_file_parses_and_a_line_without_a_reason_is_refused(tmp_path):
    entries = load_expected_skips(EXPECTED_SKIPS)
    assert entries and all(node.startswith("tests/") and "::" in node and why for node, why in entries.items())
    bad = tmp_path / "skips.txt"
    bad.write_text("# comment\n\ntests/a.py::test_x\n")
    with pytest.raises(ValueError, match="skips.txt:3"):
        load_expected_skips(bad)


class _Config:
    def __init__(self, source: str = "TESTPATHS", addopts: tuple[str, ...] = ("-ra", "-m", "not network"), **opts):
        self.args_source = getattr(pytest.Config.ArgsSource, source)
        self._addopts = list(addopts)
        self._opts = {"keyword": "", "markexpr": "not network", **opts}

    def getini(self, name):
        assert name == "addopts"
        return self._addopts

    def getoption(self, name, default=None):
        return self._opts.get(name, default)


def test_only_a_plain_testpaths_run_counts_as_full():
    assert full_run_blockers(_Config()) == []
    assert full_run_blockers(_Config(source="ARGS")) == ["paths given on the command line"]
    assert full_run_blockers(_Config(keyword="gates")) == ["-k given"]
    assert full_run_blockers(_Config(markexpr="unit")) == ["-m differs from the ini addopts"]
    assert full_run_blockers(_Config(markexpr="", addopts=())) == []
    assert full_run_blockers(_Config(deselect=["tests/a.py::t"])) == ["--deselect given"]
    assert full_run_blockers(_Config(lf=True, stepwise=True, collectonly=True)) == [
        "--lf given",
        "--sw given",
        "--collect-only given",
    ]
    assert full_run_blockers(_Config(override_ini=["testpaths=tests/unit/test_a.py"])) == ["-o testpaths/addopts given"]


def test_the_live_session_config_is_read_without_error(request):
    blockers = full_run_blockers(request.config)
    explicit_paths = request.config.args_source != pytest.Config.ArgsSource.TESTPATHS
    assert ("paths given on the command line" in blockers) is explicit_paths
