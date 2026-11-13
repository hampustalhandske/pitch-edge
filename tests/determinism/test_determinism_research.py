"""Research harness (register, `run_trial` over a grid, grade, `Ledger.export_chain`) run in separate processes.

Driver: `drivers/research.py` (inputs, output and the named volatile fields are listed in its docstring).

- Two processes with different `PYTHONHASHSEED` values produce byte-identical output.
- A run SIGKILLed right after its 2nd `Ledger.record_trial` and then resumed produces the uninterrupted output.
- Negative controls: per-decision data drawn from an unseeded generator differs between two runs, and a resume that
  skips the first unfinished unit differs from the uninterrupted run.
"""

from __future__ import annotations

import pytest
from harness import Invocation, Job, assert_detected, assert_identical, assert_killed, final, run_jobs

pytestmark = pytest.mark.integration

DRIVER = "research.py"


@pytest.fixture(scope="module")
def jobs(tmp_path_factory):
    root = tmp_path_factory.mktemp("research")
    kill = Invocation(DRIVER, 3, kill_after=2, kill_at="trial")
    return run_jobs(
        root,
        [
            Job("a", [Invocation(DRIVER, 1)]),
            Job("b", [Invocation(DRIVER, 2)]),
            Job("unseeded-a", [Invocation(DRIVER, 1, defect="unseeded")]),
            Job("unseeded-b", [Invocation(DRIVER, 2, defect="unseeded")]),
            Job("kill-resume", [kill, Invocation(DRIVER, 4)]),
            Job("kill-skip", [kill, Invocation(DRIVER, 4, defect="skip-on-resume")]),
        ],
    )


def test_two_processes_write_identical_output(jobs):
    assert_identical(final(jobs["a"]), final(jobs["b"]))
    assert '"chain_verified": true' in final(jobs["a"]).text


def test_unseeded_draw_is_detected(jobs):
    assert_detected(final(jobs["unseeded-a"]), final(jobs["unseeded-b"]))


def test_kill_and_resume_matches_the_uninterrupted_run(jobs):
    assert_killed(jobs["kill-resume"])
    assert_identical(final(jobs["a"]), final(jobs["kill-resume"]))


def test_resume_that_skips_a_unit_is_detected(jobs):
    assert_killed(jobs["kill-skip"])
    assert_detected(final(jobs["a"]), final(jobs["kill-skip"]))
