"""Paper ledger (collect, decide, settle, report) run in separate processes.

Driver: `drivers/paper.py` (inputs, schedule, output and the named volatile fields are listed in its docstring).

- Two processes with different `PYTHONHASHSEED` values produce byte-identical output.
- A run SIGKILLed right after its first settlement write and resumed produces the uninterrupted output (every
  table of the paper ledger compared); one SIGKILLed right after its 3rd decision and resumed matches it in every
  table but `decide_log`.
- Negative controls: an unseeded report bootstrap differs between two runs, and a resume whose step checkpoint
  was written before the step ran (so the interrupted settle is skipped) differs from the uninterrupted run.
- A run whose decision ids sort in a different order (production ids are uuid4) produces the same output.
- Known production gap, an `xfail(strict=True)` that flips once fixed: a kill inside `run_decide` leaves
  `decide_log` short.
"""

from __future__ import annotations

import pytest
from harness import Invocation, Job, assert_detected, assert_identical, assert_killed, final, run_jobs

pytestmark = pytest.mark.integration

DRIVER = "paper.py"


@pytest.fixture(scope="module")
def jobs(tmp_path_factory):
    root = tmp_path_factory.mktemp("paper")
    kill_settle = Invocation(DRIVER, 3, kill_after=1, kill_at="settle")
    return run_jobs(
        root,
        [
            Job("a", [Invocation(DRIVER, 1)]),
            Job("b", [Invocation(DRIVER, 2)]),
            Job("unseeded-a", [Invocation(DRIVER, 1, defect="unseeded")]),
            Job("unseeded-b", [Invocation(DRIVER, 2, defect="unseeded")]),
            Job("salted-ids", [Invocation(DRIVER, 1, defect="salted-ids")]),
            Job("kill-settle", [kill_settle, Invocation(DRIVER, 4)]),
            Job("kill-decision", [Invocation(DRIVER, 5, kill_after=3, kill_at="decision"), Invocation(DRIVER, 6)]),
            Job(
                "kill-checkpoint-first",
                [
                    Invocation(DRIVER, 3, defect="checkpoint-first", kill_after=1, kill_at="settle"),
                    Invocation(DRIVER, 4, defect="checkpoint-first"),
                ],
            ),
        ],
    )


def test_two_processes_write_identical_output(jobs):
    assert_identical(final(jobs["a"]), final(jobs["b"]))
    text = final(jobs["a"]).text
    assert '"verified": true' in text and "lake:fd_matches.betfair_ex_close" in text


def test_report_does_not_depend_on_decision_id_order(jobs):
    assert_identical(final(jobs["a"]), final(jobs["salted-ids"]))


def test_unseeded_draw_is_detected(jobs):
    assert_detected(final(jobs["unseeded-a"]), final(jobs["unseeded-b"]))


@pytest.mark.parametrize("job", ["kill-settle", "kill-decision"])
def test_kill_and_resume_matches_the_uninterrupted_run(jobs, job):
    """Every table but `decide_log` (see the next test) matches the uninterrupted run."""
    assert_killed(jobs[job])
    assert_identical(final(jobs["a"]), final(jobs[job]), excluding=(("tables", "decide_log"),))


def test_kill_after_a_settlement_leaves_the_whole_ledger_as_uninterrupted(jobs):
    assert_identical(final(jobs["a"]), final(jobs["kill-settle"]))


@pytest.mark.xfail(
    strict=True,
    reason=(
        "run_decide (PE-47; src/pitch_edge/paper/strategies.py append_decision, record_decide_run) commits each "
        "decision on its own and the strategy's decide_log row after them: a kill in between leaves decisions "
        "without their decide_log row, and the re-run cycle logs n_new_decisions for its own run only (3 of 6)"
    ),
)
def test_kill_inside_decide_leaves_the_whole_ledger_as_uninterrupted(jobs):
    assert_identical(final(jobs["a"]), final(jobs["kill-decision"]))


def test_resume_that_skips_the_interrupted_step_is_detected(jobs):
    assert_killed(jobs["kill-checkpoint-first"])
    assert_detected(final(jobs["a"]), final(jobs["kill-checkpoint-first"]))
