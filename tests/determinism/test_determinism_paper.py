"""Paper ledger (collect, decide, settle, report) run in separate processes.

Driver: `drivers/paper.py` (inputs, schedule, output and the named volatile fields are listed in its docstring).

- Two processes with different `PYTHONHASHSEED` values produce byte-identical output.
- A run SIGKILLed right after its first settlement write and resumed produces the uninterrupted output (every
  table of the paper ledger compared); one SIGKILLed right after its 3rd decision and resumed matches it in every
  table but `decide_log`.
- Negative controls: an unseeded report bootstrap differs between two runs, and a resume whose step checkpoint
  was written before the step ran (so the interrupted settle is skipped) differs from the uninterrupted run.
- Known production nondeterminism, each an `xfail(strict=True)` that flips once fixed: the report follows the
  order of the decision ids (uuid4 in production), and a kill inside `run_decide` leaves `decide_log` short.
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


@pytest.mark.xfail(
    strict=True,
    reason=(
        "PaperStore.decisions() (src/pitch_edge/paper/store.py:560) orders rows of one decided_at by decision_id, "
        "a uuid4 in production, and report.summarise sums clv_log in that row order (src/pitch_edge/paper/"
        "report.py:205 mean_clv; :194 feeds the bootstrap in the same order), so mean_clv changes in its last bit "
        "with the ids (here -0.024363950077665786 vs -0.02436395007766579)"
    ),
)
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
        "run_decide (src/pitch_edge/paper/strategies.py:480 append_decision, :486 record_decide_run) commits each "
        "decision on its own and the strategy's decide_log row after them: a kill in between leaves decisions "
        "without their decide_log row, and the re-run cycle logs n_new_decisions for its own run only (3 of 6)"
    ),
)
def test_kill_inside_decide_leaves_the_whole_ledger_as_uninterrupted(jobs):
    assert_identical(final(jobs["a"]), final(jobs["kill-decision"]))


def test_resume_that_skips_the_interrupted_step_is_detected(jobs):
    assert_killed(jobs["kill-checkpoint-first"])
    assert_detected(final(jobs["a"]), final(jobs["kill-checkpoint-first"]))
