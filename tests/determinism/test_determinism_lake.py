"""Lake mirror (`run_lake_sync`) and football-data spine (`build_football_data_spine`) run in separate processes
against a fake lake, into a fresh warehouse per run.

Driver: `drivers/lake.py` (inputs, schedule, output and the named volatile fields are listed in its docstring).

- Two processes with different `PYTHONHASHSEED` values produce byte-identical warehouses.
- A run SIGKILLed right after its 2nd committed mirror window, and one SIGKILLed right after its 1st spine
  window, each resumed, leave every table but the `pipeline_runs` health log as the uninterrupted run did.
- Negative controls: a mirror that keeps a random row per key (no total order) differs between two runs, and a
  killed unit that wrote its checkpoint without its rows is skipped on resume and detected.
- Known production gap, an `xfail(strict=True)` that flips once fixed: after a kill, `pipeline_runs` holds no row
  for the killed call and the resuming call logs only the rows it wrote itself.
"""

from __future__ import annotations

import pytest
from harness import Invocation, Job, assert_detected, assert_identical, assert_killed, final, run_jobs

pytestmark = pytest.mark.integration

DRIVER = "lake.py"
KILLS = {"kill-mirror": (2, "mirror-window"), "kill-spine": (1, "spine-window")}
HEALTH_LOG = ("tables", "pipeline_runs")


@pytest.fixture(scope="module")
def jobs(tmp_path_factory):
    root = tmp_path_factory.mktemp("lake")
    killed = [
        Job(name, [Invocation(DRIVER, 3, kill_after=n, kill_at=at), Invocation(DRIVER, 4)])
        for name, (n, at) in KILLS.items()
    ]
    skipped = [
        Job(
            f"{name}-checkpoint-first",
            [Invocation(DRIVER, 3, defect="checkpoint-first", kill_after=n, kill_at=at), Invocation(DRIVER, 4)],
        )
        for name, (n, at) in KILLS.items()
    ]
    return run_jobs(
        root,
        [
            Job("a", [Invocation(DRIVER, 1)]),
            Job("b", [Invocation(DRIVER, 2)]),
            Job("tiebreak-a", [Invocation(DRIVER, 1, defect="random-tiebreak")]),
            Job("tiebreak-b", [Invocation(DRIVER, 2, defect="random-tiebreak")]),
            *killed,
            *skipped,
        ],
    )


def test_two_processes_write_identical_output(jobs):
    assert_identical(final(jobs["a"]), final(jobs["b"]))
    text = final(jobs["a"]).text
    assert '"fdx_SWE_20210403_MalmoFF_AIK"' in text and '"fd_E0_2122_20210814_Arsenal_Chelsea"' in text
    # the tied duplicate ("Ref Z" sorts last) and the older scrape (9 goals) are never the row kept
    assert '"Ref Z"' not in text and '"ft_home_goals": 9' not in text


def test_row_pick_without_a_total_order_is_detected(jobs):
    assert_detected(final(jobs["tiebreak-a"]), final(jobs["tiebreak-b"]))


@pytest.mark.parametrize("job", list(KILLS))
def test_kill_and_resume_matches_the_uninterrupted_run(jobs, job):
    assert_killed(jobs[job])
    assert_identical(final(jobs["a"]), final(jobs[job]), excluding=(HEALTH_LOG,))


@pytest.mark.xfail(
    strict=True,
    reason=(
        "data/ingest.py:90-94 `_run` logs one pipeline_runs row after the call returns, with the rows written by "
        "that call (lake/mirror.py:476 sync_mirror, lake/spine.py:280 build_fd_spine count only this call's windows): "
        "a killed call logs nothing and the resuming call logs its remainder (42 of 126 mirror rows, 84 of 126 "
        "spine matches)"
    ),
)
@pytest.mark.parametrize("job", list(KILLS))
def test_kill_and_resume_leaves_the_health_log_as_uninterrupted(jobs, job):
    assert_identical(final(jobs["a"]), final(jobs[job]))


@pytest.mark.parametrize("job", list(KILLS))
def test_resume_that_skips_a_checkpointed_but_unwritten_unit_is_detected(jobs, job):
    assert_killed(jobs[f"{job}-checkpoint-first"])
    assert_detected(final(jobs["a"]), final(jobs[f"{job}-checkpoint-first"]))
