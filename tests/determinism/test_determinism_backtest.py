"""Feature build and walk-forward backtests (Dixon-Coles, GBDT) on the synthetic league run in separate processes.

Driver: `drivers/backtest.py` (inputs and output are listed in its docstring).

- Two processes with different `PYTHONHASHSEED` values produce byte-identical features, predictions, bets,
  summaries and calibration blocks.
- Negative control: Dixon-Coles probabilities with an unseeded jitter differ between two runs.
- No kill-and-resume test: a backtest keeps no checkpoint; it runs in memory from the features to the result, and
  a re-run starts from scratch.
"""

from __future__ import annotations

import pytest
from harness import Invocation, Job, assert_detected, assert_identical, final, run_jobs

pytestmark = pytest.mark.integration

DRIVER = "backtest.py"


@pytest.fixture(scope="module")
def jobs(tmp_path_factory):
    root = tmp_path_factory.mktemp("backtest")
    return run_jobs(
        root,
        [
            Job("a", [Invocation(DRIVER, 1)]),
            Job("b", [Invocation(DRIVER, 2)]),
            Job("unseeded-a", [Invocation(DRIVER, 1, defect="unseeded")]),
            Job("unseeded-b", [Invocation(DRIVER, 2, defect="unseeded")]),
        ],
    )


def test_two_processes_write_identical_output(jobs):
    assert_identical(final(jobs["a"]), final(jobs["b"]))
    text = final(jobs["a"]).text
    assert '"dixon_coles' in text or '"dc' in text
    assert '"gbdt"' in text and '"kelly_quarter"' in text


def test_unseeded_draw_is_detected(jobs):
    assert_detected(final(jobs["unseeded-a"]), final(jobs["unseeded-b"]))
