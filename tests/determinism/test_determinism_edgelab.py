"""edgelab dataset export and submission manifest run in separate processes.

Driver: `drivers/edgelab.py` (inputs and output are listed in its docstring); its ledger input is written once by
`drivers/research.py` and shared read-only by every run.

- Two processes with different `PYTHONHASHSEED` values write byte-identical submission folders (every file's
  sha256 compared).
- Negative control: forecasts with an unseeded jitter differ between two runs.
- No kill-and-resume test: the export and the submission have no resume concept; each run writes every file of
  the folder from scratch, and a re-run overwrites them.
"""

from __future__ import annotations

import pytest
from harness import Invocation, Job, assert_detected, assert_identical, final, run_jobs

pytestmark = pytest.mark.integration

DRIVER = "edgelab.py"


@pytest.fixture(scope="module")
def jobs(tmp_path_factory):
    root = tmp_path_factory.mktemp("edgelab")
    inputs = run_jobs(root, [Job("inputs", [Invocation("research.py", 0)])])["inputs"]
    final(inputs)
    extra = ("--inputs", str(inputs.runs[-1].work))
    return run_jobs(
        root,
        [
            Job("a", [Invocation(DRIVER, 1, extra=extra)]),
            Job("b", [Invocation(DRIVER, 2, extra=extra)]),
            Job("unseeded-a", [Invocation(DRIVER, 1, defect="unseeded", extra=extra)]),
            Job("unseeded-b", [Invocation(DRIVER, 2, defect="unseeded", extra=extra)]),
        ],
    )


def test_two_processes_write_identical_files(jobs):
    assert_identical(final(jobs["a"]), final(jobs["b"]))
    text = final(jobs["a"]).text
    for name in ("submission.json", "ledger/determinism-family.jsonl", "data/features.parquet"):
        assert f'"{name}"' in text


def test_unseeded_draw_is_detected(jobs):
    assert_detected(final(jobs["unseeded-a"]), final(jobs["unseeded-b"]))
