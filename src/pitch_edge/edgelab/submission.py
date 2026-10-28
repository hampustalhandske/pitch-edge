"""Write the `submission.json` that makes a folder an edgelab submission (edgelab contract `submission.v1`).

A submission folder holds a strategy folder, a Parquet data bundle and this manifest. The manifest's `trials`
block is built from pitch-edge's research ledger: the family's chain export is written into the folder and the
declaration carries the family, its trial count, the export's last `row_hash` and every tested
`"<hypothesis_id>|<config_hash>"` key, so edgelab can recount the search from the file alone.

    from pitch_edge.edgelab.submission import SubmissionSpec, write_submission

    spec = SubmissionSpec(hypothesis_id="h-v1", git_sha="3ad43f5", family="fam", upto=row_hash,
                          train_end=date(2025, 6, 30), test_window=(date(2025, 7, 1), date(2026, 10, 19)))
    write_submission(spec, "out/h-v1", ledger)  # the folder already holds strategy/ and data/

`declared_n` is the number of trial rows in the family up to `upto`, the same N pitch-edge's own deflated
Sharpe uses (`research/gates.py`); a re-run of one (hypothesis, config) counts again, so it is never below the
number of distinct `test_keys`, which edgelab requires. Nothing here runs or imports edgelab; `edgelab submission check <folder>`
validates the result offline.
"""

from __future__ import annotations

import dataclasses
import json
import re
from datetime import date
from pathlib import Path
from typing import Any, Literal

from pitch_edge.research.ledger import Ledger

SCHEMA_VERSION = 1
CLIENT_PROJECT = "pitch-edge"
LEDGER_EXPORT_DIR = "ledger"
"""Folder inside the submission that holds the ledger export, `<family>.jsonl`."""

_SHA = re.compile(r"[0-9a-f]{7,40}")


class SubmissionError(ValueError):
    """The spec or the folder breaks a rule of the submission contract."""


@dataclasses.dataclass(frozen=True)
class SubmissionSpec:
    """What one submission declares.

    Attributes:
        hypothesis_id: The registered pitch-edge hypothesis under test.
        git_sha: The pitch-edge commit the strategy code was taken from (7 to 40 lower-case hex characters).
        family: The ledger family whose trials are declared.
        upto: `row_hash` of the last ledger chain entry the declaration counts.
        train_end: Last date used to design or fit the strategy, or None when not stated.
        test_window: First and last day the evidence is scored on, both after `train_end`; None scores the
            strategy config's data range.
        strategy_dir: Strategy folder inside the submission folder (holds `strategy.yaml`).
        data_dir: Parquet bundle folder inside the submission folder.
        seed: Seed of every stochastic step of the run.
        profile: edgelab's sample-size profile, `"fast"` or `"full"`.
    """

    hypothesis_id: str
    git_sha: str
    family: str
    upto: str
    train_end: date | None = None
    test_window: tuple[date, date] | None = None
    strategy_dir: str = "strategy"
    data_dir: str = "data"
    seed: int = 0
    profile: Literal["fast", "full"] = "full"


def _check(spec: SubmissionSpec, folder: Path) -> None:
    """Raise `SubmissionError` for a rule `edgelab submission check` would also refuse."""
    if not _SHA.fullmatch(spec.git_sha):
        raise SubmissionError(f"git_sha {spec.git_sha!r} is not 7 to 40 lower-case hex characters.")
    if spec.seed < 0:
        raise SubmissionError("seed must be >= 0.")
    if spec.profile not in ("fast", "full"):
        raise SubmissionError(f"profile {spec.profile!r} is not 'fast' or 'full'.")
    if spec.test_window is not None:
        start, end = spec.test_window
        if not start < end:
            raise SubmissionError(f"test_window start {start} is not before its end {end}.")
        if spec.train_end is not None and not spec.train_end < start:
            raise SubmissionError(f"test_window starts {start}, not after train_end {spec.train_end}.")
    for name in (spec.strategy_dir, spec.data_dir):
        sub = (folder / name).resolve()
        if not sub.is_relative_to(folder.resolve()):
            raise SubmissionError(f"{name!r} is outside the submission folder.")
        if not sub.is_dir():
            raise SubmissionError(f"{name!r} is not a folder inside {folder}.")


def trial_declaration(lines: list[dict[str, Any]], export_path: str) -> dict[str, Any]:
    """The contract's `trials` block from `Ledger.export_chain` lines written to `export_path` (relative)."""
    summary = lines[-1]["summary"]
    if summary["n_tests"] < 1:
        raise SubmissionError(f"family {summary['family']!r} has no trials up to {summary['upto']}.")
    return {
        "family_id": summary["family"],
        "declared_n": summary["n_tests"],
        "ledger_ref": summary["upto"],
        "test_keys": summary["test_keys"],
        "ledger_export_path": export_path,
    }


def write_submission(spec: SubmissionSpec, folder: str | Path, ledger: Ledger) -> Path:
    """Write the ledger export and `submission.json` into `folder` and return the manifest's path.

    `folder` must already hold `spec.strategy_dir` and `spec.data_dir`. Re-running overwrites both files. Raises
    `SubmissionError` for a spec the contract refuses or a hypothesis not registered in `spec.family`, and
    `ProtocolError` when `spec.upto` is not a chain hash.
    """
    folder = Path(folder)
    _check(spec, folder)
    kinds = ledger.metric_kinds(spec.family)
    if spec.hypothesis_id not in kinds:
        raise SubmissionError(f"{spec.hypothesis_id!r} is not registered in family {spec.family!r}.")
    lines = ledger.export_chain(spec.family, spec.upto, kinds)
    export_rel = f"{LEDGER_EXPORT_DIR}/{spec.family}.jsonl"
    (folder / LEDGER_EXPORT_DIR).mkdir(exist_ok=True)
    (folder / export_rel).write_text("".join(json.dumps(line, sort_keys=True) + "\n" for line in lines))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": "code",
        "client": {"project": CLIENT_PROJECT, "git_sha": spec.git_sha, "hypothesis_id": spec.hypothesis_id},
        "strategy_dir": spec.strategy_dir,
        "data": {"kind": "parquet", "path": spec.data_dir},
        "trials": trial_declaration(lines, export_rel),
        "train_end": spec.train_end.isoformat() if spec.train_end else None,
        "test_window": (
            {"start": spec.test_window[0].isoformat(), "end": spec.test_window[1].isoformat()}
            if spec.test_window
            else None
        ),
        "seed": spec.seed,
        "profile": spec.profile,
    }
    path = folder / "submission.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path
