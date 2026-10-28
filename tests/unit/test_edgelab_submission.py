"""The submission writer turns a folder holding a strategy and a data bundle into an edgelab submission: the
manifest's trial declaration comes from the family's ledger export written beside it, and specs the contract
refuses are refused here first."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import date
from pathlib import Path

import duckdb
import pytest
from research_fixtures import native_record, write_registration
from test_edgelab_export import _inputs, _spec

from pitch_edge.edgelab import export_prediction_market_dataset, scaffold_strategy
from pitch_edge.edgelab.submission import SubmissionError, SubmissionSpec, write_submission
from pitch_edge.research.errors import ProtocolError
from pitch_edge.research.ledger import Ledger
from pitch_edge.research.registry import load

pytestmark = pytest.mark.unit

SHA = "3ad43f5"


@pytest.fixture
def ledger(tmp_path):
    reg = load(write_registration(tmp_path, "fam-a-test-v1", family="fam-a", grid={"k": [1, 2]}))
    led = Ledger(tmp_path / "ledger.duckdb")
    led.register(reg)
    led.record_trial(native_record(reg, {"k": 1}))
    led.record_trial(native_record(reg, {"k": 2}))
    led.record_trial(native_record(reg, {"k": 2}, metric_value=0.06))
    return led


def _last_hash(led: Ledger) -> str:
    con = duckdb.connect(str(led.path), read_only=True)
    try:
        return con.execute("SELECT row_hash FROM ledger_chain ORDER BY seq DESC LIMIT 1").fetchone()[0]
    finally:
        con.close()


def _folder(tmp_path: Path) -> Path:
    folder = tmp_path / "sub"
    (folder / "strategy").mkdir(parents=True)
    (folder / "data").mkdir()
    return folder


def _submission(led: Ledger, **kw) -> SubmissionSpec:
    base = {
        "hypothesis_id": "fam-a-test-v1",
        "git_sha": SHA,
        "family": "fam-a",
        "upto": _last_hash(led),
        "train_end": date(2025, 6, 30),
        "test_window": (date(2025, 7, 1), date(2026, 10, 19)),
    }
    base.update(kw)
    return SubmissionSpec(**base)


def test_manifest_declares_the_family_count_from_the_export(ledger, tmp_path):
    folder = _folder(tmp_path)
    path = write_submission(_submission(ledger), folder, ledger)
    manifest = json.loads(path.read_text())
    trials = manifest["trials"]
    assert trials["family_id"] == "fam-a" and trials["declared_n"] == 3
    assert trials["ledger_ref"] == _last_hash(ledger) and len(trials["test_keys"]) == 2
    export = [json.loads(x) for x in (folder / trials["ledger_export_path"]).read_text().splitlines()]
    assert export[-1]["summary"]["test_keys"] == trials["test_keys"]
    assert manifest["client"] == {"project": "pitch-edge", "git_sha": SHA, "hypothesis_id": "fam-a-test-v1"}
    assert manifest["train_end"] == "2025-06-30"
    assert manifest["test_window"] == {"start": "2025-07-01", "end": "2026-10-19"}
    assert manifest["data"] == {"kind": "parquet", "path": "data"} and manifest["kind"] == "code"


def test_unstated_windows_are_null(ledger, tmp_path):
    path = write_submission(_submission(ledger, train_end=None, test_window=None), _folder(tmp_path), ledger)
    manifest = json.loads(path.read_text())
    assert manifest["train_end"] is None and manifest["test_window"] is None


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"git_sha": "HEAD"}, "git_sha"),
        ({"test_window": (date(2025, 7, 1), date(2025, 7, 1))}, "not before its end"),
        ({"train_end": date(2025, 7, 1)}, "not after train_end"),
        ({"data_dir": "missing"}, "not a folder"),
        ({"strategy_dir": "../outside"}, "outside the submission folder"),
        ({"hypothesis_id": "other-v1"}, "not registered in family"),
        ({"seed": -1}, "seed"),
    ],
)
def test_specs_the_contract_refuses_are_refused(ledger, tmp_path, override, match):
    (tmp_path / "outside").mkdir()
    with pytest.raises(SubmissionError, match=match):
        write_submission(_submission(ledger, **override), _folder(tmp_path), ledger)


def test_unknown_upto_is_refused(ledger, tmp_path):
    with pytest.raises(ProtocolError, match="not a row_hash"):
        write_submission(_submission(ledger, upto="f" * 64), _folder(tmp_path), ledger)


EDGELAB_BIN = Path(os.environ.get("EDGELAB_DIR", "/nonexistent")) / ".venv/bin/edgelab"


@pytest.mark.integration
@pytest.mark.skipif(not EDGELAB_BIN.exists(), reason="EDGELAB_DIR unset or its venv missing")
def test_edgelab_check_accepts_the_submission(ledger, tmp_path):
    folder = tmp_path / "sub"
    s, r, f = _inputs()
    export_prediction_market_dataset(s, r, f, folder / "data")
    strategy = scaffold_strategy(_spec(data_path="../data", range_start="2025-02-01", range_end="2025-03-31"), folder)
    spec = _submission(
        ledger,
        strategy_dir=strategy.name,
        train_end=date(2025, 2, 27),
        test_window=(date(2025, 2, 28), date(2025, 3, 31)),
    )
    write_submission(spec, folder, ledger)
    arch = shutil.which("arch")

    def edgelab(*args: str) -> subprocess.CompletedProcess[str]:
        cmd = [str(EDGELAB_BIN), *args]
        return subprocess.run([arch, "-arm64", *cmd] if arch else cmd, capture_output=True, text=True, timeout=300)

    grading = edgelab("strategy", "init", strategy.name, "--root", str(folder))  # appends edgelab's policy block
    assert grading.returncode == 0, grading.stdout + grading.stderr
    result = json.loads(edgelab("submission", "check", str(folder), "--json").stdout)
    assert result["ok"], result["errors"]
