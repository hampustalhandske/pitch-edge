"""`pitch-edge research ...`: register, status, grade and ledger against a temporary ledger file."""

from __future__ import annotations

import pytest
from research_fixtures import per_decision, write_registration
from typer.testing import CliRunner

from pitch_edge.cli import app
from pitch_edge.research.ledger import Ledger
from pitch_edge.research.registry import load
from pitch_edge.research.trial import run_trial

pytestmark = pytest.mark.unit

runner = CliRunner()
HID = "sharp-close-test-v1"


def invoke(*args: str):
    res = runner.invoke(app, ["research", *args], catch_exceptions=False)
    return res


def test_register_status_grade_and_ledger(tmp_path):
    ledger_path = str(tmp_path / "ledger.duckdb")
    reg_path = write_registration(tmp_path)

    res = invoke("register", str(reg_path), "--ledger", ledger_path)
    assert res.exit_code == 0, res.output
    assert "registered" in res.output and HID in res.output
    res = invoke("register", str(reg_path), "--ledger", ledger_path)
    assert res.exit_code == 0 and "already registered" in res.output

    reg = load(reg_path)
    run_trial(reg, {}, per_decision(100, 4, 0.05), "edge_bits", Ledger(ledger_path), "snap", {"clean": True})

    res = invoke("status", "--ledger", ledger_path)
    assert res.exit_code == 0, res.output
    assert HID in res.output and "L2" in res.output

    res = invoke("status", "--family", "nothing-here", "--ledger", ledger_path)
    assert res.exit_code == 0 and HID not in res.output

    res = invoke("grade", HID, "--ledger", ledger_path)
    assert res.exit_code == 0, res.output
    assert "L2" in res.output and "BH q < 0.10" in res.output and "not-evaluated" in res.output

    res = invoke("ledger", "--id", HID, "--ledger", ledger_path)
    assert res.exit_code == 0, res.output
    assert "edge_bits" in res.output and "snap" in res.output


def test_register_refuses_an_edited_file_with_exit_code(tmp_path):
    ledger_path = str(tmp_path / "ledger.duckdb")
    reg_path = write_registration(tmp_path)
    assert invoke("register", str(reg_path), "--ledger", ledger_path).exit_code == 0
    reg_path.write_text(reg_path.read_text() + "\n# edited\n")
    res = invoke("register", str(reg_path), "--ledger", ledger_path)
    assert res.exit_code == 1
    assert "bump the version" in res.output


def test_register_reports_an_invalid_file(tmp_path):
    bad = tmp_path / "bad-v1.yaml"
    bad.write_text("id: bad-v1\nfamily: x\n")
    res = invoke("register", str(bad), "--ledger", str(tmp_path / "l.duckdb"))
    assert res.exit_code == 1
    assert "invalid registration" in res.output


def test_grade_of_unknown_id_is_none(tmp_path):
    res = invoke("grade", "nope-v1", "--ledger", str(tmp_path / "l.duckdb"))
    assert res.exit_code == 0
    assert "none" in res.output


def test_read_commands_never_create_a_ledger(tmp_path):
    path = tmp_path / "nothing" / "ledger.duckdb"
    for args in (("status",), ("grade", HID), ("ledger", "--id", HID)):
        res = invoke(*args, "--ledger", str(path))
        assert res.exit_code == 0, res.output
    assert not path.exists() and not path.parent.exists()


def test_import_trials_and_review_commands(tmp_path):
    ledger_path = str(tmp_path / "ledger.duckdb")
    reg_path = write_registration(tmp_path, grid={"theta": [0.02]}, primary_metric="clv_log")
    assert invoke("register", str(reg_path), "--ledger", ledger_path).exit_code == 0
    csv = tmp_path / "trials.csv"
    csv.write_text(
        "id,config_hash,config,registration_sha256,git_sha,data_snapshot,run_at,window,n,n_days,metric,value,ci_lo,ci_hi,p_one_sided,direction\n"
        f'{HID},x,"{{""theta"": 0.02}}",{load(reg_path).source_sha256},abc,snap,2026-10-20T22:00:00+00:00,evaluation,400,90,clv_log,0.01,0.0,0.02,0.04,positive\n'
    )
    res = invoke("import-trials", str(csv), "--id", HID, "--ledger", ledger_path)
    assert res.exit_code == 0, res.output
    assert "imported 1" in res.output
    res = invoke("import-trials", str(csv), "--id", HID, "--ledger", ledger_path)
    assert "imported 0" in res.output and "1 already in the ledger" in res.output

    res = invoke(
        "review",
        HID,
        "--kind",
        "audit",
        "--verdict",
        "SIGNED",
        "--detail",
        '{"scope": {"theta": 0.02}}',
        "--ledger",
        ledger_path,
    )
    assert res.exit_code == 0, res.output
    assert Ledger(ledger_path).reviews(HID).iloc[0]["verdict"] == "SIGNED"
    res = invoke("review", HID, "--kind", "vibes", "--verdict", "x", "--ledger", ledger_path)
    assert res.exit_code == 1


def test_import_trials_alias_option(tmp_path):
    ledger_path = str(tmp_path / "ledger.duckdb")
    reg_path = write_registration(tmp_path, primary_metric="mean CLV per bet (log)")
    assert invoke("register", str(reg_path), "--ledger", ledger_path).exit_code == 0
    csv = tmp_path / "trials.csv"
    csv.write_text(
        "id,config_hash,config,registration_sha256,git_sha,data_snapshot,run_at,window,n,n_days,metric,value,ci_lo,ci_hi,p_one_sided,direction\n"
        f"{HID},x,{{}},{load(reg_path).source_sha256},abc,snap,2026-10-20T22:00:00+00:00,evaluation,400,90,clv_mean,0.01,0.0,0.02,0.04,positive\n"
    )
    res = invoke(
        "import-trials", str(csv), "--id", HID, "--alias", "clv_mean=mean CLV per bet (log)", "--ledger", ledger_path
    )
    assert res.exit_code == 0, res.output
    assert "['clv_mean']" in res.output
    res = invoke("import-trials", str(csv), "--id", HID, "--alias", "no-equals-sign", "--ledger", ledger_path)
    assert res.exit_code == 1
