"""`run_trial`: one bootstrap-scored, provenance-stamped ledger row per registered grid point."""

from __future__ import annotations

import json

import numpy as np
import pytest
from research_fixtures import per_decision, write_registration

from pitch_edge.research.ledger import Ledger, ProtocolError
from pitch_edge.research.registry import config_hash, load
from pitch_edge.research.stats import block_bootstrap
from pitch_edge.research.trial import TrialResult, run_trial

pytestmark = pytest.mark.unit


@pytest.fixture
def setup(tmp_path):
    ledger = Ledger(tmp_path / "ledger.duckdb")
    reg = load(write_registration(tmp_path, grid={"threshold": [0.5, 0.6]}))
    ledger.register(reg)
    return ledger, reg


def test_run_trial_scores_and_writes_one_row(setup):
    ledger, reg = setup
    data = per_decision(100, 4, mu=0.05)
    res = run_trial(reg, {"threshold": 0.5}, data, "edge_bits", ledger, "snap-1", {"clean": True}, n_boot=2000, seed=3)
    assert isinstance(res, TrialResult)
    expected = block_bootstrap(data["value"], data["block"], n_boot=2000, seed=3)
    assert res.metric_value == pytest.approx(expected.estimate)
    assert (res.ci_lo, res.ci_hi) == pytest.approx((expected.lo, expected.hi))
    assert res.p_value == pytest.approx(expected.p_gt0)  # registered direction is positive
    assert res.n == 400 and res.n_blocks == 100
    assert res.config_hash == config_hash({"threshold": 0.5})
    assert res.git_sha and res.git_sha != ""

    rows = ledger.trials(hypothesis_id=reg.id)
    assert len(rows) == 1
    row = rows.iloc[0]
    assert row["trial_id"] == res.trial_id
    assert row["n"] == 400 and row["metric_name"] == "edge_bits" and row["data_snapshot"] == "snap-1"
    assert row["direction"] == "positive" and row["family"] == "sharp-close"
    extra = json.loads(row["extra_json"])
    assert extra["clean"] is True
    assert extra["n_boot"] == 2000 and extra["seed"] == 3 and extra["n_blocks"] == 100
    assert extra["registration_sha256"] == reg.source_sha256
    assert extra["is_primary_metric"] is True
    assert extra["alpha"] == 0.05 and extra["window"] == "evaluation"
    assert extra["reproduced"] is True  # run_trial re-ran the bootstrap and got identical numbers
    assert len(extra["code_hash"]) == 64
    # Per-block series: [label, sum of values, decisions], blocks in chronological order.
    assert len(extra["block_series"]) == 100
    label, total, count = extra["block_series"][0]
    day = data[data["block"] == "2022-01-01"]["value"]
    assert label == "2022-01-01" and total == pytest.approx(day.sum()) and count == 4
    assert len(extra["subperiod_estimates"]) == 3
    # Sharpe-type moments are of the per-match-day sums (the strategy's daily P&L), not per decision.
    sums = data.groupby("block")["value"].sum()
    assert extra["sr"] == pytest.approx(sums.mean() / sums.std(ddof=1))
    assert extra["sr_n"] == 100
    assert extra["kurt"] == pytest.approx(sums.kurt() + 3.0)  # stored non-excess
    assert extra["skew"] == pytest.approx(sums.skew())


def test_rerun_with_same_seed_reproduces_and_appends(setup):
    ledger, reg = setup
    data = per_decision(60, 5, mu=0.0)
    a = run_trial(reg, {"threshold": 0.6}, data, "edge_bits", ledger, "snap", n_boot=500, seed=1)
    b = run_trial(reg, {"threshold": 0.6}, data, "edge_bits", ledger, "snap", n_boot=500, seed=1)
    assert (a.metric_value, a.ci_lo, a.ci_hi, a.p_value) == (b.metric_value, b.ci_lo, b.ci_hi, b.p_value)
    assert a.trial_id != b.trial_id
    assert len(ledger.trials(hypothesis_id=reg.id)) == 2


def test_p_value_follows_registered_direction(tmp_path):
    ledger = Ledger(tmp_path / "ledger.duckdb")
    data = per_decision(80, 4, mu=-0.05)
    neg = load(write_registration(tmp_path, "neg-test-v1", direction="negative"))
    two = load(write_registration(tmp_path, "two-test-v1", direction="two-sided"))
    ledger.register(neg)
    ledger.register(two)
    boot = block_bootstrap(data["value"], data["block"], n_boot=1000, seed=0)
    r_neg = run_trial(neg, {}, data, "edge_bits", ledger, "s", n_boot=1000, seed=0)
    r_two = run_trial(two, {}, data, "edge_bits", ledger, "s", n_boot=1000, seed=0)
    assert r_neg.p_value == pytest.approx(boot.p_lt0)
    assert r_two.p_value == pytest.approx(min(1.0, 2 * min(boot.p_gt0, boot.p_lt0)))


def test_refuses_when_registration_file_was_edited(setup):
    ledger, reg = setup
    reg.source_path.write_text(reg.source_path.read_text().replace("min_n: 300", "min_n: 10"))
    with pytest.raises(ProtocolError, match="edited"):
        run_trial(reg, {"threshold": 0.5}, per_decision(10, 2, 0.0), "edge_bits", ledger, "s", n_boot=100)
    with pytest.raises(ProtocolError, match="edited"):
        run_trial(
            load(reg.source_path), {"threshold": 0.5}, per_decision(10, 2, 0.0), "edge_bits", ledger, "s", n_boot=100
        )
    assert len(ledger.trials()) == 0


def test_refuses_unregistered_hypothesis(tmp_path):
    ledger = Ledger(tmp_path / "ledger.duckdb")
    reg = load(write_registration(tmp_path))
    with pytest.raises(ProtocolError, match="not registered"):
        run_trial(reg, {}, per_decision(10, 2, 0.0), "edge_bits", ledger, "s", n_boot=100)


def test_refuses_config_outside_the_registered_grid(setup):
    ledger, reg = setup
    with pytest.raises(ProtocolError, match="grid"):
        run_trial(reg, {"threshold": 0.55}, per_decision(10, 2, 0.0), "edge_bits", ledger, "s", n_boot=100)


def test_refuses_unregistered_metric(setup):
    ledger, reg = setup
    with pytest.raises(ProtocolError, match="metric"):
        run_trial(reg, {"threshold": 0.5}, per_decision(10, 2, 0.0), "sharpe_after_hindsight", ledger, "s", n_boot=100)
    # A registered secondary metric is fine, and is flagged as not primary.
    res = run_trial(reg, {"threshold": 0.5}, per_decision(10, 2, 0.0), "roi", ledger, "s", n_boot=100)
    assert res.extra["is_primary_metric"] is False


def test_validates_the_per_decision_frame(setup):
    ledger, reg = setup
    good = per_decision(10, 2, 0.0)
    with pytest.raises(ValueError, match="columns"):
        run_trial(reg, {"threshold": 0.5}, good.drop(columns="block"), "edge_bits", ledger, "s", n_boot=100)
    dup = good.copy()
    dup.loc[1, "decision_id"] = dup.loc[0, "decision_id"]
    with pytest.raises(ValueError, match="duplicate"):
        run_trial(reg, {"threshold": 0.5}, dup, "edge_bits", ledger, "s", n_boot=100)
    bad = good.copy()
    bad.loc[0, "value"] = np.nan
    with pytest.raises(ValueError):
        run_trial(reg, {"threshold": 0.5}, bad, "edge_bits", ledger, "s", n_boot=100)
    with pytest.raises(ValueError, match="data_snapshot"):
        run_trial(reg, {"threshold": 0.5}, good, "edge_bits", ledger, "", n_boot=100)
    assert len(ledger.trials()) == 0


def test_leave_one_league_out_is_computed_with_block_bootstrap_cis(setup):
    ledger, reg = setup
    data = per_decision(60, 6, 0.05, leagues=("E0", "D1", "SP1"))
    res = run_trial(reg, {"threshold": 0.5}, data, "edge_bits", ledger, "s", n_boot=500, seed=2)
    loo = res.extra["leave_one_league_out"]
    assert set(loo) == {"E0", "D1", "SP1"}
    rest = data[data["league"] != "D1"]
    expected = block_bootstrap(rest["value"], rest["block"], n_boot=500, seed=2)
    assert loo["D1"] == pytest.approx({"est": expected.estimate, "lo": expected.lo, "hi": expected.hi, "n": len(rest)})
    assert res.extra["n_leagues"] == 3


def test_one_block_is_refused_and_nothing_is_written(setup):
    ledger, reg = setup
    data = per_decision(10, 5, 0.01)
    data["block"] = "all"
    with pytest.raises(ValueError, match="at least 2 blocks"):
        run_trial(reg, {"threshold": 0.5}, data, "edge_bits", ledger, "s", n_boot=100)
    assert ledger.trials().empty


def test_per_grid_point_direction_is_used(tmp_path):
    ledger = Ledger(tmp_path / "ledger.duckdb")
    reg = load(
        write_registration(
            tmp_path,
            direction="per segment (see grid)",
            grid={"segment": [{"name": "fav", "direction": "negative"}, {"name": "long", "direction": "positive"}]},
        )
    )
    ledger.register(reg)
    data = per_decision(50, 4, -0.05)
    fav = run_trial(reg, reg.expand_grid()[0], data, "edge_bits", ledger, "s", n_boot=500)
    assert fav.direction == "negative" and fav.p_value == fav.p_lt0


def test_code_is_archived_when_the_tree_is_dirty(setup, tmp_path, monkeypatch):
    ledger, reg = setup
    study = tmp_path / "study.py"
    study.write_text("print('study v1')\n")
    monkeypatch.setattr("pitch_edge.research.trial.git_state", lambda: ("abc123", True))
    res = run_trial(
        reg, {"threshold": 0.5}, per_decision(20, 3, 0.1), "edge_bits", ledger, "s", n_boot=200, code_files=[study]
    )
    code_hash = res.extra["code_hash"]
    archive = ledger.path.parent / reg.id / "code" / code_hash
    assert (archive / "study.py").read_text() == "print('study v1')\n"
    assert (archive / "src" / "pitch_edge" / "research" / "trial.py").exists()
    assert res.extra["code_files"][0].endswith("study.py")
    study.write_text("print('study v2')\n")
    res2 = run_trial(
        reg, {"threshold": 0.5}, per_decision(20, 3, 0.1), "edge_bits", ledger, "s", n_boot=200, code_files=[study]
    )
    assert res2.extra["code_hash"] != code_hash


def test_no_archive_when_the_tree_is_clean(setup, monkeypatch):
    ledger, reg = setup
    monkeypatch.setattr("pitch_edge.research.trial.git_state", lambda: ("abc123", False))
    res = run_trial(reg, {"threshold": 0.5}, per_decision(20, 3, 0.1), "edge_bits", ledger, "s", n_boot=200)
    assert not (ledger.path.parent / reg.id / "code").exists()
    assert res.extra["git_dirty"] is False and len(res.extra["code_hash"]) == 64


def test_unknown_window_is_refused(setup):
    ledger, reg = setup
    with pytest.raises(ProtocolError, match="window"):
        run_trial(
            reg, {"threshold": 0.5}, per_decision(20, 3, 0.1), "edge_bits", ledger, "s", n_boot=200, window="hindsight"
        )
    run_trial(
        reg, {"threshold": 0.5}, per_decision(20, 3, 0.1), "edge_bits", ledger, "s", n_boot=200, window="exploratory"
    )


def test_caller_cannot_override_computed_provenance(setup):
    ledger, reg = setup
    res = run_trial(
        reg,
        {"threshold": 0.5},
        per_decision(20, 3, 0.1),
        "edge_bits",
        ledger,
        "s",
        {
            "n_boot": 999999,
            "p_gt0": 0.0,
            "subperiod_estimates": [1.0, 2.0, 3.0],
            "sr": 9.0,
            "kurt": 3.0,
            "alpha": 0.5,
            "n_blocks": 10_000,
            "within_ceiling": True,
        },
        n_boot=200,
    )
    assert res.extra["n_boot"] == 200
    assert res.extra["p_gt0"] != 0.0
    # Computed statistics win; the caller's variants are kept beside them, never used by the gates.
    assert res.extra["subperiod_estimates"] != [1.0, 2.0, 3.0]
    assert res.extra["caller_subperiod_estimates"] == [1.0, 2.0, 3.0]
    assert res.extra["caller_n_boot"] == 999999
    assert res.extra["within_ceiling"] is True
    assert res.extra["sr"] != 9.0 and res.extra["caller_sr"] == 9.0
    assert res.extra["alpha"] == 0.05 and res.extra["n_blocks"] == 20
