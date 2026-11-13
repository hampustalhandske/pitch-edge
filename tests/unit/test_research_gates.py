"""The L0-L5 evidence ladder (`research/STANDARDS.md`) evaluated on synthetic ledgers, one rung at a
time — including the ways a careless or motivated study could try to game each rung."""

from __future__ import annotations

import json
import shutil

import duckdb
import numpy as np
import pandas as pd
import pytest
from research_fixtures import native_record, per_decision, write_registration

from pitch_edge.research.gates import Grade, grade, results_by_role
from pitch_edge.research.ledger import Ledger, TrialRecord
from pitch_edge.research.registry import config_hash, load
from pitch_edge.research.stats import block_bootstrap, deflated_sharpe_ratio, expected_max_sharpe
from pitch_edge.research.trial import run_trial

pytestmark = pytest.mark.unit

HID = "sharp-close-test-v1"
L3_EXTRA = {"clean": True, "within_ceiling": True}
CLV = {"primary_metric": "clv_log", "secondary_metrics": []}


def status(g: Grade, rung: str, name: str) -> str:
    return criterion(g, rung, name).status


def criterion(g: Grade, rung: str, name: str):
    matches = [c for c in g.criteria if c.rung == rung and c.name == name]
    assert matches, f"no criterion {rung}/{name} in {[(c.rung, c.name) for c in g.criteria]}"
    return matches[0]


def subdir(tmp_path, name: str):
    d = tmp_path / name
    d.mkdir()
    return d


def make(tmp_path, hypothesis_id: str = HID, **reg_kwargs):
    ledger = Ledger(tmp_path / "ledger.duckdb")
    reg = load(write_registration(tmp_path, hypothesis_id, **reg_kwargs))
    ledger.register(reg)
    return ledger, reg


def strong_l3(tmp_path, **reg_kwargs):
    ledger, reg = make(tmp_path, **reg_kwargs)
    run_trial(reg, {}, per_decision(100, 4, 0.08, seed=1), reg.primary_metric, ledger, "snap", L3_EXTRA, seed=0)
    return ledger, reg


def two_phase(
    n_blocks: int, per_block: int, early: float, late: float, sd: float = 0.05, seed: int = 0
) -> pd.DataFrame:
    """Mean `early` in the first third of the blocks and `late` afterwards."""
    df = per_decision(n_blocks, per_block, 0.0, sd=sd, seed=seed)
    first = df["block"].isin(sorted(df["block"].unique())[: n_blocks // 3])
    df["value"] += np.where(first, early, late)
    return df


# --------------------------------------------------------------------------- L0


def test_unregistered_hypothesis_has_no_rung(tmp_path):
    g = grade("nope-v1", Ledger(tmp_path / "l.duckdb"))
    assert g.rung == "none"
    assert status(g, "L0", "registered") == "fail"
    assert all(c.status == "not-evaluated" for c in g.criteria if c.rung != "L0")


def test_registered_without_trials_is_l0(tmp_path):
    ledger, _ = make(tmp_path)
    g = grade(HID, ledger)
    assert g.rung == "L0"
    assert status(g, "L0", "file unchanged since registration") == "pass"
    assert status(g, "L0", "ledger hash chain intact") == "pass"
    assert all(c.status == "not-evaluated" for c in g.criteria if c.rung == "L1")


def test_edited_registration_file_fails_l0(tmp_path):
    ledger, reg = make(tmp_path)
    reg.source_path.write_text(reg.source_path.read_text() + "\n# edited\n")
    g = grade(HID, ledger)
    assert g.rung == "none"
    assert status(g, "L0", "file unchanged since registration") == "fail"


def test_tampered_ledger_fails_l0(tmp_path):
    ledger, _ = strong_l3(tmp_path)
    assert grade(HID, ledger).rung == "L3"
    con = duckdb.connect(str(ledger.path))
    con.execute("UPDATE trials SET ci_lo = ci_lo + 0.001")
    con.close()
    g = grade(HID, ledger)
    assert status(g, "L0", "ledger hash chain intact") == "fail"
    assert g.rung == "none"


def test_imported_trial_older_than_the_declared_registration_fails_l0(tmp_path):
    ledger, reg = make(tmp_path)  # declared registered_at 2026-10-20T21:00Z
    extra = {"imported": True, "window": "evaluation", "is_primary_metric": True}
    ledger.record_trial(
        native_record(reg, extra_json=json.dumps(extra), run_at=pd.Timestamp("2026-10-20T20:42:00").to_pydatetime())
    )
    g = grade(HID, ledger)
    assert status(g, "L0", "registered before first trial") == "fail"
    assert g.rung == "none"


# --------------------------------------------------------------------------- L1


def test_missing_clean_flag_stops_at_l0(tmp_path):
    ledger, reg = make(tmp_path)
    run_trial(reg, {}, per_decision(100, 4, 0.05), "edge_bits", ledger, "snap")
    g = grade(HID, ledger)
    assert g.rung == "L0"
    assert status(g, "L1", "clean (point-in-time + reproducible)") == "not-evaluated"


def test_clean_flag_false_fails_l1(tmp_path):
    ledger, reg = make(tmp_path)
    run_trial(reg, {}, per_decision(100, 4, 0.05), "edge_bits", ledger, "snap", {"clean": False})
    assert grade(HID, ledger).rung == "L0"


def test_rerunning_until_it_passes_does_not_work(tmp_path):
    # C1: the first run of a grid point is its result of record; later runs are reproducibility checks.
    ledger, reg = make(tmp_path, grid={"k": [1, 2, 3]})
    first = run_trial(reg, {"k": 1}, per_decision(100, 4, 0.0, seed=1000), "edge_bits", ledger, "snap0", L3_EXTRA)
    assert first.ci_lo < 0
    for s in range(1, 400):
        r = run_trial(
            reg, {"k": 1}, per_decision(100, 4, 0.0, seed=1000 + s), "edge_bits", ledger, f"snap{s}", L3_EXTRA
        )
        if r.ci_lo > 0 and r.p_value < 0.01:
            break
    assert r.ci_lo > 0, "precondition: found a lucky rerun"
    g = grade(HID, ledger)
    assert g.trial_id == first.trial_id
    assert status(g, "L1", "reruns reproduce") == "fail"
    assert g.rung == "L0"


def test_identical_rerun_and_internal_reproduction_pass(tmp_path):
    ledger, reg = make(tmp_path)
    data = per_decision(100, 4, 0.05)
    run_trial(reg, {}, data, "edge_bits", ledger, "snap", {"clean": True})
    assert status(grade(HID, ledger), "L1", "reruns reproduce") == "pass"  # run_trial's own re-run
    run_trial(reg, {}, data, "edge_bits", ledger, "snap", {"clean": True})
    assert criterion(grade(HID, ledger), "L1", "reruns reproduce").value == "2 runs"


def test_reproduction_is_not_evaluated_without_any_evidence(tmp_path):
    ledger, reg = make(tmp_path)
    ledger.record_trial(native_record(reg))  # written directly: no `reproduced` flag, no rerun
    assert status(grade(HID, ledger), "L1", "reruns reproduce") == "not-evaluated"


def test_provenance_clean_tree_or_archived_code(tmp_path, monkeypatch):
    ledger, reg = make(tmp_path)
    monkeypatch.setattr("pitch_edge.research.trial.git_state", lambda: ("abc123", False))
    run_trial(reg, {}, per_decision(100, 4, 0.05), "edge_bits", ledger, "snap", {"clean": True})
    assert status(grade(HID, ledger), "L1", "provenance recorded") == "pass"

    ledger2, reg2 = make(subdir(tmp_path, "dirty"))
    monkeypatch.setattr("pitch_edge.research.trial.git_state", lambda: ("abc123", True))
    res = run_trial(reg2, {}, per_decision(100, 4, 0.05), "edge_bits", ledger2, "snap", {"clean": True})
    assert status(grade(HID, ledger2), "L1", "provenance recorded") == "pass"
    shutil.rmtree(ledger2.path.parent / HID / "code" / res.extra["code_hash"])
    assert status(grade(HID, ledger2), "L1", "provenance recorded") == "fail"

    ledger3, reg3 = make(subdir(tmp_path, "nogit"))
    ledger3.record_trial(native_record(reg3, git_sha="unknown"))
    assert status(grade(HID, ledger3), "L1", "provenance recorded") == "fail"


# --------------------------------------------------------------------------- L1 by reproduction


def _boot_record(reg, data, seed=0, **over):
    """A ledger row with exactly the numbers run_trial would compute on `data`, but none of its
    provenance: what a pre-harness ("legacy") native run looks like."""
    b = block_bootstrap(data["value"].to_numpy(), data["block"].to_numpy(), n_boot=2000, seed=seed)
    extra = {"p_gt0": b.p_gt0, "p_lt0": b.p_lt0, "n_blocks": b.n_blocks, "n_boot": 2000, "seed": seed, "alpha": 0.05}
    extra |= over.pop("extra", {})
    return native_record(
        reg, n=b.n, metric_value=b.estimate, ci_lo=b.lo, ci_hi=b.hi, git_sha="6d29de4-dirty", extra=extra, **over
    )


def _imported(reg, config, **over):
    extra = {"imported": True, "window": "evaluation", "is_primary_metric": True, "original_config": config,
             "off_grid": False} | over.pop("extra", {})  # fmt: skip
    return native_record(reg, config, extra_json=json.dumps(extra), **over)


def test_a_legacy_record_gets_provenance_from_an_exact_native_rerun(tmp_path):
    ledger, reg = make(tmp_path)
    data = per_decision(100, 4, 0.08, seed=1)
    legacy = _boot_record(reg, data, extra={"clean": True, "registration_sha256": reg.source_sha256})
    ledger.record_trial(legacy)
    assert status(grade(HID, ledger), "L1", "provenance recorded") == "fail"  # before the rerun
    rerun = run_trial(reg, {}, data, "edge_bits", ledger, "regrade", L3_EXTRA)
    g = grade(HID, ledger)
    assert status(g, "L1", "provenance recorded") == "pass"
    assert status(g, "L1", "reruns reproduce") == "pass"
    assert g.record_trial_id == legacy.trial_id and g.trial_id == rerun.trial_id
    assert g.rung == "L3", g.table().to_string()


def test_a_legacy_record_is_not_rescued_by_a_rerun_that_differs(tmp_path):
    ledger, reg = make(tmp_path)
    data = per_decision(100, 4, 0.08, seed=1)
    ledger.record_trial(_boot_record(reg, data, extra={"clean": True, "registration_sha256": reg.source_sha256}))
    run_trial(reg, {}, data.iloc[:-4], "edge_bits", ledger, "regrade", L3_EXTRA)  # one match-day fewer
    g = grade(HID, ledger)
    assert status(g, "L1", "reruns reproduce") == "fail"
    assert g.rung == "L0"


def test_an_imported_record_reproduced_natively_is_graded_with_the_native_statistics(tmp_path):
    ledger, reg = make(tmp_path, grid={"theta": [0.0, 0.02]}, **CLV)
    data = per_decision(100, 4, 0.08, seed=1)
    b = block_bootstrap(data["value"].to_numpy(), data["block"].to_numpy(), n_boot=2000, seed=0)
    est6 = float(f"{b.estimate:.6g}")  # the study wrote 6 significant figures
    imp = _imported(reg, {"theta": 0.0}, n=b.n, metric_value=est6, ci_lo=est6 - 0.5, ci_hi=est6 + 0.5, p_value=0.0,
                    metric_name="clv_log_mean")  # fmt: skip
    ledger.record_trial(imp)
    assert grade(HID, ledger).rung == "L0"  # imported alone: no clean flag, no provenance
    rerun = run_trial(reg, {"theta": 0.0}, data, "clv_log", ledger, "regrade", L3_EXTRA)
    g = grade(HID, ledger)
    assert g.record_trial_id == imp.trial_id and g.trial_id == rerun.trial_id
    assert status(g, "L1", "reruns reproduce") == "pass"
    ci = criterion(g, "L2", "CI excludes 0 in registered direction").value
    assert f"{rerun.ci_lo:.6g}" in ci  # the native CI, not the study's
    assert g.rung in ("L2", "L3")


def test_an_imported_record_with_a_different_estimate_is_not_reproduced(tmp_path):
    ledger, reg = make(tmp_path, grid={"theta": [0.0, 0.02]}, **CLV)
    data = per_decision(100, 4, 0.08, seed=1)
    b = block_bootstrap(data["value"].to_numpy(), data["block"].to_numpy(), n_boot=2000, seed=0)
    ledger.record_trial(_imported(reg, {"theta": 0.0}, n=b.n, metric_value=b.estimate * 1.001, ci_lo=0.0, ci_hi=0.2,
                                  metric_name="clv_log_mean"))  # fmt: skip
    run_trial(reg, {"theta": 0.0}, data, "clv_log", ledger, "regrade", L3_EXTRA)
    g = grade(HID, ledger)
    assert status(g, "L1", "reruns reproduce") == "fail"
    assert g.rung == "L0"


def test_the_reproduced_imported_variant_is_the_grid_points_record(tmp_path):
    # Two study rows map onto one grid point (e.g. markets ou25 and ah when only theta is a grid axis); the
    # native rerun reproduces one of them, which identifies it as the registered grid point.
    ledger, reg = make(tmp_path, grid={"theta": [0.0, 0.02]}, **CLV)
    data = per_decision(100, 4, 0.08, seed=1)
    b = block_bootstrap(data["value"].to_numpy(), data["block"].to_numpy(), n_boot=2000, seed=0)
    ah = _imported(reg, {"theta": 0.0}, n=777, metric_value=-0.02, ci_lo=-0.03, ci_hi=-0.01, p_value=1.0,
                   extra={"original_config": {"theta": 0.0, "market": "ah"}})  # fmt: skip
    ou = _imported(reg, {"theta": 0.0}, n=b.n, metric_value=b.estimate, ci_lo=0.0, ci_hi=0.2, p_value=0.0,
                   extra={"original_config": {"theta": 0.0, "market": "ou25"}})  # fmt: skip
    ledger.record_trial(ah)
    ledger.record_trial(ou)
    run_trial(reg, {"theta": 0.0}, data, "clv_log", ledger, "regrade", L3_EXTRA)
    g = grade(HID, ledger)
    assert g.record_trial_id == ou.trial_id
    assert status(g, "L1", "reruns reproduce") == "pass"
    bh = criterion(g, "L3", "BH q < 0.10 across family") if g.rung in ("L2", "L3") else None
    assert bh is not None and "m=3" in bh.detail  # theta=0 (ou25 via rerun), the ah variant, unrun theta=0.02


def test_off_grid_imported_rows_are_never_graded_but_count_in_m(tmp_path):
    ledger, reg = make(tmp_path, grid={"theta": [0.0]}, **CLV)
    off = {"source": "max_open", "theta": 0.0}
    ledger.record_trial(_imported(reg, {"theta": 0.0}, config_json=json.dumps(off), config_hash=config_hash(off),
                                  p_value=0.0, extra={"off_grid": True, "original_config": off}))  # fmt: skip
    g = grade(HID, ledger)
    assert g.trial_id is None and g.rung == "L0"
    run_trial(reg, {"theta": 0.0}, per_decision(100, 4, 0.08, seed=1), "clv_log", ledger, "s", L3_EXTRA)
    assert "m=2" in criterion(grade(HID, ledger), "L3", "BH q < 0.10 across family").detail


# --------------------------------------------------------------------------- L2


def test_null_effect_stops_at_l1(tmp_path):
    ledger, reg = make(tmp_path)
    run_trial(reg, {}, per_decision(100, 4, 0.0, seed=1), "edge_bits", ledger, "snap", {"clean": True})
    g = grade(HID, ledger)
    assert g.rung == "L1"
    assert status(g, "L2", "CI excludes 0 in registered direction") == "fail"
    assert all(c.status == "not-evaluated" for c in g.criteria if c.rung in ("L3", "L4", "L5"))


def test_fewer_than_300_decisions_fail_l2_even_if_min_n_is_lower(tmp_path):
    ledger, reg = make(tmp_path, min_n=100)
    run_trial(reg, {}, per_decision(50, 4, 0.05), "edge_bits", ledger, "snap", {"clean": True})
    g = grade(HID, ledger)
    assert status(g, "L2", "CI excludes 0 in registered direction") == "pass"
    assert status(g, "L2", "decisions >= max(min_n, 300)") == "fail"
    assert g.rung == "L1"


def test_too_few_bootstrap_draws_fail_l2(tmp_path):
    ledger, reg = make(tmp_path)
    run_trial(reg, {}, per_decision(100, 4, 0.05), "edge_bits", ledger, "snap", {"clean": True}, n_boot=500)
    assert status(grade(HID, ledger), "L2", "bootstrap draws >= 2000") == "fail"


def test_too_few_match_day_blocks_fail_l2(tmp_path):
    # C2: three huge blocks give a CI that only reflects three draws' worth of information.
    ledger, reg = make(tmp_path)
    run_trial(reg, {}, per_decision(10, 40, 0.05), "edge_bits", ledger, "snap", {"clean": True})
    g = grade(HID, ledger)
    c = criterion(g, "L2", "blocks >= 30")
    assert c.status == "fail" and c.value == 10
    assert g.rung == "L1"


def test_a_wider_alpha_cannot_buy_a_ci_that_excludes_zero(tmp_path):
    # I1: a 50% "CI" excludes zero far more easily than the 95% one the standard asks for.
    ledger, reg = make(tmp_path)
    run_trial(reg, {}, per_decision(150, 4, 0.006, seed=5), "edge_bits", ledger, "s", {"clean": True}, alpha=0.5)
    g = grade(HID, ledger)
    assert status(g, "L2", "alpha <= 0.05") == "fail"
    assert g.rung == "L1"


def test_strong_signal_without_l3_evidence_is_l2(tmp_path):
    ledger, reg = make(tmp_path)
    run_trial(reg, {}, per_decision(100, 4, 0.05), "edge_bits", ledger, "snap", {"clean": True})
    g = grade(HID, ledger)
    assert g.rung == "L2"
    assert status(g, "L3", "within literature ceiling") == "not-evaluated"


def test_negative_direction_needs_a_negative_ci(tmp_path):
    ledger, reg = make(tmp_path, direction="negative")
    run_trial(reg, {}, per_decision(100, 4, -0.05), "edge_bits", ledger, "snap", L3_EXTRA)
    assert grade(HID, ledger).rung == "L3"
    ledger2, reg2 = make(subdir(tmp_path, "pos"), direction="negative")
    run_trial(reg2, {}, per_decision(100, 4, 0.05), "edge_bits", ledger2, "snap", {"clean": True})
    assert grade(HID, ledger2).rung == "L1"


def test_only_the_evaluation_window_is_graded(tmp_path):
    # A registered secondary segment never becomes the graded headline, however strong it is.
    ledger, reg = make(tmp_path, extra_windows={"post_segment": ["2024-01-01", "2024-12-31"]})
    ev = run_trial(reg, {}, per_decision(100, 4, 0.0, seed=1), "edge_bits", ledger, "s", L3_EXTRA)
    run_trial(reg, {}, per_decision(40, 4, 0.2, seed=2), "edge_bits", ledger, "s", L3_EXTRA, window="post_segment")
    g = grade(HID, ledger)
    assert g.trial_id == ev.trial_id
    assert g.rung == "L1"
    gradable, secondary = results_by_role(HID, ledger)
    assert list(gradable["trial_id"]) == [ev.trial_id]
    assert list(secondary["window"]) == ["post_segment"]


def test_a_secondary_segment_alone_is_not_gradable(tmp_path):
    ledger, reg = make(tmp_path, extra_windows={"post_segment": ["2024-01-01", "2024-12-31"]})
    run_trial(reg, {}, per_decision(100, 4, 0.2), "edge_bits", ledger, "s", L3_EXTRA, window="post_segment")
    g = grade(HID, ledger)
    assert g.rung == "L0" and g.trial_id is None
    assert "evaluation window" in criterion(g, "L1", "clean (point-in-time + reproducible)").detail


def test_secondary_segments_still_count_in_family_bh(tmp_path):
    ledger, reg = make(tmp_path, extra_windows={"post_segment": ["2024-01-01", "2024-12-31"]})
    run_trial(reg, {}, per_decision(100, 4, 0.08), "edge_bits", ledger, "s", L3_EXTRA)
    run_trial(reg, {}, per_decision(100, 4, 0.0, seed=5), "edge_bits", ledger, "s", L3_EXTRA, window="post_segment")
    assert "m=2" in criterion(grade(HID, ledger), "L3", "BH q < 0.10 across family").detail


# --------------------------------------------------------------------------- L3 (a) BH


def test_full_l3(tmp_path):
    ledger, _ = strong_l3(tmp_path)
    g = grade(HID, ledger)
    assert g.rung == "L3", g.table().to_string()
    assert status(g, "L3", "BH q < 0.10 across family") == "pass"
    assert status(g, "L3", "deflated Sharpe >= 0.95") == "n/a"  # edge_bits: probability kind
    assert status(g, "L3", "PBO < 0.25") == "n/a"  # no grid
    assert status(g, "L3", "sub-period sign stability") == "pass"
    assert status(g, "L3", "not carried by one league") == "pass"
    assert status(g, "L4", "skeptic-auditor signed") == "not-evaluated"


def test_bh_counts_unrun_grid_points_and_registered_but_unrun_hypotheses(tmp_path):
    ledger, reg = make(tmp_path, grid={"k": [1, 2, 3]})
    res = run_trial(reg, {"k": 1}, per_decision(100, 4, 0.02, seed=3), "edge_bits", ledger, "snap", L3_EXTRA)
    assert 0.003 < res.p_value < 0.01 and res.ci_lo > 0  # precondition: marginal but L2-clearing
    c = criterion(grade(HID, ledger), "L3", "BH q < 0.10 across family")
    assert c.status == "pass" and "m=3" in c.detail  # the two unrun grid points count with p = 1
    assert c.value == pytest.approx(res.p_value * 3)

    for i in range(20):  # twenty registered ideas in the same family that were never run
        ledger.register(load(write_registration(tmp_path, f"sharp-close-idea{i}-v1")))
    c = criterion(grade(HID, ledger), "L3", "BH q < 0.10 across family")
    assert c.status == "fail" and "m=23" in c.detail
    assert grade(HID, ledger).rung == "L2"


def test_bh_ignores_other_families(tmp_path):
    ledger, reg = make(tmp_path)
    run_trial(reg, {}, per_decision(100, 4, 0.02, seed=3), "edge_bits", ledger, "snap", L3_EXTRA)
    for i in range(20):
        ledger.register(load(write_registration(tmp_path, f"elsewhere-idea{i}-v1", family="elsewhere")))
    assert "m=1" in criterion(grade(HID, ledger), "L3", "BH q < 0.10 across family").detail


def test_exploratory_runs_count_in_m_but_never_as_discoveries(tmp_path):
    ledger, reg = make(tmp_path)
    run_trial(reg, {}, per_decision(100, 4, 0.02, seed=3), "edge_bits", ledger, "snap", L3_EXTRA)
    run_trial(reg, {}, per_decision(100, 4, 0.2), "edge_bits", ledger, "snap", {"clean": True}, window="exploratory")
    g = grade(HID, ledger)
    assert json.loads(ledger.trials().iloc[0]["extra_json"])["window"] == "evaluation"
    c = criterion(g, "L3", "BH q < 0.10 across family")
    assert "m=2" in c.detail
    assert g.trial_id == ledger.trials().iloc[0]["trial_id"]  # the exploratory run is never graded


# --------------------------------------------------------------------------- L3 (b) DSR


def _family_sr(ledger: Ledger, kind_ids: set[str]) -> list[float]:
    t = ledger.trials()
    return [json.loads(e)["sr"] for h, e in zip(t["hypothesis_id"], t["extra_json"], strict=True) if h in kind_ids]


def test_dsr_uses_every_registered_test_and_same_kind_sr_variance(tmp_path):
    ledger, reg = make(tmp_path, primary_metric="clv_log", secondary_metrics=[])
    run_trial(reg, {}, per_decision(100, 4, 0.10, seed=1), "clv_log", ledger, "snap", L3_EXTRA)
    returns = {HID}
    for i in range(3):
        other = load(
            write_registration(tmp_path, f"sharp-close-other{i}-v1", primary_metric="clv_log", secondary_metrics=[])
        )
        ledger.register(other)
        returns.add(other.id)
        run_trial(other, {}, per_decision(100, 4, 0.01 * i, seed=10 + i), "clv_log", ledger, "snap", n_boot=200)
    # A probability-kind study in the same family with a wild Sharpe: counts in N, not in var(SR).
    prob = load(
        write_registration(tmp_path, "sharp-close-prob-v1", primary_metric="mean_prob_gap", secondary_metrics=[])
    )
    ledger.register(prob)
    run_trial(prob, {}, per_decision(100, 4, 1.0, sd=0.01, seed=99), "mean_prob_gap", ledger, "snap", n_boot=200)
    ledger.register(
        load(write_registration(tmp_path, "sharp-close-unrun-v1", primary_metric="clv_log", secondary_metrics=[]))
    )

    g = grade(HID, ledger)
    c = criterion(g, "L3", "deflated Sharpe >= 0.95")
    mine = json.loads(ledger.trials(hypothesis_id=HID).iloc[0]["extra_json"])
    expected = deflated_sharpe_ratio(
        mine["sr"],
        mine["sr_n"],
        mine["skew"],
        mine["kurt"],
        n_trials=6,
        var_trial_sr=float(np.var(_family_sr(ledger, returns), ddof=1)),
    )
    assert c.value == pytest.approx(expected)
    assert "N=6" in c.detail
    assert c.status == ("pass" if expected >= 0.95 else "fail")


def test_dsr_alone_equals_psr_against_zero(tmp_path):
    ledger, _ = strong_l3(tmp_path, primary_metric="clv_log", secondary_metrics=[])
    c = criterion(grade(HID, ledger), "L3", "deflated Sharpe >= 0.95")
    assert c.status == "pass" and "N=1" in c.detail


def test_dsr_not_evaluated_when_the_sr_variance_cannot_be_estimated(tmp_path):
    # I4: two registered grid points but only one run: N = 2, one SR -> the variance is unknown.
    ledger, reg = make(tmp_path, primary_metric="clv_log", secondary_metrics=[], grid={"k": [1, 2]})
    run_trial(reg, {"k": 1}, per_decision(100, 4, 0.10), "clv_log", ledger, "snap", L3_EXTRA)
    g = grade(HID, ledger)
    assert status(g, "L3", "deflated Sharpe >= 0.95") == "not-evaluated"
    assert g.rung == "L2"


def test_dsr_deflates_a_modest_sharpe_in_a_dispersed_family(tmp_path):
    ledger, reg = make(tmp_path, primary_metric="clv_log", secondary_metrics=[])
    run_trial(reg, {}, per_decision(100, 4, 0.05, seed=1), "clv_log", ledger, "snap", L3_EXTRA)
    assert status(grade(HID, ledger), "L3", "deflated Sharpe >= 0.95") == "pass"  # alone: DSR = PSR(0)
    for i in range(6):
        other = load(
            write_registration(tmp_path, f"sharp-close-other{i}-v1", primary_metric="clv_log", secondary_metrics=[])
        )
        ledger.register(other)
        run_trial(other, {}, per_decision(100, 4, 0.04 * (i - 2), seed=10 + i), "clv_log", ledger, "snap", n_boot=200)
    g = grade(HID, ledger)
    assert status(g, "L3", "deflated Sharpe >= 0.95") == "fail", g.table().to_string()
    assert g.rung == "L2"
    assert expected_max_sharpe(7, 0.1) > 0


def test_caller_cannot_inflate_the_sharpe_ratio(tmp_path):
    ledger, reg = make(tmp_path, primary_metric="clv_log", secondary_metrics=[], grid={"k": [1, 2]})
    for k in (1, 2):
        run_trial(
            reg,
            {"k": k},
            per_decision(100, 4, 0.004, seed=k),
            "clv_log",
            ledger,
            "s",
            L3_EXTRA | {"sr": 5.0, "kurt": 3.0},
        )
    c = criterion(grade(HID, ledger, config_hash=config_hash({"k": 1})), "L3", "deflated Sharpe >= 0.95")
    assert "SR=5" not in c.detail


# --------------------------------------------------------------------------- L3 (c) PBO


def test_pbo_fails_when_the_grid_winner_is_luck(tmp_path):
    ledger, reg = make(tmp_path, grid={"k": [1, 2, 3, 4]}, **CLV)
    for i, cfg in enumerate(reg.expand_grid()):
        run_trial(reg, cfg, per_decision(100, 4, 0.05, seed=20 + i), "clv_log", ledger, "snap", L3_EXTRA)
    c = criterion(grade(HID, ledger), "L3", "PBO < 0.25")
    assert c.status == "fail" and c.value > 0.25


def test_pbo_passes_when_one_config_dominates(tmp_path):
    ledger, reg = make(tmp_path, grid={"k": [1, 2, 3, 4]}, **CLV)
    for i, cfg in enumerate(reg.expand_grid()):
        mu = 0.08 if cfg["k"] == 1 else 0.0
        run_trial(reg, cfg, per_decision(100, 4, mu, seed=30 + i), "clv_log", ledger, "snap", L3_EXTRA)
    g = grade(HID, ledger)
    assert g.config_hash == config_hash({"k": 1})
    c = criterion(g, "L3", "PBO < 0.25")
    assert c.status == "pass" and c.value < 0.05
    assert g.rung == "L3", g.table().to_string()


def test_pbo_uses_the_union_of_blocks(tmp_path):
    # Config k=1 bets every day; the others only on even days. Missing days are zero-filled (no bets).
    ledger, reg = make(tmp_path, grid={"k": [1, 2, 3, 4]}, **CLV)
    for i, cfg in enumerate(reg.expand_grid()):
        df = per_decision(100, 4, 0.08 if cfg["k"] == 1 else 0.0, seed=40 + i)
        if cfg["k"] != 1:
            days = sorted(df["block"].unique())
            df = df[df["block"].isin(days[::2])]
        run_trial(reg, cfg, df, "clv_log", ledger, "snap", L3_EXTRA)
    c = criterion(grade(HID, ledger), "L3", "PBO < 0.25")
    assert c.status == "pass" and "100 blocks" in c.detail


def test_pbo_is_not_applicable_when_grid_points_are_identical(tmp_path):
    ledger, reg = make(tmp_path, grid={"k": [1, 2, 3, 4]}, **CLV)
    data = per_decision(100, 4, 0.05)
    for cfg in reg.expand_grid():
        run_trial(reg, cfg, data, "clv_log", ledger, "snap", L3_EXTRA)
    c = criterion(grade(HID, ledger), "L3", "PBO < 0.25")
    assert c.status == "n/a" and "identical" in c.detail


def test_dsr_and_pbo_are_never_pass_for_predictive_metrics(tmp_path):
    # L3(f): a probability-kind metric with a dominant grid point still gets n/a, not "pass".
    ledger, reg = make(tmp_path, grid={"k": [1, 2, 3, 4]})
    assert reg.metric_kind == "probability"
    for i, cfg in enumerate(reg.expand_grid()):
        mu = 0.08 if cfg["k"] == 1 else 0.0
        run_trial(reg, cfg, per_decision(100, 4, mu, seed=30 + i), "edge_bits", ledger, "snap", L3_EXTRA)
    g = grade(HID, ledger)
    assert status(g, "L3", "deflated Sharpe >= 0.95") == "n/a"
    assert status(g, "L3", "PBO < 0.25") == "n/a"
    assert "probability" in criterion(g, "L3", "PBO < 0.25").detail


def test_pbo_waits_for_the_whole_grid(tmp_path):
    ledger, reg = make(tmp_path, grid={"k": [1, 2, 3, 4]}, **CLV)
    run_trial(reg, {"k": 1}, per_decision(100, 4, 0.08), "clv_log", ledger, "snap", L3_EXTRA)
    assert status(grade(HID, ledger), "L3", "PBO < 0.25") == "not-evaluated"


# --------------------------------------------------------------------------- L3 (d), (e)


def test_unstable_sub_periods_fail_l3(tmp_path):
    ledger, reg = make(tmp_path)
    data = two_phase(99, 4, early=0.2, late=-0.01)
    res = run_trial(reg, {}, data, "edge_bits", ledger, "snap", L3_EXTRA)
    assert res.ci_lo > 0  # precondition: the pooled effect clears L2
    g = grade(HID, ledger)
    assert status(g, "L3", "sub-period sign stability") == "fail"
    assert g.rung == "L2"


def test_league_carry_needs_a_league_column_or_the_single_league_flag(tmp_path):
    ledger, reg = make(tmp_path)
    run_trial(reg, {}, per_decision(100, 4, 0.05, leagues=None), "edge_bits", ledger, "s", L3_EXTRA)
    assert status(grade(HID, ledger), "L3", "not carried by one league") == "not-evaluated"
    d = subdir(tmp_path, "single")
    path = write_registration(d)
    path.write_text(path.read_text() + "single_league: true\n")
    ledger2, reg2 = Ledger(d / "l.duckdb"), load(path)
    ledger2.register(reg2)
    run_trial(reg2, {}, per_decision(100, 4, 0.05, leagues=None), "edge_bits", ledger2, "s", L3_EXTRA)
    assert status(grade(HID, ledger2), "L3", "not carried by one league") == "n/a"


def test_one_league_in_the_data_is_not_leave_one_out_evidence(tmp_path):
    ledger, reg = make(tmp_path)
    run_trial(reg, {}, per_decision(100, 4, 0.05, leagues=("E0",)), "edge_bits", ledger, "s", L3_EXTRA)
    c = criterion(grade(HID, ledger), "L3", "not carried by one league")
    assert c.status == "not-evaluated" and "single_league" in c.detail


def test_caller_supplied_leave_one_out_signs_are_ignored(tmp_path):
    ledger, reg = make(tmp_path)
    fake = {"leave_one_league_out": {"E0": 0.04, "D1": 0.05}}
    res = run_trial(reg, {}, per_decision(100, 4, 0.05, leagues=None), "edge_bits", ledger, "s", L3_EXTRA | fake)
    assert res.extra["caller_leave_one_league_out"] == fake["leave_one_league_out"]
    assert status(grade(HID, ledger), "L3", "not carried by one league") == "not-evaluated"


def test_effect_carried_by_one_league_fails_even_with_same_signs(tmp_path):
    # D1 carries the effect; without D1 the E0-only estimate is still positive but its CI includes 0.
    ledger, reg = make(tmp_path)
    df = per_decision(100, 4, 0.0, seed=3)
    df.loc[df["league"] == "D1", "value"] += 0.1
    df.loc[df["league"] == "E0", "value"] += 0.015
    res = run_trial(reg, {}, df, "edge_bits", ledger, "s", L3_EXTRA)
    loo = res.extra["leave_one_league_out"]
    assert loo["D1"]["est"] > 0 and loo["D1"]["lo"] < 0  # precondition: same sign, CI spans 0
    c = criterion(grade(HID, ledger), "L3", "not carried by one league")
    assert c.status == "fail" and "D1" in c.detail


def test_effect_spread_across_leagues_passes(tmp_path):
    ledger, _ = strong_l3(tmp_path)
    c = criterion(grade(HID, ledger), "L3", "not carried by one league")
    assert c.status == "pass" and c.value == "2 of 2"


def test_effect_above_ceiling_fails_l3(tmp_path):
    ledger, reg = make(tmp_path)
    run_trial(reg, {}, per_decision(100, 4, 0.05), "edge_bits", ledger, "snap", L3_EXTRA | {"within_ceiling": False})
    g = grade(HID, ledger)
    assert status(g, "L3", "within literature ceiling") == "fail"
    assert g.rung == "L2"


# --------------------------------------------------------------------------- L4 / L5


def test_l4_needs_auditor_and_edgelab(tmp_path):
    ledger, _ = strong_l3(tmp_path, **CLV)
    ledger.record_review(HID, "audit", "SIGNED", {"file": "research/audits/x.md"})
    g = grade(HID, ledger)
    assert status(g, "L4", "skeptic-auditor signed") == "pass"
    assert status(g, "L4", "edgelab grade >= promising") == "not-evaluated"
    assert g.rung == "L3"
    ledger.record_review(HID, "edgelab", "promising")
    assert grade(HID, ledger).rung == "L4"


def test_scoped_audit_reviews_apply_only_to_matching_configs(tmp_path):
    ledger, reg = make(tmp_path, grid={"source": ["max_open", "bet365_open"]})
    run_trial(reg, {"source": "max_open"}, per_decision(100, 4, 0.08, seed=1), "edge_bits", ledger, "s", L3_EXTRA)
    run_trial(reg, {"source": "bet365_open"}, per_decision(100, 4, 0.08, seed=2), "edge_bits", ledger, "s", L3_EXTRA)
    ledger.record_review(HID, "audit", "SIGNED", {"scope": {"source": "max_open"}})
    ledger.record_review(HID, "audit", "REJECTED", {"scope": {"source": "bet365_open"}})
    ledger.record_review(HID, "edgelab", "supported")
    max_open = grade(HID, ledger, config_hash=config_hash({"source": "max_open"}))
    bet365 = grade(HID, ledger, config_hash=config_hash({"source": "bet365_open"}))
    assert status(max_open, "L4", "skeptic-auditor signed") == "pass"
    assert status(bet365, "L4", "skeptic-auditor signed") == "fail"


def test_edgelab_data_fit_refusal_is_l4_partial(tmp_path):
    ledger, _ = strong_l3(tmp_path, **CLV)
    ledger.record_review(HID, "audit", "SIGNED")
    ledger.record_review(HID, "edgelab", "not_assessable", {"reason": "minimum detectable Sharpe too high"})
    g = grade(HID, ledger)
    assert status(g, "L4", "edgelab grade >= promising") == "partial"
    assert g.rung == "L4-partial"


def test_edgelab_no_edge_fails_l4_and_latest_review_wins(tmp_path):
    ledger, _ = strong_l3(tmp_path, **CLV)
    ledger.record_review(HID, "audit", "SIGNED")
    ledger.record_review(HID, "edgelab", "supported")
    ledger.record_review(HID, "edgelab", "no_edge")
    assert grade(HID, ledger).rung == "L3"


def test_audit_rejection_fails_l4(tmp_path):
    ledger, _ = strong_l3(tmp_path, **CLV)
    ledger.record_review(HID, "audit", "REJECTED")
    ledger.record_review(HID, "edgelab", "supported")
    assert grade(HID, ledger).rung == "L3"


@pytest.mark.parametrize(
    ("detail", "rung"),
    [
        ({"n_settled": 200, "clv_ci_lo": 0.001}, "L5"),
        ({"n_settled": 90, "weeks": 9, "clv_ci_lo": 0.002}, "L5"),
        ({"n_settled": 100, "weeks": 3, "clv_ci_lo": 0.01}, "L4"),
        ({"n_settled": 400, "clv_ci_lo": -0.001}, "L4"),
    ],
)
def test_l5_paper_trading(tmp_path, detail, rung):
    ledger, _ = strong_l3(tmp_path, **CLV)
    ledger.record_review(HID, "audit", "SIGNED")
    ledger.record_review(HID, "edgelab", "supported")
    ledger.record_review(HID, "paper", "settled", detail)
    assert grade(HID, ledger).rung == rung


def test_predictive_findings_top_out_at_l4_on_the_auditor_signature(tmp_path):
    ledger, reg = strong_l3(tmp_path)  # edge_bits: probability kind
    assert reg.metric_kind == "probability"
    ledger.record_review(HID, "audit", "SIGNED")
    ledger.record_review(HID, "paper", "settled", {"n_settled": 400, "clv_ci_lo": 0.01})
    g = grade(HID, ledger)
    assert status(g, "L4", "edgelab grade >= promising") == "n/a"
    assert all(c.status == "n/a" for c in g.criteria if c.rung == "L5")
    assert g.rung == "L4"
    ledger.record_review(HID, "audit", "REJECTED")
    assert grade(HID, ledger).rung == "L3"


def test_return_strategies_still_need_edgelab_and_paper(tmp_path):
    ledger, _ = strong_l3(tmp_path, **CLV)
    ledger.record_review(HID, "audit", "SIGNED")
    g = grade(HID, ledger)
    assert status(g, "L4", "edgelab grade >= promising") == "not-evaluated" and g.rung == "L3"


# --------------------------------------------------------------------------- selection and output


def test_grade_can_target_a_specific_config(tmp_path):
    ledger, reg = make(tmp_path, grid={"k": [1, 2]})
    run_trial(reg, {"k": 1}, per_decision(100, 4, 0.05), "edge_bits", ledger, "snap", L3_EXTRA)
    run_trial(reg, {"k": 2}, per_decision(100, 4, 0.0, seed=4), "edge_bits", ledger, "snap", L3_EXTRA)
    assert grade(HID, ledger).config_hash == config_hash({"k": 1})
    g2 = grade(HID, ledger, config_hash=config_hash({"k": 2}))
    assert g2.config_hash == config_hash({"k": 2})
    assert g2.rung == "L1"


def test_secondary_metric_trials_are_not_graded(tmp_path):
    ledger, reg = make(tmp_path)
    run_trial(reg, {}, per_decision(100, 4, 0.05), "roi", ledger, "snap", L3_EXTRA)
    assert grade(HID, ledger).rung == "L0"


def test_imported_rows_cannot_clear_l1_on_their_own(tmp_path):
    ledger, reg = make(tmp_path)
    extra = {"imported": True, "window": "evaluation", "is_primary_metric": True}
    rec: TrialRecord = native_record(reg, extra_json=json.dumps(extra))
    ledger.record_trial(rec)
    g = grade(HID, ledger)
    assert g.trial_id == rec.trial_id
    assert g.rung == "L0"


def test_table_has_one_row_per_criterion(tmp_path):
    ledger, _ = strong_l3(tmp_path)
    g = grade(HID, ledger)
    t = g.table()
    assert list(t.columns) == ["rung", "criterion", "status", "value", "threshold", "detail"]
    assert len(t) == len(g.criteria)
    assert set(t["rung"]) == {"L0", "L1", "L2", "L3", "L4", "L5"}


def test_grading_never_creates_a_ledger(tmp_path):
    path = tmp_path / "missing" / "l.duckdb"
    assert grade(HID, Ledger(path)).rung == "none"
    assert not path.exists()
