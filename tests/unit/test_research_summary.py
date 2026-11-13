"""Read-only evidence summary over the research ledger, and the STANDARDS.md ratchet (display floor),
on synthetic ledgers."""

from __future__ import annotations

import json

import pytest
from research_fixtures import per_decision, write_registration

from pitch_edge.research.ledger import Ledger
from pitch_edge.research.registry import load
from pitch_edge.research.summary import display_floor, evidence_summary, meets_floor
from pitch_edge.research.trial import run_trial

pytestmark = pytest.mark.unit

L3_EXTRA = {"clean": True, "within_ceiling": True}
CLV = {"primary_metric": "clv_log", "secondary_metrics": []}


def _register(ledger: Ledger, directory, hypothesis_id: str, family: str, **kwargs):
    d = directory / hypothesis_id
    d.mkdir()
    reg = load(write_registration(d, hypothesis_id, family=family, **kwargs))
    ledger.register(reg)
    return reg


def _strong(ledger: Ledger, reg, mu: float = 0.08, seed: int = 1):
    """A trial that clears L3 on its own (no reviews yet)."""
    return run_trial(reg, {}, per_decision(100, 4, mu, seed=seed), reg.primary_metric, ledger, "snap", L3_EXTRA)


def _sign_off(ledger: Ledger, hypothesis_id: str) -> None:
    ledger.record_review(hypothesis_id, "audit", "SIGNED", {"file": "research/audits/x.md"})
    ledger.record_review(hypothesis_id, "edgelab", "promising")


def test_missing_ledger_gives_an_empty_summary_with_the_l4_floor(tmp_path):
    ledger = Ledger(tmp_path / "ledger.duckdb")
    summary = evidence_summary(ledger)
    assert summary.families == []
    assert display_floor(summary) == "L4"
    assert summary.validated_edge_exists is False
    assert not ledger.path.exists()  # summarising never creates the ledger


def test_family_best_rung_champion_and_headline_of_the_graded_trial(tmp_path):
    ledger = Ledger(tmp_path / "ledger.duckdb")
    strong = _register(ledger, tmp_path, "fam-a-strong-v1", "fam-a")
    _register(ledger, tmp_path, "fam-a-idle-v1", "fam-a")
    trial = _strong(ledger, strong)

    summary = evidence_summary(ledger)
    (fam,) = summary.families
    assert fam.family == "fam-a"
    assert {h.id: h.rung for h in fam.hypotheses} == {"fam-a-strong-v1": "L3", "fam-a-idle-v1": "L0"}
    assert fam.best_rung == "L3"
    assert fam.champion == "fam-a-strong-v1"
    head = summary.hypothesis("fam-a-strong-v1").headline
    assert head is not None
    assert (head.metric, head.n, head.trial_id) == ("edge_bits", 400, trial.trial_id)
    assert head.estimate == pytest.approx(trial.metric_value)
    assert (head.ci_lo, head.ci_hi) == pytest.approx((trial.ci_lo, trial.ci_hi))
    assert summary.hypothesis("fam-a-idle-v1").headline is None


def test_no_strategy_at_l4_keeps_the_floor_at_l4_and_nothing_is_an_edge(tmp_path):
    ledger = Ledger(tmp_path / "ledger.duckdb")
    _strong(ledger, _register(ledger, tmp_path, "fam-a-strong-v1", "fam-a"))
    summary = evidence_summary(ledger)
    floor = display_floor(summary)
    assert floor == "L4"
    assert not meets_floor(summary.hypothesis("fam-a-strong-v1").rung, floor)
    assert summary.validated_edge_exists is False


def test_an_l4_champion_sets_the_floor_and_is_shown_as_an_edge(tmp_path):
    ledger = Ledger(tmp_path / "ledger.duckdb")
    _strong(ledger, _register(ledger, tmp_path, "fam-a-strong-v1", "fam-a", **CLV))
    _sign_off(ledger, "fam-a-strong-v1")
    summary = evidence_summary(ledger)
    assert summary.hypothesis("fam-a-strong-v1").rung == "L4"
    assert display_floor(summary) == "L4"
    assert summary.validated_edge_exists is True


def test_an_l5_champion_raises_the_floor_for_every_family(tmp_path):
    ledger = Ledger(tmp_path / "ledger.duckdb")
    _strong(ledger, _register(ledger, tmp_path, "fam-a-strong-v1", "fam-a", **CLV))
    _sign_off(ledger, "fam-a-strong-v1")
    _strong(ledger, _register(ledger, tmp_path, "fam-b-strong-v1", "fam-b", **CLV), seed=2)
    _sign_off(ledger, "fam-b-strong-v1")
    ledger.record_review("fam-b-strong-v1", "paper", "settled", {"n_settled": 200, "clv_ci_lo": 0.001})

    summary = evidence_summary(ledger)
    assert summary.hypothesis("fam-b-strong-v1").rung == "L5"
    floor = display_floor(summary)
    assert floor == "L5"
    assert not meets_floor(summary.hypothesis("fam-a-strong-v1").rung, floor)  # L4 no longer enough
    assert meets_floor("L5", floor)


def test_l4_partial_does_not_reach_the_floor(tmp_path):
    ledger = Ledger(tmp_path / "ledger.duckdb")
    _strong(ledger, _register(ledger, tmp_path, "fam-a-strong-v1", "fam-a", **CLV))
    ledger.record_review("fam-a-strong-v1", "audit", "SIGNED")
    ledger.record_review("fam-a-strong-v1", "edgelab", "not_assessable")
    summary = evidence_summary(ledger)
    assert summary.hypothesis("fam-a-strong-v1").rung == "L4-partial"
    assert display_floor(summary) == "L4"
    assert summary.validated_edge_exists is False


def test_rung_ties_go_to_the_most_recent_hypothesis(tmp_path):
    ledger = Ledger(tmp_path / "ledger.duckdb")
    _register(ledger, tmp_path, "fam-a-old-v1", "fam-a")
    _register(ledger, tmp_path, "fam-a-new-v1", "fam-a")
    (fam,) = evidence_summary(ledger).families
    assert fam.best_rung == "L0"
    assert fam.champion == "fam-a-new-v1"


def test_reviews_carry_verdict_scope_and_whether_they_apply_to_the_headline(tmp_path):
    ledger = Ledger(tmp_path / "ledger.duckdb")
    _strong(ledger, _register(ledger, tmp_path, "fam-a-strong-v1", "fam-a"))
    ledger.record_review("fam-a-strong-v1", "audit", "REJECTED", {"scope": {"source": "bet365_open"}})
    ledger.record_review("fam-a-strong-v1", "edgelab", "no_edge")
    reviews = evidence_summary(ledger).hypothesis("fam-a-strong-v1").reviews
    assert [(r.kind, r.verdict, r.scope, r.applies_to_headline) for r in reviews] == [
        ("audit", "REJECTED", {"source": "bet365_open"}, False),
        ("edgelab", "no_edge", None, True),
    ]


def test_to_dict_is_json_with_floor_and_edge_flag(tmp_path):
    ledger = Ledger(tmp_path / "ledger.duckdb")
    _strong(ledger, _register(ledger, tmp_path, "fam-a-strong-v1", "fam-a"))
    payload = json.loads(json.dumps(evidence_summary(ledger).to_dict()))
    assert payload["display_floor"] == "L4"
    assert payload["validated_edge_exists"] is False
    (fam,) = payload["families"]
    assert fam["champion"] == {"id": "fam-a-strong-v1", "rung": "L3"}
    (hyp,) = fam["hypotheses"]
    assert hyp["headline"]["metric"] == "edge_bits"
    assert hyp["shown_as_edge"] is False


@pytest.mark.parametrize(
    ("rung", "floor", "expected"),
    [("L3", "L4", False), ("L4-partial", "L4", False), ("L4", "L4", True), ("L5", "L4", True), ("none", "L4", False)],
)
def test_meets_floor_orders_rungs(rung, floor, expected):
    assert meets_floor(rung, floor) is expected


def test_load_status_combines_both_ledgers_and_reuses_the_result_until_a_file_changes(tmp_path):
    from pitch_edge.research.summary import load_status

    ledger_path, paper_path = tmp_path / "ledger.duckdb", tmp_path / "paper.duckdb"
    first = load_status(ledger_path, paper_path)
    assert first.research.families == []
    assert first.paper.available is False
    again = load_status(ledger_path, paper_path)  # nothing changed on disk
    assert again.research is first.research and again.paper is first.paper

    _register(Ledger(ledger_path), tmp_path, "fam-a-idle-v1", "fam-a")
    second = load_status(ledger_path, paper_path)
    assert second.research is not first.research
    assert [f.family for f in second.research.families] == ["fam-a"]
    payload = json.loads(json.dumps(second.to_dict()))
    assert set(payload) == {"research", "paper"}
    assert payload["research"]["display_floor"] == "L4"


def test_a_paper_ledger_change_reuses_the_research_summary(tmp_path):
    from pitch_edge.paper.store import PaperStore
    from pitch_edge.research.summary import load_status

    ledger_path, paper_path = tmp_path / "ledger.duckdb", tmp_path / "paper.duckdb"
    _register(Ledger(ledger_path), tmp_path, "fam-a-idle-v1", "fam-a")
    first = load_status(ledger_path, paper_path)
    with PaperStore(paper_path):
        pass  # creates the paper ledger file
    second = load_status(ledger_path, paper_path)
    assert second.paper.available is True and first.paper.available is False
    assert second.research is first.research  # the ladder was not regraded


def test_to_dict_says_how_each_headline_was_selected(tmp_path):
    ledger = Ledger(tmp_path / "ledger.duckdb")
    _strong(ledger, _register(ledger, tmp_path, "fam-a-strong-v1", "fam-a"))
    payload = evidence_summary(ledger).to_dict()
    assert "smallest p" in payload["headline_selection"]


# --------------------------------------------------------------------------- champion eligibility


def test_an_in_scope_rejected_audit_cannot_be_champion(tmp_path):
    ledger = Ledger(tmp_path / "ledger.duckdb")
    _strong(ledger, _register(ledger, tmp_path, "fam-a-strong-v1", "fam-a"))
    _register(ledger, tmp_path, "fam-a-idle-v1", "fam-a")
    ledger.record_review("fam-a-strong-v1", "audit", "REJECTED", {"basis": "stale-book artefact"})
    summary = evidence_summary(ledger)
    (fam,) = summary.families
    assert summary.hypothesis("fam-a-strong-v1").rung == "L3"  # the rung itself is unchanged
    assert fam.champion == "fam-a-idle-v1" and fam.best_rung == "L0"
    entry = next(h for h in summary.to_dict()["families"][0]["hypotheses"] if h["id"] == "fam-a-strong-v1")
    assert entry["champion_eligible"] is False and "REJECTED" in entry["ineligible_reason"]


def test_a_rejection_scoped_to_another_config_does_not_disqualify(tmp_path):
    ledger = Ledger(tmp_path / "ledger.duckdb")
    _strong(ledger, _register(ledger, tmp_path, "fam-a-strong-v1", "fam-a"))
    ledger.record_review("fam-a-strong-v1", "audit", "REJECTED", {"scope": {"source": "bet365_open"}})
    assert evidence_summary(ledger).families[0].champion == "fam-a-strong-v1"


def test_a_later_signed_audit_restores_eligibility(tmp_path):
    ledger = Ledger(tmp_path / "ledger.duckdb")
    _strong(ledger, _register(ledger, tmp_path, "fam-a-strong-v1", "fam-a"))
    ledger.record_review("fam-a-strong-v1", "audit", "REJECTED")
    ledger.record_review("fam-a-strong-v1", "audit", "SIGNED")
    assert evidence_summary(ledger).families[0].champion == "fam-a-strong-v1"


def test_post_hoc_hypotheses_are_never_champions_or_edges(tmp_path):
    ledger = Ledger(tmp_path / "ledger.duckdb")
    _strong(ledger, _register(ledger, tmp_path, "fam-a-posthoc-v1", "fam-a", extra_fields={"post_hoc": True}, **CLV))
    _sign_off(ledger, "fam-a-posthoc-v1")
    summary = evidence_summary(ledger)
    assert summary.hypothesis("fam-a-posthoc-v1").rung == "L4"
    (fam,) = summary.families
    assert fam.champion is None and fam.best_rung == "none"
    assert summary.validated_edge_exists is False
    entry = summary.to_dict()["families"][0]["hypotheses"][0]
    assert entry["shown_as_edge"] is False and entry["ineligible_reason"] == "post-hoc registration"


# --------------------------------------------------------------------------- headline transparency


def test_largest_n_grid_point_is_shown_beside_the_lowest_p_headline(tmp_path):
    ledger = Ledger(tmp_path / "ledger.duckdb")
    reg = _register(ledger, tmp_path, "fam-a-grid-v1", "fam-a", grid={"theta": [0.0, 0.05]})
    small = run_trial(reg, {"theta": 0.05}, per_decision(80, 4, 0.12, seed=1), "edge_bits", ledger, "s", L3_EXTRA)
    big = run_trial(reg, {"theta": 0.0}, per_decision(150, 6, 0.01, seed=2), "edge_bits", ledger, "s", L3_EXTRA)
    assert small.p_value <= big.p_value and small.n < big.n  # precondition
    h = evidence_summary(ledger).hypothesis("fam-a-grid-v1")
    assert h.headline is not None and h.headline.trial_id == small.trial_id
    assert h.largest_n is not None and h.largest_n.trial_id == big.trial_id
    d = next(x for x in evidence_summary(ledger).to_dict()["families"][0]["hypotheses"])
    assert d["headline"]["selected_by"] == "lowest p among 2 evaluation-window grid points"
    assert d["largest_n_grid_point"]["n"] == 900 and "largest n" in d["largest_n_grid_point"]["label"]


def test_single_grid_point_has_no_separate_largest_n(tmp_path):
    ledger = Ledger(tmp_path / "ledger.duckdb")
    _strong(ledger, _register(ledger, tmp_path, "fam-a-strong-v1", "fam-a"))
    d = evidence_summary(ledger).to_dict()["families"][0]["hypotheses"][0]
    assert d["largest_n_grid_point"] is None


def test_secondary_segments_are_reported_but_never_the_headline(tmp_path):
    ledger = Ledger(tmp_path / "ledger.duckdb")
    reg = _register(
        ledger, tmp_path, "fam-a-seg-v1", "fam-a", extra_windows={"post_segment": ["2024-01-01", "2024-12-31"]}
    )
    ev = run_trial(reg, {}, per_decision(100, 4, 0.0, seed=1), "edge_bits", ledger, "s", L3_EXTRA)
    run_trial(reg, {}, per_decision(9, 4, 0.3), "edge_bits", ledger, "s", L3_EXTRA, window="post_segment")
    h = evidence_summary(ledger).hypothesis("fam-a-seg-v1")
    assert h.headline is not None and h.headline.trial_id == ev.trial_id
    assert [s.window for s in h.secondary_segments] == ["post_segment"]
    d = evidence_summary(ledger).to_dict()["families"][0]["hypotheses"][0]
    assert d["secondary_segments"][0]["n"] == 36 and "never graded" in d["secondary_segments"][0]["label"]


def test_largest_n_is_omitted_when_the_headline_already_has_the_largest_n(tmp_path):
    ledger = Ledger(tmp_path / "ledger.duckdb")
    reg = _register(ledger, tmp_path, "fam-a-grid-v1", "fam-a", grid={"theta": [0.0, 0.05]})
    # The weaker point runs first, so an n-tie broken by run time would pick it over the headline.
    run_trial(reg, {"theta": 0.0}, per_decision(100, 4, 0.01, seed=2), "edge_bits", ledger, "s", L3_EXTRA)
    run_trial(reg, {"theta": 0.05}, per_decision(100, 4, 0.08, seed=1), "edge_bits", ledger, "s", L3_EXTRA)
    assert evidence_summary(ledger).hypothesis("fam-a-grid-v1").largest_n is None


# --------------------------------------------------------------------------- predictive findings


def test_a_predictive_l4_is_never_an_edge_and_never_raises_the_floor(tmp_path):
    ledger = Ledger(tmp_path / "ledger.duckdb")
    pred = _register(ledger, tmp_path, "fam-a-pred-v1", "fam-a")  # edge_bits: probability kind
    assert pred.metric_kind == "probability"
    _strong(ledger, pred)
    ledger.record_review("fam-a-pred-v1", "audit", "SIGNED")
    summary = evidence_summary(ledger)
    assert summary.hypothesis("fam-a-pred-v1").rung == "L4"
    (fam,) = summary.families
    assert fam.champion == "fam-a-pred-v1"  # still the family's champion
    assert fam.return_champion is None
    assert display_floor(summary) == "L4"
    assert summary.validated_edge_exists is False
    d = summary.to_dict()["families"][0]
    assert d["hypotheses"][0]["shown_as_edge"] is False and d["return_champion"] is None


def test_the_floor_comes_from_return_kind_champions_only(tmp_path):
    ledger = Ledger(tmp_path / "ledger.duckdb")
    pred = _register(ledger, tmp_path, "fam-a-pred-v1", "fam-a")
    _strong(ledger, pred)
    ledger.record_review("fam-a-pred-v1", "audit", "SIGNED")
    strat = _register(ledger, tmp_path, "fam-a-strat-v1", "fam-a", **CLV)
    _strong(ledger, strat, seed=2)
    summary = evidence_summary(ledger)
    (fam,) = summary.families
    # The strategy stops at L2: its DSR has N = 2 family tests but only one return-kind Sharpe ratio.
    assert fam.champion == "fam-a-pred-v1" and summary.hypothesis("fam-a-strat-v1").rung == "L2"
    assert fam.return_champion == "fam-a-strat-v1" and fam.return_best_rung == "L2"
    assert display_floor(summary) == "L4"
    assert summary.to_dict()["families"][0]["return_champion"] == {"id": "fam-a-strat-v1", "rung": "L2"}
