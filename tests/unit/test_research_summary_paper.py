"""Read-only forward (paper-ledger) status per strategy on a tmp paper DB and a tmp research ledger: the
primary numbers come from the evaluation window and lake-sourced (final) closes only.

Dates sit in 2030 so every decision is later than the ledger's registration time whenever the suite runs
(the paper report only counts decisions made after the ledger recorded the registration)."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
import yaml
from research_fixtures import registration_text

from pitch_edge.paper.store import Decision, PaperStore, Quote, Settlement
from pitch_edge.research.ledger import Ledger
from pitch_edge.research.registry import load
from pitch_edge.research.stats import block_bootstrap
from pitch_edge.research.summary import load_paper_status, paper_summary

pytestmark = pytest.mark.unit

HID = "paper-live-test-v1"
EXPLORATORY_DAY = datetime(2030, 1, 1, 12, 0, tzinfo=UTC)
EVAL_START = datetime(2030, 1, 2, 12, 0, tzinfo=UTC)


def _ledger_with_window(tmp_path) -> Ledger:
    doc = yaml.safe_load(registration_text(HID, family="paper-test", primary_metric="clv_log"))
    doc["window"] = {"exploratory": ["2030-01-01", "2030-01-01"], "evaluation": ["2030-01-02", "2030-12-31"]}
    path = tmp_path / f"{HID}.yaml"
    path.write_text(yaml.safe_dump(doc, sort_keys=False))
    ledger = Ledger(tmp_path / "ledger.duckdb")
    ledger.register(load(path))
    return ledger


def _decide(store: PaperStore, i: int, decided_at: datetime, strategy: str = "live-test", hid: str = HID) -> str:
    ko = decided_at + timedelta(hours=6)
    q = Quote(
        source="odds_api:pinnacle",
        source_event_id=f"e{strategy}{i}",
        home=f"Home {strategy} {i}",
        away=f"Away {strategy} {i}",
        league="EPL",
        kickoff_at=ko,
        selection="home",
        captured_at=decided_at,
        decimal_odds=2.0,
    )
    store.add_quotes([q])
    fk = store.quotes().query("source_event_id == @q.source_event_id")["fixture_key"].iloc[0]
    d = Decision(
        strategy_id=strategy,
        hypothesis_id=hid,
        decided_at=decided_at,
        fixture_key=fk,
        kickoff_at=ko,
        venue="odds_api:pinnacle",
        selection="home",
        price_taken=2.0,
        paper_stake_units=1.0,
        model_prob=0.5,
        inputs_quote_ids=[q.quote_id],
        code_version="sha",
    )
    store.append_decision(d)
    return d.decision_id


def _settlement(decision_id: str, clv: float, source: str) -> Settlement:
    return Settlement(
        decision_id=decision_id,
        closing_price=2.0,
        fair_close_odds=2.0,
        closing_source=source,
        result="home",
        clv_log=clv,
        pnl=1.0,
        settled_at=datetime(2031, 1, 1, tzinfo=UTC),
    )


def _settle(store: PaperStore, decision_id: str, clv: float) -> None:
    store.append_settlement(_settlement(decision_id, clv, "lake:fd_matches.betfair_ex_close"))


def _populate(store: PaperStore) -> list[float]:
    early = _decide(store, 0, EXPLORATORY_DAY)
    _settle(store, early, 0.5)
    clvs = [0.01, 0.03, -0.01, 0.02]
    for i, clv in enumerate(clvs, start=1):
        _settle(store, _decide(store, i, EVAL_START + timedelta(days=i)), clv)
    own = _settlement(_decide(store, 5, EVAL_START + timedelta(days=5)), 0.9, "own_snapshot:odds_api:pinnacle@x")
    store.append_provisional_settlement(own)
    _decide(store, 6, EVAL_START + timedelta(days=6))  # not settled yet
    void = _decide(store, 7, EVAL_START + timedelta(days=7))
    store.flag_decision(void, "void", "postponed", flagged_at=EVAL_START + timedelta(days=8))
    _settle(store, _decide(store, 0, EVAL_START, strategy="control-test", hid="control"), 0.0)
    return clvs


def test_counts_and_clv_use_the_evaluation_window_and_lake_closes_only(tmp_path):
    ledger = _ledger_with_window(tmp_path)
    with PaperStore(tmp_path / "paper.duckdb") as store:
        clvs = _populate(store)
        status = paper_summary(store, ledger, registry_dir=tmp_path, n_boot=500, seed=1)
    s = status.strategy("live-test")
    assert "2030-01-02..2030-12-31" in s.registration_cut
    assert (s.n_decided, s.n_void, s.n_settled, s.n_provisional, s.n_exploratory) == (7, 1, 4, 1, 1)
    assert s.mean_clv == pytest.approx(np.mean(clvs))
    days = [str((EVAL_START + timedelta(days=i, hours=6)).date()) for i in range(1, 5)]
    expected = block_bootstrap(clvs, days, n_boot=500, seed=1)
    assert (s.clv_ci_lo, s.clv_ci_hi) == pytest.approx((expected.lo, expected.hi))
    assert s.l5_sample_met is False


def test_unregistered_strategies_have_no_evaluation_window_and_are_not_evidence(tmp_path):
    ledger = _ledger_with_window(tmp_path)
    with PaperStore(tmp_path / "paper.duckdb") as store:
        _populate(store)
        status = paper_summary(store, ledger, registry_dir=tmp_path, n_boot=200)
    c = status.strategy("control-test")
    assert c.counts_as_evidence is False
    assert (c.n_decided, c.n_settled) == (1, 1)
    assert status.strategy("live-test").counts_as_evidence is True


def test_too_few_settled_bets_leave_the_ci_empty(tmp_path):
    ledger = _ledger_with_window(tmp_path)
    with PaperStore(tmp_path / "paper.duckdb") as store:
        _settle(store, _decide(store, 1, EVAL_START), 0.02)
        status = paper_summary(store, ledger, registry_dir=tmp_path, n_boot=200)
    s = status.strategy("live-test")
    assert s.n_settled == 1
    assert s.clv_ci_lo is None and s.clv_ci_hi is None
    assert json.loads(json.dumps(status.to_dict()))["strategies"][0]["clv_ci95"] is None


def test_load_paper_status_reports_a_missing_ledger_without_creating_it(tmp_path):
    path = tmp_path / "paper" / "paper.duckdb"
    status = load_paper_status(path, Ledger(tmp_path / "ledger.duckdb"), registry_dir=tmp_path)
    assert status.available is False
    assert status.strategies == []
    assert not path.exists()


def test_load_paper_status_reads_an_existing_ledger(tmp_path):
    ledger = _ledger_with_window(tmp_path)
    path = tmp_path / "paper.duckdb"
    with PaperStore(path) as store:
        _populate(store)
    status = load_paper_status(path, ledger, registry_dir=tmp_path, n_boot=200)
    assert status.available is True
    assert {s.strategy_id for s in status.strategies} == {"live-test", "control-test"}
