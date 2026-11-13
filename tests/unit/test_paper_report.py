"""Paper report per strategy, and the `pitch-edge paper` CLI group wiring."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
import responses
from typer.testing import CliRunner

from pitch_edge.cli import app
from pitch_edge.paper.collectors import KALSHI_BASE, POLYMARKET_GAMMA
from pitch_edge.paper.report import report_notes, summarise
from pitch_edge.paper.store import Decision, PaperStore, Quote, Settlement
from pitch_edge.research.stats import block_bootstrap

pytestmark = pytest.mark.unit

T0 = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)


HYP = "bookmaker-line-structure-live-soft-vs-sharp-v1"
REG_RECORDED = datetime(2026, 10, 20, 20, 54)  # ledger registration time (naive UTC, as the ledger stores it)


class FakeLedger:
    def registration(self, hypothesis_id):
        return (
            {"id": HYP, "recorded_at": REG_RECORDED, "registered_at": datetime(2026, 10, 20, 21, 30)}
            if (hypothesis_id == HYP)
            else None
        )


@pytest.fixture
def registry(tmp_path):
    d = tmp_path / "registry"
    d.mkdir()
    (d / f"{HYP}.yaml").write_text(
        "window:\n  exploratory: [2026-10-20, 2026-10-20]\n  evaluation: [2026-10-21, 2027-06-30]\n"
    )
    return d


def _club(i, side):
    """Synthetic club names that no two fixtures share closely enough to be linked."""
    return f"{side}{'abcdefghijklmnopqrstuvwxyz'[i] * 5}"


def _add(
    store,
    i,
    decided,
    league="EPL",
    strategy="soft-vs-sharp-live",
    hyp=HYP,
    clv=0.01,
    final=True,
    provisional=False,
    won=True,
    source="lake:fd_matches.betfair_ex_close",
):
    ko = decided + timedelta(days=1 + i % 3, hours=2)
    q = Quote(
        source="odds_api:pinnacle",
        source_event_id=f"e{i}",
        home=_club(i, "h"),
        away=_club(i, "a"),
        league=league,
        kickoff_at=ko,
        selection="home",
        captured_at=decided,
        decimal_odds=2.0,
    )
    store.add_quotes([q])
    fk = store.quotes().query("source_event_id == @q.source_event_id")["fixture_key"].iloc[0]
    d = Decision(
        strategy_id=strategy,
        hypothesis_id=hyp,
        decided_at=decided,
        fixture_key=fk,
        kickoff_at=ko,
        venue="odds_api:williamhill",
        selection="home",
        price_taken=2.0,
        paper_stake_units=1.0,
        model_prob=0.5,
        inputs_quote_ids=[q.quote_id],
        code_version="sha",
    )
    store.append_decision(d)
    s = Settlement(
        decision_id=d.decision_id,
        closing_price=2.0,
        fair_close_odds=2.0,
        closing_source=source if final else "own_snapshot:odds_api:pinnacle@t",
        result="home" if won else "away",
        clv_log=clv,
        pnl=1.0 if won else -1.0,
        settled_at=ko + timedelta(days=1),
    )
    if final:
        store.append_settlement(s)
    elif provisional:
        store.append_provisional_settlement(s)
    return d


def test_primary_metric_uses_only_lake_closes_inside_the_evaluation_window_after_ledger_registration(
    tmp_path, registry
):
    with PaperStore(tmp_path / "p.duckdb") as store:
        before = datetime(2026, 10, 20, 21, 8, tzinfo=UTC)  # after ledger registration but exploratory window
        _add(store, 0, before, clv=0.5)
        evals = [datetime(2026, 10, 21 + k, 9, tzinfo=UTC) for k in range(4)]
        clvs = [0.02, -0.01, 0.03, 0.0]
        for i, (t, c) in enumerate(zip(evals, clvs, strict=True), start=1):
            _add(store, i, t, clv=c, won=i % 2 == 0)
        _add(store, 10, datetime(2026, 10, 22, 9, tzinfo=UTC), clv=0.9, final=False, provisional=True)
        ucl = _add(store, 11, datetime(2026, 10, 22, 10, tzinfo=UTC), league="UCL", final=False, provisional=True)
        store.flag_decision(ucl.decision_id, "no_benchmark", "no OddsPapi fixture", flagged_at=T0)
        _add(
            store,
            13,
            datetime(2026, 10, 22, 11, tzinfo=UTC),
            league="UCL",
            clv=0.7,
            source="lake:oddspapi.pinnacle_close",
        )
        voided = _add(store, 12, datetime(2026, 10, 23, 9, tzinfo=UTC), clv=0.8)
        store.flag_decision(voided.decision_id, "void", "test", flagged_at=datetime(2026, 10, 25, tzinfo=UTC))
        _add(store, 20, datetime(2026, 10, 22, 9, tzinfo=UTC), strategy="control-fav-pinnacle", hyp="control")
        rep = summarise(store, registry_dir=registry, ledger=FakeLedger(), n_boot=500, seed=1)
    rows = rep[rep["benchmark"] == "betfair_close"].set_index(["strategy_id", "segment"])
    p = rows.loc[("soft-vs-sharp-live", "primary")]
    assert p["n_decided"] == 6  # 4 final + provisional + void, inside the evaluation window
    assert (p["n_final"], p["n_provisional"], p["n_no_benchmark"], p["n_void"]) == (4, 1, 0, 1)
    assert p["n_provisional_own_snapshot"] == 1  # its row is priced from our own Pinnacle snapshot
    assert p["mean_clv"] == pytest.approx(np.mean(clvs))
    finals_days = [str((t + timedelta(days=1 + i % 3, hours=2)).date()) for i, t in enumerate(evals, start=1)]
    boot = block_bootstrap(clvs, finals_days, n_boot=500, seed=1)
    assert (p["clv_ci_lo"], p["clv_ci_hi"]) == pytest.approx((boot.lo, boot.hi))
    assert p["provisional_mean_clv"] == pytest.approx(0.9)
    assert p["roi"] == pytest.approx(0.0)  # 2 wins at +1, 2 losses at -1
    e = rows.loc[("soft-vs-sharp-live", "exploratory")]
    assert e["n_decided"] == 1 and e["mean_clv"] == pytest.approx(0.5)
    u = rows.loc[("control-fav-pinnacle", "unregistered")]
    assert u["n_decided"] == 1
    pin = rep[(rep["benchmark"] == "pinnacle_close") & (rep["segment"] == "primary")].iloc[0]
    assert (pin["n_decided"], pin["n_final"], pin["n_no_benchmark"]) == (2, 1, 1)
    assert pin["mean_clv"] == pytest.approx(0.7)
    notes = " ".join(report_notes(rep))
    assert "UCL" in notes and "pinnacle_close" in notes and "never pooled" in notes
    assert "pm-vs-sharp-gap" in notes and "placeholder" in notes and "no longer runs" in notes
    assert "max" in notes.lower() and "tautolog" in notes.lower()
    assert "2026-10-20T20:54" in notes  # the ledger registration time is what the cut uses


def test_decisions_of_a_strategy_without_a_class_still_report(tmp_path, registry):
    with PaperStore(tmp_path / "p.duckdb") as store:
        _add(store, 21, datetime(2026, 10, 22, 9, tzinfo=UTC), strategy="pm-vs-sharp-gap", hyp="pending-r1")
        rep = summarise(store, registry_dir=registry, ledger=FakeLedger(), n_boot=100, seed=1)
    row = rep.set_index(["strategy_id", "segment"]).loc[("pm-vs-sharp-gap", "unregistered")]
    assert row["n_decided"] == 1 and row["hypothesis_id"] == "pending-r1"


def test_summary_of_an_empty_ledger_is_empty(tmp_path, registry):
    with PaperStore(tmp_path / "p.duckdb") as store:
        assert summarise(store, registry_dir=registry, ledger=FakeLedger()).empty


runner = CliRunner()


def test_cli_paper_report_and_status_on_an_empty_ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("PITCH_EDGE_PAPER_DB", str(tmp_path / "paper.duckdb"))
    res = runner.invoke(app, ["paper", "report"])
    assert res.exit_code == 0, res.output
    assert "no decisions" in res.output.lower()
    res = runner.invoke(app, ["paper", "status"])
    assert res.exit_code == 0, res.output
    assert "quotes" in res.output


@responses.activate
def test_cli_paper_collect_and_decide_with_public_sources(tmp_path, monkeypatch):
    monkeypatch.setenv("PITCH_EDGE_PAPER_DB", str(tmp_path / "paper.duckdb"))
    monkeypatch.setenv("PITCH_EDGE_DATA_DIR", str(tmp_path))
    responses.add(responses.GET, f"{POLYMARKET_GAMMA}/events/keyset", json={"events": [], "next_cursor": None})
    responses.add(responses.GET, f"{KALSHI_BASE}/events", json={"events": [], "cursor": ""})
    res = runner.invoke(app, ["paper", "collect", "--sources", "polymarket,kalshi"])
    assert res.exit_code == 0, res.output
    assert "polymarket: 0 quotes" in res.output and "kalshi: 0 quotes" in res.output
    res = runner.invoke(app, ["paper", "decide"])
    assert res.exit_code == 0, res.output
    assert "control-fav-pinnacle: 0" in res.output and "pm-vs-sharp-gap" not in res.output


def test_cli_collect_odds_api_without_a_key_fails_cleanly(tmp_path, monkeypatch):
    monkeypatch.setenv("PITCH_EDGE_PAPER_DB", str(tmp_path / "paper.duckdb"))
    monkeypatch.setenv("PITCH_EDGE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ODDS_API_KEY", "")
    res = runner.invoke(app, ["paper", "collect", "--sources", "odds_api"])
    assert "ODDS_API_KEY" in res.output
    assert res.exit_code == 1


def test_cli_decide_stamps_code_hash_and_archives_the_code(tmp_path, monkeypatch):
    from pitch_edge.paper.provenance import PAPER_ROOT, code_hash

    db = tmp_path / "paper" / "paper.duckdb"
    monkeypatch.setenv("PITCH_EDGE_PAPER_DB", str(db))
    ko = datetime.now(UTC) + timedelta(days=2)
    with PaperStore(db) as s:
        s.add_quotes(
            [
                Quote(
                    source="odds_api:pinnacle",
                    source_event_id="e1",
                    home="Arsenal",
                    away="Everton",
                    league="EPL",
                    kickoff_at=ko,
                    selection=sel,
                    captured_at=datetime.now(UTC) - timedelta(minutes=5),
                    decimal_odds=o,
                )
                for sel, o in (("home", 1.5), ("draw", 4.5), ("away", 7.0))
            ]
        )
    res = runner.invoke(app, ["paper", "decide", "--strategy", "control-fav-pinnacle"])
    assert res.exit_code == 0, res.output
    h = code_hash(PAPER_ROOT)
    with PaperStore(db) as s:
        assert s.decisions().iloc[0]["code_hash"] == h
    assert (db.parent / "code" / h / "store.py").exists()


def test_tracebacks_never_print_local_variables():
    from pitch_edge.paper.cli import paper_app

    assert app.pretty_exceptions_show_locals is False
    assert paper_app.pretty_exceptions_show_locals is False
