"""`soft-vs-sharp-live`: back the best soft-book price when it beats the de-vigged sharp price in the same
the-odds-api response by > 2%, with sanity caps; and the per-snapshot opportunity-frequency log."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from pitch_edge.paper.pricing import power_devig_odds
from pitch_edge.paper.store import PaperStore, Quote
from pitch_edge.paper.strategies import Snapshot, SoftVsSharpLive, available_strategies, run_decide

pytestmark = pytest.mark.unit

NOW = datetime(2026, 10, 21, 9, 0, tzinfo=UTC)
CAPTURED = NOW - timedelta(minutes=5)
KO = datetime(2026, 10, 24, 14, 0, tzinfo=UTC)
SEL = ("home", "draw", "away")
SHARP = (2.0, 3.6, 4.2)
FAIR = [1 / p for p in power_devig_odds(SHARP)]  # fair decimal odds


def _book(book, odds, captured=CAPTURED, event="e1", home="Ipswich Town", away="Nottingham Forest"):
    return [
        Quote(
            source=f"odds_api:{book}",
            source_event_id=event,
            home=home,
            away=away,
            league="EPL",
            kickoff_at=KO,
            selection=s,
            captured_at=captured,
            decimal_odds=o,
        )
        for s, o in zip(SEL, odds, strict=True)
    ]


@pytest.fixture
def store(tmp_path):
    s = PaperStore(tmp_path / "paper.duckdb")
    yield s
    s.close()


def _snap(store, as_of=NOW):
    return Snapshot.from_store(store, as_of=as_of, code_version="sha")


def test_backs_the_best_soft_price_above_two_percent_over_fair(store):
    store.add_quotes(_book("pinnacle", SHARP))
    good = round(FAIR[0] * 1.04, 2)
    store.add_quotes(_book("williamhill", (good, 3.3, 3.9)))
    store.add_quotes(_book("bet365", (round(FAIR[0] * 1.03, 2), 3.4, 4.0)))
    [d] = SoftVsSharpLive().decide(_snap(store))
    assert (d.selection, d.price_taken, d.venue) == ("home", good, "odds_api:williamhill")
    assert d.hypothesis_id == "bookmaker-line-structure-live-soft-vs-sharp-v1"
    assert d.strategy_id == "soft-vs-sharp-live"
    assert d.params["book"] == "williamhill" and d.params["sharp_source"] == "odds_api:pinnacle"
    assert d.params["edge"] == pytest.approx(good / FAIR[0] - 1)
    assert d.model_prob == pytest.approx(1 / FAIR[0])
    assert len(d.inputs_quote_ids) == 4


def test_no_decision_at_or_below_the_threshold(store):
    store.add_quotes(_book("pinnacle", SHARP))
    store.add_quotes(_book("williamhill", (round(FAIR[0] * 1.015, 3), 3.3, 3.9)))
    assert SoftVsSharpLive().decide(_snap(store)) == []


def test_rejects_prices_more_than_ten_percent_above_the_sharp_quote(store):
    store.add_quotes(_book("pinnacle", SHARP))
    store.add_quotes(_book("williamhill", (SHARP[0] * 1.11, 3.3, 3.9)))
    assert SoftVsSharpLive().decide(_snap(store)) == []


def test_exchanges_and_sharp_books_are_never_the_soft_book(store):
    store.add_quotes(_book("pinnacle", SHARP))
    store.add_quotes(_book("betfair_ex_uk", (round(FAIR[0] * 1.05, 2), 3.7, 4.4)))
    store.add_quotes(_book("matchbook", (round(FAIR[0] * 1.05, 2), 3.7, 4.4)))
    assert SoftVsSharpLive().decide(_snap(store)) == []


def test_sharp_reference_falls_back_to_betfair_then_smarkets(store):
    store.add_quotes(_book("smarkets", SHARP))
    store.add_quotes(_book("williamhill", (round(FAIR[0] * 1.04, 2), 3.3, 3.9)))
    [d] = SoftVsSharpLive().decide(_snap(store))
    assert d.params["sharp_source"] == "odds_api:smarkets"


def test_soft_and_sharp_must_come_from_the_same_response(store):
    store.add_quotes(_book("pinnacle", SHARP, captured=CAPTURED))
    store.add_quotes(_book("williamhill", (round(FAIR[0] * 1.04, 2), 3.3, 3.9), captured=CAPTURED - timedelta(hours=1)))
    assert SoftVsSharpLive(max_quote_age_h=2).decide(_snap(store)) == []


def test_stale_responses_are_ignored(store):
    store.add_quotes(_book("pinnacle", SHARP, captured=NOW - timedelta(hours=3)))
    store.add_quotes(_book("williamhill", (round(FAIR[0] * 1.04, 2), 3.3, 3.9), captured=NOW - timedelta(hours=3)))
    assert SoftVsSharpLive(max_quote_age_h=1).decide(_snap(store)) == []


def test_one_decision_per_fixture_with_the_largest_edge(store):
    store.add_quotes(_book("pinnacle", SHARP))
    store.add_quotes(_book("williamhill", (round(FAIR[0] * 1.03, 2), round(FAIR[1] * 1.06, 2), 3.9)))
    [d] = SoftVsSharpLive().decide(_snap(store))
    assert d.selection == "draw"


def test_run_decide_logs_opportunity_frequency_per_snapshot(store):
    store.add_quotes(_book("pinnacle", SHARP))
    store.add_quotes(_book("williamhill", (round(FAIR[0] * 1.04, 2), 3.3, 3.9)))
    store.add_quotes(_book("pinnacle", SHARP, event="e2", home="Arsenal", away="Everton"))
    store.add_quotes(_book("williamhill", (1.9, 3.3, 3.9), event="e2", home="Arsenal", away="Everton"))
    strat = SoftVsSharpLive()
    run_decide(store, [strat], now=NOW, code_version="sha")
    run_decide(store, [strat], now=NOW + timedelta(minutes=10), code_version="sha")
    log = store.decide_log()
    assert list(log["n_fixtures_evaluated"]) == [2, 2]
    assert list(log["n_fixtures_qualifying"]) == [1, 1]  # counted even when already decided
    assert list(log["n_new_decisions"]) == [1, 0]
    assert (log["strategy_id"] == "soft-vs-sharp-live").all()


def test_registered_as_a_builtin():
    assert "soft-vs-sharp-live" in available_strategies()


def test_decision_params_record_each_books_last_update_staleness(store):
    def stamped(book, odds, age_s):
        out = []
        for q in _book(book, odds):
            stamp = (CAPTURED - timedelta(seconds=age_s)).strftime("%Y-%m-%dT%H:%M:%SZ")
            out.append(Quote(**{**q.__dict__, "raw_json": json.dumps({"last_update": stamp})}))
        return out

    store.add_quotes(stamped("pinnacle", SHARP, 120))
    store.add_quotes(stamped("williamhill", (round(FAIR[0] * 1.04, 2), 3.3, 3.9), 3600))
    [d] = SoftVsSharpLive().decide(_snap(store))
    assert d.params["soft_last_update_age_s"] == 3600
    assert d.params["sharp_last_update_age_s"] == 120
