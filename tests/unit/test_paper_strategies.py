"""Paper strategies: the Pinnacle-favourite control, the Polymarket-vs-sharp gap placeholder, the decide
loop (one decision per fixture per strategy), and the structural no-order-placement guarantee."""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from pitch_edge.paper.pricing import power_devig_odds, prediction_market_decimal_odds
from pitch_edge.paper.store import PaperStore, Quote
from pitch_edge.paper.strategies import (
    ControlFavPinnacle,
    PmVsSharpGap,
    Snapshot,
    available_strategies,
    run_decide,
)

pytestmark = pytest.mark.unit

NOW = datetime(2026, 10, 20, 18, 0, tzinfo=UTC)
KO = datetime(2026, 10, 24, 14, 0, tzinfo=UTC)
SEL = ("home", "draw", "away")


def _book(source, odds, captured=NOW - timedelta(hours=1), event="e1", kickoff=KO):
    return [
        Quote(
            source=source,
            source_event_id=event,
            home="Ipswich Town",
            away="Nottingham Forest",
            league="EPL",
            kickoff_at=kickoff,
            selection=s,
            captured_at=captured,
            decimal_odds=o,
        )
        for s, o in zip(SEL, odds, strict=True)
    ]


def _pm(asks, captured=NOW - timedelta(minutes=10), size=100.0, event="pm1"):
    return [
        Quote(
            source="polymarket",
            source_event_id=event,
            home="Ipswich Town FC",
            away="Nottingham Forest FC",
            league="EPL",
            kickoff_at=KO,
            selection=s,
            captured_at=captured,
            bid=None if a is None else a - 0.02,
            ask=a,
            size=size,
            instrument_id=f"tok-{s}",
        )
        for s, a in zip(SEL, asks, strict=True)
    ]


@pytest.fixture
def store(tmp_path):
    s = PaperStore(tmp_path / "paper.duckdb")
    yield s
    s.close()


def _snap(store, as_of=NOW):
    return Snapshot.from_store(store, as_of=as_of, code_version="testsha")


def test_control_backs_the_pinnacle_favourite_at_pinnacle(store):
    store.add_quotes(_book("odds_api:pinnacle", (3.14, 2.34, 3.48)))
    store.add_quotes(_book("odds_api:betfair_ex_uk", (1.5, 9.0, 9.0)))  # ignored: not Pinnacle
    [d] = ControlFavPinnacle().decide(_snap(store))
    assert (d.selection, d.price_taken, d.venue) == ("draw", 2.34, "odds_api:pinnacle")
    assert d.hypothesis_id == "control" and d.strategy_id == "control-fav-pinnacle"
    assert d.model_prob == pytest.approx(power_devig_odds([3.14, 2.34, 3.48])[1])
    assert len(d.inputs_quote_ids) == 3 and d.code_version == "testsha"
    assert d.decided_at == NOW and d.kickoff_at == KO


def test_control_needs_all_three_pinnacle_prices(store):
    store.add_quotes(_book("odds_api:pinnacle", (3.14, 2.34, 3.48))[:2])
    assert ControlFavPinnacle().decide(_snap(store)) == []


def test_control_ignores_stale_quotes(store):
    store.add_quotes(_book("odds_api:pinnacle", (3.14, 2.34, 3.48), captured=NOW - timedelta(hours=30)))
    assert ControlFavPinnacle(max_quote_age_h=26).decide(_snap(store)) == []


def test_snapshot_excludes_fixtures_that_have_kicked_off(store):
    store.add_quotes(
        _book(
            "odds_api:pinnacle",
            (3.14, 2.34, 3.48),
            kickoff=NOW - timedelta(minutes=1),
            captured=NOW - timedelta(hours=2),
        )
    )
    assert ControlFavPinnacle().decide(_snap(store)) == []


def test_pm_gap_backs_the_outcome_whose_ask_plus_fee_is_below_the_sharp_fair_price(store):
    sharp = (2.0, 3.6, 4.2)
    store.add_quotes(_book("odds_api:pinnacle", sharp))
    fair = power_devig_odds(sharp)
    home_ask = round(fair[0] - 0.06, 2)  # cost = ask + fee, still > theta below fair
    store.add_quotes(_pm((home_ask, 0.30, 0.25)))
    [d] = PmVsSharpGap(theta=0.03).decide(_snap(store))
    assert d.selection == "home" and d.venue == "polymarket"
    assert d.price_taken == pytest.approx(prediction_market_decimal_odds(home_ask, 0.05))
    assert d.hypothesis_id == "pending-r1" and d.model_prob == pytest.approx(fair[0])
    assert len(d.inputs_quote_ids) == 4  # three sharp prices + the Polymarket quote taken
    assert d.params["theta"] == 0.03 and d.params["sharp_source"] == "odds_api:pinnacle"


def test_pm_gap_does_nothing_when_the_gap_is_below_theta(store):
    sharp = (2.0, 3.6, 4.2)
    store.add_quotes(_book("odds_api:pinnacle", sharp))
    fair = power_devig_odds(sharp)
    store.add_quotes(_pm((round(fair[0] - 0.02, 2), 0.32, 0.27)))
    assert PmVsSharpGap(theta=0.03).decide(_snap(store)) == []


def test_pm_gap_falls_back_to_betfair_and_skips_stale_sharp_or_missing_asks(store):
    sharp = (2.0, 3.6, 4.2)
    store.add_quotes(_book("odds_api:betfair_ex_uk", sharp))
    fair = power_devig_odds(sharp)
    store.add_quotes(_pm((round(fair[0] - 0.06, 2), None, None)))
    [d] = PmVsSharpGap().decide(_snap(store))
    assert d.params["sharp_source"] == "odds_api:betfair_ex_uk"
    assert PmVsSharpGap(max_sharp_age_h=0.5).decide(_snap(store)) == []


def test_pm_gap_requires_enough_size_at_the_ask(store):
    sharp = (2.0, 3.6, 4.2)
    store.add_quotes(_book("odds_api:pinnacle", sharp))
    store.add_quotes(_pm((round(power_devig_odds(sharp)[0] - 0.06, 2), 0.3, 0.25), size=1.0))
    assert PmVsSharpGap(min_ask_size=5).decide(_snap(store)) == []


def test_run_decide_appends_once_per_fixture_and_strategy(store):
    store.add_quotes(_book("odds_api:pinnacle", (3.14, 2.34, 3.48)))
    strategies = [ControlFavPinnacle()]
    assert run_decide(store, strategies, now=NOW, code_version="sha") == {"control-fav-pinnacle": 1}
    assert run_decide(store, strategies, now=NOW + timedelta(minutes=5), code_version="sha") == {
        "control-fav-pinnacle": 0
    }
    assert len(store.decisions()) == 1


def test_available_strategies_lists_the_builtins():
    assert set(available_strategies()) >= {"control-fav-pinnacle", "pm-vs-sharp-gap"}


def test_paper_package_has_no_order_or_authenticated_trading_path():
    pkg = Path(__file__).resolve().parents[2] / "src" / "pitch_edge" / "paper"
    forbidden = re.compile(
        r"\.post\(|\.put\(|\.delete\(|/orders?\b|place_?order|create_?order|/portfolio|"
        r"KALSHI-ACCESS|POLY_API_KEY|private_key",
        re.IGNORECASE,
    )
    hits = [
        f"{p.name}:{i}"
        for p in pkg.rglob("*.py")
        for i, line in enumerate(p.read_text().splitlines(), 1)
        if forbidden.search(line)
    ]
    assert hits == []


class _Boom(ControlFavPinnacle):
    strategy_id = "boom"

    def decide(self, snapshot):
        raise RuntimeError("strategy bug")


def test_a_failing_strategy_does_not_stop_the_others(store):
    store.add_quotes(_book("odds_api:pinnacle", (3.14, 2.34, 3.48)))
    added = run_decide(store, [_Boom(), ControlFavPinnacle()], now=NOW, code_version="sha", code_hash="h")
    assert added == {"boom": 0, "control-fav-pinnacle": 1}
    log = store.decide_log().set_index("strategy_id")
    assert log.loc["boom", "error"].startswith("RuntimeError")


def test_decisions_carry_the_code_hash(store):
    store.add_quotes(_book("odds_api:pinnacle", (3.14, 2.34, 3.48)))
    run_decide(store, [ControlFavPinnacle()], now=NOW, code_version="sha", code_hash="abc123")
    assert store.decisions().iloc[0]["code_hash"] == "abc123"
