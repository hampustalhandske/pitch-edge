"""UCL / UEL settlement against the de-vigged OddsPapi Pinnacle close (mocked lake)."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from pitch_edge.paper.pricing import clv_log, power_devig_odds
from pitch_edge.paper.settle import PINNACLE_CLOSE_SOURCE, LakeCloseSource, settle
from pitch_edge.paper.store import Decision, PaperStore, Quote

pytestmark = pytest.mark.unit

KO = datetime(2026, 10, 21, 19, 0, tzinfo=UTC)
DECIDED = KO - timedelta(days=1)
SEL = ("home", "draw", "away")
CLOSE = (1.9, 3.8, 4.2)


class CountingLake:
    """A fake lake that answers every query with one fixed frame and records the queries."""

    def __init__(self, frame: pd.DataFrame):
        self.frame, self.calls = frame, []

    def sql(self, query, params=None):
        self.calls.append((query, params))
        return self.frame


def _frame(fixtures=None, **kw):
    """OddsPapi window rows: one row per fixture and outcome. `fixtures` = [(id, home, away, kickoff)]."""
    fixtures = fixtures or [("f1", "Paris Saint-Germain", "FC Barcelona", KO)]
    odds = kw.get("odds", CLOSE)
    age = kw.get("age", timedelta(minutes=3))
    active = kw.get("active", (True, True, True))
    drop = kw.get("drop", ())
    rows = []
    for fid, home, away, ko in fixtures:
        for o, price, act in zip(SEL, odds, active, strict=True):
            if o in drop:
                continue
            rows.append((fid, home, away, pd.Timestamp(ko), o, price, act, pd.Timestamp(ko - age)))
    cols = ["fixture_id", "home_team", "away_team", "kickoff_utc", "outcome", "price", "active", "observed_at"]
    return pd.DataFrame(rows, columns=cols)


def _close(frame, home="Paris Saint Germain", away="Barcelona", ko=KO):
    return LakeCloseSource(CountingLake(frame)).pinnacle_close("UCL", ko, home, away)


def test_matched_fixture_returns_the_latest_pinnacle_close():
    assert _close(_frame()).odds == CLOSE


@pytest.mark.parametrize(
    ("paper", "oddspapi"),
    [
        (("Paris Saint Germain", "Barcelona"), ("Paris Saint-Germain", "FC Barcelona")),
        (("Bodø/Glimt", "Lille"), ("Bodoe/Glimt", "Lille OSC")),
        (("Inter Milan", "Feyenoord"), ("Inter Milano", "Feyenoord Rotterdam")),
        (("Sporting Lisbon", "Shakhtar Donetsk"), ("Sporting CP", "FC Shakhtar Donetsk")),
    ],
)
def test_team_name_variants_match(paper, oddspapi):
    frame = _frame([("f1", oddspapi[0], oddspapi[1], KO)])
    assert _close(frame, *paper).odds == CLOSE


def test_kickoff_outside_the_tolerance_does_not_match():
    got = _close(_frame(), ko=KO + timedelta(hours=5))
    assert got.odds is None and got.retry


def test_no_fixture_found_is_a_retry_not_a_verdict():
    got = _close(_frame([("f2", "Real Madrid", "RB Leipzig", KO)]))
    assert got.odds is None and got.retry and "no OddsPapi UCL fixture matches" in got.reason
    empty = _close(_frame().iloc[0:0])
    assert empty.odds is None and empty.retry


@pytest.mark.parametrize(
    ("kw", "why"),
    [
        ({"age": timedelta(minutes=16)}, "stale"),
        ({"drop": ("draw",)}, "lacks draw"),
        ({"active": (True, False, True)}, "inactive"),
        ({"odds": (1.9, 0.0, 4.2)}, "price <= 1"),
    ],
)
def test_invalid_close_gives_no_benchmark_and_says_why(kw, why):
    got = _close(_frame(**kw))
    assert got.odds is None and not got.retry and why in got.reason


def test_close_exactly_fifteen_minutes_before_kickoff_is_valid():
    assert _close(_frame(age=timedelta(minutes=15))).odds == CLOSE


def test_two_matching_fixtures_are_refused():
    two = [("f1", "Paris Saint-Germain", "FC Barcelona", KO), ("f2", "Paris Saint-Germain", "FC Barcelona", KO)]
    got = _close(_frame(two))
    assert got.odds is None and not got.retry and "2 OddsPapi UCL fixtures" in got.reason


def test_lake_is_asked_once_per_competition_and_day():
    lake = CountingLake(_frame())
    source = LakeCloseSource(lake)
    for _ in range(3):
        source.pinnacle_close("UCL", KO, "Paris Saint Germain", "Barcelona")
    source.pinnacle_close("UCL", KO + timedelta(hours=1), "Real Madrid", "RB Leipzig")
    assert len(lake.calls) == 1
    assert lake.calls[0][1][0] == "UCL"
    source.pinnacle_close("UCL", KO + timedelta(days=1), "Real Madrid", "RB Leipzig")
    assert len(lake.calls) == 2


@pytest.fixture
def store(tmp_path):
    s = PaperStore(tmp_path / "paper.duckdb")
    yield s
    s.close()


def _seed(
    store,
    i=0,
    venue="odds_api:williamhill",
    selection="home",
    price=2.1,
    home="Paris Saint Germain",
    away="Barcelona",
    ko=KO,
):
    q = Quote(
        source=venue if venue == "polymarket" else "odds_api:pinnacle",
        source_event_id=f"e{i}",
        home=home,
        away=away,
        league="UCL",
        kickoff_at=ko,
        selection=selection,
        captured_at=ko - timedelta(days=1),
        decimal_odds=2.0,
    )
    store.add_quotes([q])
    fk = store.quotes().query("source_event_id == @q.source_event_id")["fixture_key"].iloc[0]
    d = Decision(
        strategy_id=f"s{i}",
        hypothesis_id="control",
        decided_at=ko - timedelta(days=1),
        fixture_key=fk,
        kickoff_at=ko,
        venue=venue,
        selection=selection,
        price_taken=price,
        paper_stake_units=1.0,
        model_prob=0.5,
        inputs_quote_ids=[store.quotes()["quote_id"].iloc[-1]],
        code_version="sha",
    )
    store.append_decision(d)
    return d


def test_ucl_decision_settles_against_the_devigged_pinnacle_close_without_waiting_for_a_result(store):
    d = _seed(store)
    lake = CountingLake(_frame())
    assert settle(store, LakeCloseSource(lake), now=KO + timedelta(hours=4)) == {"settled": 1, "waiting": 0}
    s = store.settlements().iloc[0]
    fair = 1 / power_devig_odds(CLOSE)[0]
    assert s["closing_source"] == PINNACLE_CLOSE_SOURCE and s["sharp_source"] == PINNACLE_CLOSE_SOURCE
    assert s["closing_price"] == 1.9 and s["fair_close_odds"] == pytest.approx(fair)
    assert s["clv_log"] == pytest.approx(math.log(d.price_taken / fair)) == pytest.approx(clv_log(2.1, fair))
    assert pd.isna(s["result"]) and pd.isna(s["pnl"])
    assert store.decision_flags().empty


def test_settle_asks_the_lake_once_for_many_ucl_decisions_of_a_day(store):
    for i in range(4):
        _seed(store, i)
    lake = CountingLake(_frame())
    assert settle(store, LakeCloseSource(lake), now=KO + timedelta(hours=4))["settled"] == 4
    assert len(lake.calls) == 1


def test_implausible_book_sum_is_no_benchmark(store):
    d = _seed(store)
    lake = CountingLake(_frame(odds=(1.2, 2.0, 3.0)))
    now = KO + timedelta(days=5)
    assert settle(store, LakeCloseSource(lake), now=now)["settled"] == 0
    flag = store.decision_flags().iloc[0]
    assert flag["decision_id"] == d.decision_id and flag["flag"] == "no_benchmark" and "book sum" in flag["reason"]


def test_invalid_close_waits_inside_grace_then_flags_no_benchmark_once(store):
    d = _seed(store)
    lake = CountingLake(_frame(age=timedelta(minutes=40)))
    assert settle(store, LakeCloseSource(lake), now=KO + timedelta(hours=4)) == {"settled": 0, "waiting": 1}
    assert store.decision_flags().empty
    after = KO + timedelta(days=5)
    settle(store, LakeCloseSource(lake), now=after)
    settle(store, LakeCloseSource(lake), now=after + timedelta(hours=1))
    flags = store.decision_flags()
    assert len(flags) == 1 and flags.iloc[0]["decision_id"] == d.decision_id and "stale" in flags.iloc[0]["reason"]
    assert len(store.provisional_settlements()) == 1 and store.settlements().empty


def test_a_fixture_not_listed_yet_stays_waiting_and_is_never_flagged_before_the_horizon(store):
    _seed(store)
    empty = CountingLake(_frame().iloc[0:0])
    for now in (KO + timedelta(hours=4), KO + timedelta(days=5), KO + timedelta(days=20)):
        settle(store, LakeCloseSource(empty), now=now)
    assert store.decision_flags().empty and store.settlements().empty
    # the fixture appears later: the decision settles finally and supersedes the provisional row
    assert settle(store, LakeCloseSource(CountingLake(_frame())), now=KO + timedelta(days=21))["settled"] == 1
    assert store.settlements().iloc[0]["closing_source"] == PINNACLE_CLOSE_SOURCE


def test_no_valid_close_by_the_final_horizon_is_flagged_no_benchmark(store):
    d = _seed(store)
    empty = CountingLake(_frame().iloc[0:0])
    settle(store, LakeCloseSource(empty), now=KO + timedelta(days=31))
    settle(store, LakeCloseSource(empty), now=KO + timedelta(days=32))
    flags = store.decision_flags()
    assert (
        len(flags) == 1 and flags.iloc[0]["decision_id"] == d.decision_id and "final horizon" in flags.iloc[0]["reason"]
    )
    assert empty.calls == []


def test_polymarket_ucl_decision_keeps_its_bar_close_and_records_the_pinnacle_sharp_source(store):
    _seed(store, venue="polymarket")
    q = store.quotes()
    assert q["source"].iloc[0] == "polymarket"
    lake = CountingLake(_frame())
    settle(store, LakeCloseSource(lake), now=KO + timedelta(days=5))
    assert store.settlements().empty or store.settlements().iloc[0]["closing_source"] != PINNACLE_CLOSE_SOURCE
    assert store.decision_flags().empty


def test_a_listed_fixture_without_pinnacle_rows_is_retried():
    frame = _frame().iloc[:1].copy()
    frame[["outcome", "price", "active", "observed_at"]] = None  # a LEFT JOIN row with no odds
    got = _close(frame)
    assert got.odds is None and got.retry and "no Pinnacle 1x2 rows" in got.reason


def test_a_latest_row_with_price_zero_is_no_close_even_if_an_earlier_row_was_fine():
    # the lake query returns only the latest row per outcome; a 0.0 latest row must not fall back to an older one
    got = _close(_frame(odds=(0.0, 3.8, 4.2)))
    assert got.odds is None and not got.retry and "home" in got.reason
