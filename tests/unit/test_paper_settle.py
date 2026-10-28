"""Settlement: closing-source preference, CLV with power de-vig, P&L, grace period, and the lake adapter."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from pitch_edge.paper.pricing import power_devig_odds, power_normalise_probs
from pitch_edge.paper.settle import FdClose, LakeCloseSource, settle
from pitch_edge.paper.store import Decision, PaperStore, Quote

pytestmark = pytest.mark.unit

KO = datetime(2026, 10, 24, 14, 0, tzinfo=UTC)
DECIDED = KO - timedelta(days=3)
SEL = ("home", "draw", "away")


class FakeSource:
    def __init__(self, fd: FdClose | None = None, bars: dict[str, float] | None = None):
        self.fd, self.bars = fd, bars or {}
        self.bar_calls: list[tuple[list[str], datetime]] = []

    def fd_match(self, league, kickoff_at, home, away):
        return self.fd

    def pm_last_bars(self, asset_ids, before):
        self.bar_calls.append((list(asset_ids), before))
        return {a: self.bars[a] for a in asset_ids if a in self.bars}


@pytest.fixture
def store(tmp_path):
    s = PaperStore(tmp_path / "paper.duckdb")
    yield s
    s.close()


def _seed(store, venue="odds_api:pinnacle", selection="home", price=2.2, late=True, late_kickoff=None):
    pin = [
        Quote(
            source="odds_api:pinnacle",
            source_event_id="e1",
            home="Ipswich Town",
            away="Nottingham Forest",
            league="EPL",
            kickoff_at=KO,
            selection=s,
            captured_at=DECIDED - timedelta(hours=1),
            decimal_odds=o,
        )
        for s, o in zip(SEL, (2.1, 3.4, 3.9), strict=True)
    ]
    late_pin = [
        Quote(
            source="odds_api:pinnacle",
            source_event_id="e1",
            home="Ipswich Town",
            away="Nottingham Forest",
            league="EPL",
            kickoff_at=late_kickoff or KO,
            selection=s,
            captured_at=KO - timedelta(hours=2),
            decimal_odds=o,
        )
        for s, o in zip(SEL, (2.0, 3.5, 4.1), strict=True)
    ]
    pm = [
        Quote(
            source="polymarket",
            source_event_id="pm1",
            home="Ipswich Town FC",
            away="Nottingham Forest FC",
            league="EPL",
            kickoff_at=KO,
            selection=s,
            captured_at=DECIDED - timedelta(minutes=5),
            bid=a - 0.02,
            ask=a,
            size=50,
            instrument_id=f"tok-{s}",
        )
        for s, a in zip(SEL, (0.44, 0.28, 0.26), strict=True)
    ]
    store.add_quotes(pin + (late_pin if late else []) + pm)
    q = store.quotes()
    inputs = list(
        q[
            (q["source"] == ("polymarket" if venue == "polymarket" else venue))
            & (q["captured_at"] <= pd.Timestamp(DECIDED))
        ]["quote_id"]
    )
    d = Decision(
        strategy_id="s",
        hypothesis_id="control",
        decided_at=DECIDED,
        fixture_key=q["fixture_key"].iloc[0],
        kickoff_at=KO,
        venue=venue,
        selection=selection,
        price_taken=price,
        paper_stake_units=1.0,
        model_prob=0.45,
        inputs_quote_ids=inputs,
        code_version="sha",
    )
    store.append_decision(d)
    return d


def test_bookmaker_decision_settles_against_betfair_close_from_the_lake(store):
    d = _seed(store)
    close = (2.0, 3.6, 4.4)
    n = settle(store, FakeSource(FdClose(betfair_close=close, result="home")), now=KO + timedelta(days=1))
    assert n == {"settled": 1, "waiting": 0}
    s = store.settlements().iloc[0]
    fair_odds = 1 / power_devig_odds(close)[0]
    assert s["closing_source"] == "lake:fd_matches.betfair_ex_close"
    assert s["closing_price"] == 2.0 and s["fair_close_odds"] == pytest.approx(fair_odds)
    assert s["clv_log"] == pytest.approx(math.log(d.price_taken / fair_odds))
    assert s["result"] == "home" and s["pnl"] == pytest.approx(1.2)


def test_losing_bet_has_negative_unit_pnl(store):
    _seed(store, selection="draw", price=3.4)
    settle(store, FakeSource(FdClose(betfair_close=(2.0, 3.6, 4.4), result="away")), now=KO + timedelta(days=1))
    assert store.settlements().iloc[0]["pnl"] == -1.0


def test_polymarket_decision_settles_against_the_last_pre_kickoff_bar(store):
    d = _seed(store, venue="polymarket", selection="home", price=2.2)
    bars = {"tok-home": 0.47, "tok-draw": 0.27, "tok-away": 0.28}
    src = FakeSource(FdClose(betfair_close=(2.0, 3.6, 4.4), result="home"), bars)
    settle(store, src, now=KO + timedelta(days=1))
    s = store.settlements().iloc[0]
    fair_p = power_normalise_probs([0.47, 0.27, 0.28])[0]
    assert s["closing_source"] == "lake:pm_soccer_bars_1h"
    assert s["closing_price"] == pytest.approx(1 / 0.47)
    assert s["clv_log"] == pytest.approx(math.log(d.price_taken * fair_p))
    assert s["sharp_fair_close_odds"] == pytest.approx(1 / power_devig_odds((2.0, 3.6, 4.4))[0])
    assert sorted(src.bar_calls[0][0]) == ["tok-away", "tok-draw", "tok-home"] and src.bar_calls[0][1] == KO


def test_waits_for_the_lake_close_inside_the_grace_period(store):
    _seed(store)
    assert settle(store, FakeSource(None), now=KO + timedelta(days=1), grace=timedelta(days=4)) == {
        "settled": 0,
        "waiting": 1,
    }
    assert store.settlements().empty


def test_falls_back_to_our_own_last_pre_kickoff_snapshot_after_grace_and_labels_it(store):
    d = _seed(store)
    settle(store, FakeSource(None), now=KO + timedelta(days=5), grace=timedelta(days=4))
    assert store.settlements().empty  # only a lake-sourced close is final
    s = store.provisional_settlements().iloc[0]
    assert s["closing_source"].startswith("own_snapshot:odds_api:pinnacle@")
    fair_odds = 1 / power_devig_odds((2.0, 3.5, 4.1))[0]  # the latest capture before kickoff
    assert s["fair_close_odds"] == pytest.approx(fair_odds)
    assert s["clv_log"] == pytest.approx(math.log(d.price_taken / fair_odds))
    assert pd.isna(s["result"]) and pd.isna(s["pnl"])


def test_does_not_settle_before_the_match_is_over(store):
    _seed(store)
    assert settle(store, FakeSource(FdClose((2.0, 3.6, 4.4), "home")), now=KO + timedelta(hours=1)) == {
        "settled": 0,
        "waiting": 0,
    }


class FakeLake:
    def __init__(self, frames):
        self.frames, self.calls = frames, []

    def sql(self, query, params=None):
        self.calls.append((query, params))
        return self.frames.pop(0)


def test_lake_adapter_matches_fd_rows_by_team_name_and_maps_the_result():
    fd = pd.DataFrame(
        {
            "match_date": [pd.Timestamp("2026-10-24")] * 2,
            "home_team": ["Ipswich", "Arsenal"],
            "away_team": ["Nott'm Forest", "Everton"],
            "ft_result": ["A", "H"],
            "betfair_ex_home_close": [2.0, 1.4],
            "betfair_ex_draw_close": [3.6, 5.0],
            "betfair_ex_away_close": [4.4, 9.0],
        }
    )
    lake = FakeLake([fd])
    got = LakeCloseSource(lake).fd_match("EPL", KO, "Ipswich Town", "Nottingham Forest")
    assert got == FdClose(betfair_close=(2.0, 3.6, 4.4), result="away")
    assert lake.calls[0][1][0] == "E0"


def test_lake_adapter_returns_none_for_leagues_outside_football_data():
    lake = FakeLake([])
    assert LakeCloseSource(lake).fd_match("UCL", KO, "A", "B") is None
    assert lake.calls == []


def test_lake_adapter_reads_last_bar_known_before_kickoff():
    lake = FakeLake([pd.DataFrame({"asset_id": ["t1", "t2"], "close": [0.4, 0.3]})])
    assert LakeCloseSource(lake).pm_last_bars(["t1", "t2"], KO) == {"t1": 0.4, "t2": 0.3}
    query, params = lake.calls[0]
    assert "INTERVAL 1 HOUR" in query and params[-1] == KO.isoformat()


def test_fallback_snapshot_must_postdate_the_decision_else_no_close(store):
    _seed(store, late=False)  # the only Pinnacle snapshot is the one the decision itself used
    settle(store, FakeSource(None), now=KO + timedelta(days=5), grace=timedelta(days=4))
    s = store.provisional_settlements().iloc[0]
    assert s["closing_source"] == "none" and pd.isna(s["clv_log"])


def test_a_later_lake_close_supersedes_a_provisional_settlement(store):
    d = _seed(store)
    settle(store, FakeSource(None), now=KO + timedelta(days=5), grace=timedelta(days=4))
    assert settle(store, FakeSource(None), now=KO + timedelta(days=6)) == {"settled": 0, "waiting": 0}
    assert len(store.provisional_settlements()) == 1  # written once
    settle(store, FakeSource(FdClose(betfair_close=(2.0, 3.6, 4.4), result="home")), now=KO + timedelta(days=7))
    final = store.settlements()
    assert list(final["decision_id"]) == [d.decision_id]
    assert final.iloc[0]["closing_source"] == "lake:fd_matches.betfair_ex_close"


def test_lake_close_is_not_sought_beyond_the_final_horizon(store):
    _seed(store)
    src = FakeSource(FdClose(betfair_close=(2.0, 3.6, 4.4), result="home"))
    settle(store, src, now=KO + timedelta(days=40), final_horizon=timedelta(days=30))
    assert store.settlements().empty and len(store.provisional_settlements()) == 1


def test_a_moved_kickoff_voids_the_decision(store):
    d = _seed(store, late_kickoff=KO + timedelta(days=3))  # a later quote lists a new kickoff: postponed
    n = settle(store, FakeSource(FdClose(betfair_close=(2.0, 3.6, 4.4), result="home")), now=KO + timedelta(days=1))
    assert n["settled"] == 0
    flags = store.decision_flags()
    assert list(flags["decision_id"]) == [d.decision_id] and flags.iloc[0]["flag"] == "void"
    assert "kickoff" in flags.iloc[0]["reason"]
    assert store.settlements().empty and store.unsettled_decisions().empty


@pytest.mark.parametrize("close", [(1.5, 2.5, 3.0), (3.0, 4.5, 6.0)])
def test_closing_prices_with_an_implausible_book_sum_are_rejected(store, close):
    _seed(store)
    # 1/1.5+1/2.5+1/3 = 1.40 (> 1.05) and 1/3+1/4.5+1/6 = 0.72 (< 0.95): data errors, not a close
    assert settle(store, FakeSource(FdClose(betfair_close=close, result="home")), now=KO + timedelta(days=1)) == {
        "settled": 0,
        "waiting": 1,
    }


def _fd_window_frame():
    return pd.DataFrame(
        {
            "match_date": [pd.Timestamp("2026-10-24")] * 2,
            "home_team": ["Ipswich", "Arsenal"],
            "away_team": ["Nott'm Forest", "Everton"],
            "ft_result": ["A", "H"],
            "betfair_ex_home_close": [2.0, 1.4],
            "betfair_ex_draw_close": [3.6, 5.0],
            "betfair_ex_away_close": [4.4, 9.0],
        }
    )


def test_lake_adapter_queries_each_division_and_day_once():
    lake = FakeLake([_fd_window_frame()])
    source = LakeCloseSource(lake)
    first = source.fd_match("EPL", KO, "Ipswich Town", "Nottingham Forest")
    second = source.fd_match("EPL", KO + timedelta(hours=3), "Arsenal", "Everton")
    again = source.fd_match("EPL", KO, "Ipswich Town", "Nottingham Forest")
    assert first == again == FdClose(betfair_close=(2.0, 3.6, 4.4), result="away")
    assert second == FdClose(betfair_close=(1.4, 5.0, 9.0), result="home")
    assert len(lake.calls) == 1


def test_lake_adapter_does_not_retry_a_failed_window_within_one_settle():
    class DownLake:
        calls = 0

        def sql(self, query, params=None):
            DownLake.calls += 1
            raise RuntimeError("lake down")

    source = LakeCloseSource(DownLake())
    for _ in range(3):
        with pytest.raises(RuntimeError, match="lake down"):
            source.fd_match("EPL", KO, "Ipswich Town", "Nottingham Forest")
    assert DownLake.calls == 1


def test_lake_adapter_queries_the_same_token_set_once():
    lake = FakeLake([pd.DataFrame({"asset_id": ["t1", "t2"], "close": [0.4, 0.3]})])
    source = LakeCloseSource(lake)
    assert source.pm_last_bars(["t1", "t2"], KO) == source.pm_last_bars(["t2", "t1"], KO) == {"t1": 0.4, "t2": 0.3}
    assert len(lake.calls) == 1
