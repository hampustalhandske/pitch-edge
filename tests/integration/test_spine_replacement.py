"""Spine precedence is order-independent, and the backtester degrades honestly without Pinnacle prices."""

from __future__ import annotations

import pandas as pd
import pytest

from pitch_edge.backtest.engine import WalkForwardBacktester, WalkForwardConfig
from pitch_edge.data.ingest import store_matches
from pitch_edge.features.build import FeatureBuilder
from pitch_edge.models import DixonColesMatchModel

pytestmark = pytest.mark.integration


def _row(source: str, match_id: str, **extra) -> pd.DataFrame:
    base = {
        "match_id": match_id,
        "date": pd.Timestamp("2023-08-12"),
        "league": "England - Premier League",
        "league_code": "E0",
        "season": "2023/24",
        "home_team": "Arsenal",
        "away_team": "Man United",
        "home_goals": 2.0,
        "away_goals": 1.0,
        "source": source,
    }
    return pd.DataFrame([{**base, **extra}])


def test_football_data_row_replaces_secondary_row_and_keeps_elo(warehouse):
    secondary = _row(
        "club_football_match_data",
        "cfmd_E0_20230812_Arsenal_ManUnited",
        home_elo=1900.0,
        away_elo=1850.0,
        MktH=1.9,
        MktD=3.5,
        MktA=4.0,
    )
    assert store_matches(warehouse, secondary) == 1
    assert warehouse.count("odds") == 3

    spine = _row(
        "football_data_co_uk",
        "fd_E0_2324_20230812_Arsenal_ManUnited",
        PSH=1.85,
        PSD=3.7,
        PSA=4.4,
        PSCH=1.82,
        PSCD=3.75,
        PSCA=4.55,
    )
    assert store_matches(warehouse, spine) == 1  # net count unchanged: one deleted, one inserted
    m = warehouse.read("matches")
    assert len(m) == 1 and m.iloc[0]["source"] == "football_data_co_uk"
    assert m.iloc[0]["home_elo"] == 1900.0  # carried over from the replaced row
    odds = warehouse.read("odds")
    assert set(odds["bookmaker"]) == {"PS"} and len(odds) == 6  # old Mkt odds removed


def test_secondary_after_spine_is_deduped_not_inserted(warehouse):
    spine = _row("football_data_co_uk", "fd_1", PSH=1.85, PSD=3.7, PSA=4.4)
    store_matches(warehouse, spine)
    from pitch_edge.data.ingest import _attach_elo_to_spine

    dup = _row("club_football_match_data", "cfmd_1", home_elo=1700.0, away_elo=1600.0, home_form5=1.0, away_form5=2.0)
    _attach_elo_to_spine(warehouse, dup)
    m = warehouse.read("matches")
    assert len(m) == 1 and m.iloc[0]["home_elo"] == 1700.0


def test_engine_falls_back_to_market_average_price_when_no_pinnacle(synthetic_league_matches):
    m = synthetic_league_matches.rename(columns={"PSH": "MktH", "PSD": "MktD", "PSA": "MktA"}).drop(
        columns=["PSCH", "PSCD", "PSCA", "B365H", "B365D", "B365A"]
    )
    f = FeatureBuilder().build(m)
    bt = WalkForwardBacktester(WalkForwardConfig(min_train_matches=200, retrain_every_days=120, edge_threshold=0.0))
    res = bt.run(f, DixonColesMatchModel())
    assert bt.bet_price_source == "Mkt"
    assert bt.closing_price_source == "bet_price_no_closing_available"
    assert not res.bets.empty
    assert (res.bets["closing_odds"] == res.bets["bet_odds"]).all()  # CLV is zero by construction, and flagged
    assert (res.summaries["mean_clv_pct"].abs() < 1e-9).all()
    assert (res.summaries["bet_price_source"] == "Mkt").all()
