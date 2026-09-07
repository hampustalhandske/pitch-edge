from __future__ import annotations

import pandas as pd
import pytest

from pitch_edge.agents.risk import RiskLimits, RiskManager, RiskState

pytestmark = pytest.mark.unit


def _edges(n: int = 3, edge: float = 0.08, odds: float = 2.5) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "match_id": [f"m{i}" for i in range(n)],
            "date": pd.Timestamp("2024-03-01"),
            "home_team": "H",
            "away_team": "A",
            "outcome": "home",
            "model_probability": 1 / odds + edge,
            "market_probability": 1 / odds,
            "edge": edge,
            "decimal_odds": odds,
        }
    )


def test_sizes_positive_edges_and_caps_stake():
    rm = RiskManager(RiskLimits(max_stake_pct=0.02), RiskState(bankroll=1000))
    props = rm.size(_edges(), "gbdt")
    assert len(props) == 3
    assert all(p.stake <= 20.0 + 1e-9 for p in props)
    assert all("Kelly" in p.rationale for p in props)


def test_filters_low_edge_and_extreme_odds():
    rm = RiskManager(RiskLimits(min_edge=0.05, max_odds=5.0))
    assert rm.size(_edges(edge=0.02), "m") == []
    assert rm.size(_edges(odds=8.0), "m") == []


def test_daily_exposure_cap():
    rm = RiskManager(
        RiskLimits(max_stake_pct=0.05, max_daily_exposure_pct=0.06, max_match_exposure_pct=0.05),
        RiskState(bankroll=1000),
    )
    props = rm.size(_edges(n=5, edge=0.15, odds=2.0), "m")
    assert sum(p.stake for p in props) <= 60.0 + 1e-6


def test_drawdown_circuit_breaker():
    rm = RiskManager(RiskLimits(drawdown_halt_pct=0.2), RiskState(bankroll=700, peak_bankroll=1000))
    assert rm.halted()
    assert rm.size(_edges(), "m") == []


def test_settle_updates_peak():
    rm = RiskManager()
    rm.settle(+100)
    assert rm.state.bankroll == 1100 and rm.state.peak_bankroll == 1100
    rm.settle(-50)
    assert rm.state.peak_bankroll == 1100
