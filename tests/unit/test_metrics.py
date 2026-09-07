from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pitch_edge.backtest.metrics import clv, max_drawdown, roi, sharpe_ratio, summarize_backtest

pytestmark = pytest.mark.unit


def test_clv_positive_when_line_moves_in_your_favour():
    # You bet at 2.20, closing price drifted to 2.00 (market shortened the
    # price you bet on) -> positive CLV.
    result = clv(np.array([2.20]), np.array([2.00]))
    assert result[0] > 0


def test_clv_negative_when_line_moves_against_you():
    result = clv(np.array([2.00]), np.array([2.20]))
    assert result[0] < 0


def test_clv_zero_when_bet_at_closing_price():
    result = clv(np.array([2.20, 3.5]), np.array([2.20, 3.5]))
    assert result == pytest.approx([0.0, 0.0])


def test_roi_basic():
    stakes = np.array([10.0, 10.0, 10.0])
    returns = np.array([10.0, -10.0, -10.0])  # won one (profit 10), lost two
    assert roi(stakes, returns) == pytest.approx(-10.0 / 30.0)


def test_roi_zero_stakes_returns_zero():
    assert roi(np.array([0.0, 0.0]), np.array([0.0, 0.0])) == 0.0


def test_sharpe_ratio_zero_std_returns_zero():
    assert sharpe_ratio(np.array([0.1, 0.1, 0.1])) == 0.0


def test_sharpe_ratio_positive_for_consistently_positive_returns():
    returns = np.array([0.1, 0.05, 0.15, 0.08])
    assert sharpe_ratio(returns) > 0


def test_max_drawdown_flat_path_is_zero():
    assert max_drawdown(np.array([100, 100, 100])) == 0.0


def test_max_drawdown_detects_dip():
    path = np.array([100, 150, 75, 120])
    # peak 150 -> trough 75 => 50% drawdown
    assert max_drawdown(path) == pytest.approx(0.5)


def test_summarize_backtest_missing_columns_raises():
    with pytest.raises(ValueError, match="missing columns"):
        summarize_backtest(pd.DataFrame({"stake": [1]}))


def test_summarize_backtest_empty_returns_zeros():
    empty = pd.DataFrame(columns=["stake", "profit", "bet_odds", "closing_odds", "won"])
    summary = summarize_backtest(empty)
    assert summary["n_bets"] == 0
    assert summary["roi"] == 0.0


def test_summarize_backtest_basic_scorecard():
    df = pd.DataFrame(
        {
            "stake": [10.0, 10.0],
            "profit": [8.0, -10.0],
            "bet_odds": [1.8, 2.5],
            "closing_odds": [1.7, 2.6],
            "won": [1, 0],
        }
    )
    summary = summarize_backtest(df)
    assert summary["n_bets"] == 2
    assert summary["hit_rate"] == pytest.approx(0.5)
    assert summary["total_profit"] == pytest.approx(-2.0)
    assert summary["roi"] == pytest.approx(-2.0 / 20.0)
