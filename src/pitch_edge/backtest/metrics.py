"""Backtest evaluation metrics: CLV, ROI, Sharpe, vig-adjusted returns.

CLV (closing line value) — not raw ROI — is the standard proof of edge in
the sports-trading literature, because ROI over a realistic sample size is
dominated by variance. A strategy with consistently positive CLV against
the closing line has a real edge even if a particular sample's ROI is flat
or negative.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def clv(bet_odds: np.ndarray, closing_odds: np.ndarray) -> np.ndarray:
    """Per-bet CLV as the percentage difference between the odds you got and
    the closing odds, in implied-probability terms:
        clv = (1/bet_odds - 1/closing_odds) is WRONG sign convention;
    we use the standard "odds beaten" formulation instead:
        clv_pct = closing_implied_prob / bet_implied_prob - 1
    Positive means you got better odds than the closing line implied you
    should have (i.e., the market moved against your side after you bet).
    """
    bet_odds = np.asarray(bet_odds, dtype=float)
    closing_odds = np.asarray(closing_odds, dtype=float)
    bet_implied = 1.0 / bet_odds
    closing_implied = 1.0 / closing_odds
    return closing_implied / bet_implied - 1.0


def roi(stakes: np.ndarray, returns: np.ndarray) -> float:
    """returns = amount won (0 for a loss, stake*(odds-1) for a win), net of stake
    already excluded — i.e. profit, not gross payout."""
    stakes = np.asarray(stakes, dtype=float)
    returns = np.asarray(returns, dtype=float)
    if stakes.sum() == 0:
        return 0.0
    return float(returns.sum() / stakes.sum())


def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """Per-bet Sharpe-style ratio of returns (not annualized; use consistent
    units for returns across the series being compared)."""
    returns = np.asarray(returns, dtype=float)
    excess = returns - risk_free_rate
    std = excess.std(ddof=1)
    if std < 1e-12:
        return 0.0
    return float(excess.mean() / std)


def max_drawdown(bankroll_path: np.ndarray) -> float:
    path = np.asarray(bankroll_path, dtype=float)
    running_max = np.maximum.accumulate(path)
    drawdown = (running_max - path) / np.where(running_max == 0, 1, running_max)
    return float(drawdown.max())


def summarize_backtest(results: pd.DataFrame) -> dict[str, float]:
    """results columns: stake, profit, bet_odds, closing_odds, won (0/1).

    Returns the standard scorecard: ROI, mean CLV, Sharpe, hit rate, N bets.
    """
    required = {"stake", "profit", "bet_odds", "closing_odds", "won"}
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"summarize_backtest missing columns: {sorted(missing)}")
    if results.empty:
        return {
            "n_bets": 0,
            "roi": 0.0,
            "mean_clv_pct": 0.0,
            "sharpe": 0.0,
            "hit_rate": 0.0,
            "total_profit": 0.0,
        }

    clv_pct = clv(results["bet_odds"].to_numpy(), results["closing_odds"].to_numpy())
    per_bet_return = (results["profit"] / results["stake"].replace(0, np.nan)).fillna(0.0).to_numpy()

    return {
        "n_bets": len(results),
        "roi": roi(results["stake"].to_numpy(), results["profit"].to_numpy()),
        "mean_clv_pct": float(np.mean(clv_pct)),
        "sharpe": sharpe_ratio(per_bet_return),
        "hit_rate": float(results["won"].mean()),
        "total_profit": float(results["profit"].sum()),
    }
