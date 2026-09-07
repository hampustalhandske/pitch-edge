"""Fractional Kelly staking and Monte Carlo bankroll simulation.

Full Kelly is theoretically growth-optimal but has ruinous variance under
model uncertainty, so the industry standard (and what recruiters expect to
see) is *fractional* Kelly — typically 1/4 to 1/2 Kelly — applied against a
paper bankroll only, per the project's no-auto-bet constraint.
"""

from __future__ import annotations

import numpy as np


def kelly_fraction(model_probability: float, decimal_odds: float) -> float:
    """Full-Kelly optimal stake fraction of bankroll for a single bet.

    f* = (b*p - q) / b, where b = decimal_odds - 1, p = win prob, q = 1 - p.
    Returns 0.0 (no bet) when the edge is non-positive.
    """
    if not (0.0 < model_probability < 1.0):
        raise ValueError(f"model_probability must be in (0, 1), got {model_probability}")
    if decimal_odds <= 1.0:
        raise ValueError(f"decimal_odds must be > 1.0, got {decimal_odds}")

    b = decimal_odds - 1.0
    q = 1.0 - model_probability
    f_star = (b * model_probability - q) / b
    return max(f_star, 0.0)


def fractional_kelly_stake(
    model_probability: float,
    decimal_odds: float,
    bankroll: float,
    fraction: float = 0.25,
    max_stake_pct: float = 0.05,
) -> float:
    """Stake size in bankroll currency units, capped at max_stake_pct of bankroll
    as a hard risk-management ceiling regardless of what Kelly suggests."""
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    f_star = kelly_fraction(model_probability, decimal_odds)
    stake_pct = min(f_star * fraction, max_stake_pct)
    return stake_pct * bankroll


def simulate_bankroll_path(
    probabilities: np.ndarray,
    decimal_odds: np.ndarray,
    outcomes: np.ndarray,
    starting_bankroll: float = 1000.0,
    fraction: float = 0.25,
    max_stake_pct: float = 0.05,
) -> np.ndarray:
    """Sequential fractional-Kelly bankroll simulation over a bet sequence.

    outcomes: 1 if the bet won, 0 if it lost. Returns the bankroll path
    (length len(outcomes) + 1, including the starting value).
    """
    n = len(probabilities)
    if not (len(decimal_odds) == len(outcomes) == n):
        raise ValueError("probabilities, decimal_odds, and outcomes must have equal length")

    path = np.empty(n + 1)
    path[0] = starting_bankroll
    bankroll = starting_bankroll
    for i in range(n):
        if bankroll <= 0:
            path[i + 1 :] = 0.0
            break
        stake = fractional_kelly_stake(
            probabilities[i], decimal_odds[i], bankroll, fraction=fraction, max_stake_pct=max_stake_pct
        )
        if outcomes[i]:
            bankroll += stake * (decimal_odds[i] - 1.0)
        else:
            bankroll -= stake
        path[i + 1] = bankroll
    return path


def monte_carlo_ruin_risk(
    probabilities: np.ndarray,
    decimal_odds: np.ndarray,
    starting_bankroll: float = 1000.0,
    fraction: float = 0.25,
    max_stake_pct: float = 0.05,
    n_simulations: int = 1000,
    ruin_threshold_pct: float = 0.2,
    seed: int = 42,
) -> dict[str, float]:
    """Resample bet outcomes (from the model's own claimed probabilities) many
    times to estimate drawdown/ruin risk of a fractional-Kelly staking plan.

    ruin_threshold_pct: bankroll fraction below which we count a path as "ruined".
    """
    rng = np.random.default_rng(seed)
    n_bets = len(probabilities)
    final_bankrolls = np.empty(n_simulations)
    max_drawdowns = np.empty(n_simulations)
    ruin_count = 0

    for sim in range(n_simulations):
        outcomes = rng.random(n_bets) < probabilities
        path = simulate_bankroll_path(
            probabilities,
            decimal_odds,
            outcomes.astype(int),
            starting_bankroll=starting_bankroll,
            fraction=fraction,
            max_stake_pct=max_stake_pct,
        )
        final_bankrolls[sim] = path[-1]
        running_max = np.maximum.accumulate(path)
        drawdown = (running_max - path) / np.where(running_max == 0, 1, running_max)
        max_drawdowns[sim] = drawdown.max()
        if path.min() <= starting_bankroll * ruin_threshold_pct:
            ruin_count += 1

    return {
        "median_final_bankroll": float(np.median(final_bankrolls)),
        "p05_final_bankroll": float(np.percentile(final_bankrolls, 5)),
        "p95_final_bankroll": float(np.percentile(final_bankrolls, 95)),
        "mean_max_drawdown": float(np.mean(max_drawdowns)),
        "ruin_probability": ruin_count / n_simulations,
    }
