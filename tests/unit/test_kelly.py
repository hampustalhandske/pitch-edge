from __future__ import annotations

import numpy as np
import pytest

from pitch_edge.backtest.kelly import (
    fractional_kelly_stake,
    kelly_fraction,
    monte_carlo_ruin_risk,
    simulate_bankroll_path,
)

pytestmark = pytest.mark.unit


def test_kelly_fraction_positive_edge():
    # p=0.6, decimal odds=2.0 (fair coin priced at evens) -> clear edge
    f = kelly_fraction(0.6, 2.0)
    assert f == pytest.approx(0.2)


def test_kelly_fraction_no_edge_returns_zero():
    # p exactly matches implied probability -> zero edge -> zero stake
    f = kelly_fraction(0.5, 2.0)
    assert f == pytest.approx(0.0, abs=1e-9)


def test_kelly_fraction_negative_edge_clamped_to_zero():
    f = kelly_fraction(0.3, 2.0)
    assert f == 0.0


def test_kelly_fraction_rejects_invalid_probability():
    with pytest.raises(ValueError):
        kelly_fraction(0.0, 2.0)
    with pytest.raises(ValueError):
        kelly_fraction(1.0, 2.0)


def test_kelly_fraction_rejects_invalid_odds():
    with pytest.raises(ValueError):
        kelly_fraction(0.5, 1.0)


def test_fractional_kelly_stake_respects_cap():
    # Even with huge edge, stake should never exceed max_stake_pct of bankroll.
    stake = fractional_kelly_stake(0.95, 1.5, bankroll=1000.0, fraction=1.0, max_stake_pct=0.05)
    assert stake == pytest.approx(50.0)


def test_fractional_kelly_stake_scales_with_fraction():
    full = fractional_kelly_stake(0.6, 2.0, bankroll=1000.0, fraction=1.0, max_stake_pct=1.0)
    quarter = fractional_kelly_stake(0.6, 2.0, bankroll=1000.0, fraction=0.25, max_stake_pct=1.0)
    assert quarter == pytest.approx(full * 0.25)


def test_fractional_kelly_stake_rejects_invalid_fraction():
    with pytest.raises(ValueError):
        fractional_kelly_stake(0.6, 2.0, 1000.0, fraction=0.0)
    with pytest.raises(ValueError):
        fractional_kelly_stake(0.6, 2.0, 1000.0, fraction=1.5)


def test_simulate_bankroll_path_length_and_start():
    probs = np.array([0.6, 0.55, 0.5])
    odds = np.array([2.0, 2.0, 2.0])
    outcomes = np.array([1, 0, 1])
    path = simulate_bankroll_path(probs, odds, outcomes, starting_bankroll=1000.0)
    assert len(path) == 4
    assert path[0] == 1000.0


def test_simulate_bankroll_path_bust_floors_at_zero():
    # A guaranteed-loss sequence with max stake should never push bankroll negative.
    probs = np.array([0.9] * 10)
    odds = np.array([2.0] * 10)
    outcomes = np.zeros(10, dtype=int)  # lose every bet
    path = simulate_bankroll_path(probs, odds, outcomes, starting_bankroll=100.0, max_stake_pct=0.5)
    assert (path >= 0).all()


def test_simulate_bankroll_path_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        simulate_bankroll_path(np.array([0.5, 0.5]), np.array([2.0]), np.array([1, 0]))


def test_monte_carlo_ruin_risk_returns_expected_keys():
    probs = np.array([0.55] * 30)
    odds = np.array([2.0] * 30)
    result = monte_carlo_ruin_risk(probs, odds, n_simulations=100, seed=1)
    assert set(result) == {
        "median_final_bankroll",
        "p05_final_bankroll",
        "p95_final_bankroll",
        "mean_max_drawdown",
        "ruin_probability",
    }
    assert 0.0 <= result["ruin_probability"] <= 1.0


def test_monte_carlo_ruin_risk_is_deterministic_given_seed():
    probs = np.array([0.5] * 20)
    odds = np.array([2.0] * 20)
    r1 = monte_carlo_ruin_risk(probs, odds, n_simulations=50, seed=7)
    r2 = monte_carlo_ruin_risk(probs, odds, n_simulations=50, seed=7)
    assert r1 == r2
