"""Price math for the paper ledger: prediction-market fee conversion, power de-vig, CLV and P&L."""

from __future__ import annotations

import math

import pytest

from pitch_edge.paper.pricing import (
    clv_log,
    paper_pnl,
    power_devig_odds,
    power_normalise_probs,
    prediction_market_decimal_odds,
    taker_fee,
)

pytestmark = pytest.mark.unit


def test_polymarket_fee_is_rate_times_p_times_one_minus_p():
    assert taker_fee(0.5, rate=0.05) == pytest.approx(0.0125)
    assert taker_fee(0.2, rate=0.05) == pytest.approx(0.05 * 0.2 * 0.8)


def test_prediction_market_odds_include_the_fee_on_top_of_the_ask():
    # ask 0.56 costs 0.56 + 0.05 * 0.56 * 0.44 = 0.57232 per share that pays 1
    assert prediction_market_decimal_odds(0.56, fee_rate=0.05) == pytest.approx(1 / 0.57232)
    # without a fee it is just 1 / ask
    assert prediction_market_decimal_odds(0.25, fee_rate=0.0) == pytest.approx(4.0)


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.2, float("nan")])
def test_prediction_market_odds_reject_prices_outside_the_open_unit_interval(bad):
    with pytest.raises(ValueError):
        prediction_market_decimal_odds(bad, fee_rate=0.05)


def test_power_devig_returns_probabilities_summing_to_one_and_preserves_order():
    probs = power_devig_odds([3.14, 2.34, 3.48])
    assert sum(probs) == pytest.approx(1.0, abs=1e-9)
    assert probs[1] > probs[0] > probs[2]


def test_power_devig_is_identity_for_a_fair_book():
    probs = power_devig_odds([2.0, 4.0, 4.0])
    assert probs == pytest.approx([0.5, 0.25, 0.25], abs=1e-9)


def test_power_devig_takes_more_margin_off_longshots_than_favourites():
    odds = [1.5, 4.2, 7.0]
    raw = [1 / o for o in odds]
    proportional = [r / sum(raw) for r in raw]
    power = power_devig_odds(odds)
    # power method: favourite gets a larger fair share than under proportional normalisation
    assert power[0] > proportional[0]
    assert power[2] < proportional[2]


def test_power_normalise_handles_an_underround_set_of_prices():
    probs = power_normalise_probs([0.30, 0.25, 0.40])  # sums to 0.95
    assert sum(probs) == pytest.approx(1.0, abs=1e-9)


def test_clv_log_is_log_ratio_of_taken_to_fair_close():
    assert clv_log(2.2, 2.0) == pytest.approx(math.log(1.1))
    assert clv_log(2.0, 2.0) == 0.0
    assert clv_log(1.8, 2.0) < 0


def test_paper_pnl_win_loss_and_unknown():
    assert paper_pnl(stake=1.0, price=2.5, won=True) == pytest.approx(1.5)
    assert paper_pnl(stake=2.0, price=2.5, won=False) == pytest.approx(-2.0)
    assert paper_pnl(stake=1.0, price=2.5, won=None) is None
