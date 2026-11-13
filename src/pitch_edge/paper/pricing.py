"""Price conversions used by the paper ledger.

- `taker_fee` / `prediction_market_decimal_odds`: turn a prediction-market ask price (dollars per
  share that pays $1) into the decimal odds a taker actually gets once the fee is added. Polymarket
  sports fee: `rate * p * (1 - p)` per share, rate 0.05 (docs.polymarket.com/trading/fees).
- `power_devig_odds` / `power_normalise_probs`: the power method for removing the bookmaker margin
  (find k with sum(p_i ** k) = 1, fair p_i = p_i ** k). It takes relatively more margin off longshots
  than proportional normalisation, matching the favourite-longshot bias in 1X2 prices.
- `clv_log`: closing-line value of one bet, ln(price taken / fair closing odds).
- `paper_pnl`: profit of a paper stake given the outcome.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from scipy.optimize import brentq

POLYMARKET_FEE_RATE = 0.05
# Kalshi's published taker fee is 0.07 * C * P * (1 - P) per order, rounded up to the cent. The fee
# schedule PDF could not be fetched when this was written, so treat this constant as unverified.
KALSHI_FEE_RATE = 0.07


def taker_fee(p: float, rate: float = POLYMARKET_FEE_RATE) -> float:
    """Fee per share for a taker buying at price `p` (dollars per $1-payout share)."""
    return rate * p * (1.0 - p)


def prediction_market_decimal_odds(p_ask: float, fee_rate: float = POLYMARKET_FEE_RATE) -> float:
    """Decimal odds of buying one $1-payout share at `p_ask` plus the taker fee: 1 / (p + fee(p)).

    Raises ValueError unless 0 < p_ask < 1.
    """
    if not (0.0 < p_ask < 1.0):
        raise ValueError(f"prediction-market price must be in (0, 1), got {p_ask}")
    return 1.0 / (p_ask + taker_fee(p_ask, fee_rate))


def power_normalise_probs(probs: Sequence[float]) -> list[float]:
    """Rescale implied probabilities to sum to 1 with the power method (works for over- and underround)."""
    ps = [float(p) for p in probs]
    if not ps or any(not (0.0 < p < 1.0) for p in ps):
        raise ValueError(f"implied probabilities must all be in (0, 1), got {ps}")
    if abs(sum(ps) - 1.0) < 1e-12:
        return ps

    def gap(k: float) -> float:
        return sum(p**k for p in ps) - 1.0

    k = brentq(gap, 1e-3, 100.0, xtol=1e-14)
    return [p**k for p in ps]


def power_devig_odds(odds: Sequence[float]) -> list[float]:
    """Fair probabilities from a full set of decimal odds (e.g. home/draw/away) via the power method."""
    if any(o <= 1.0 for o in odds):
        raise ValueError(f"decimal odds must be > 1, got {list(odds)}")
    return power_normalise_probs([1.0 / o for o in odds])


def clv_log(price_taken: float, fair_close_odds: float) -> float:
    """Closing-line value: ln(price_taken / fair_close_odds); positive means we beat the close."""
    return math.log(price_taken / fair_close_odds)


def paper_pnl(stake: float, price: float, won: bool | None) -> float | None:
    """Profit of a back bet at decimal odds `price`: stake*(price-1) if won, -stake if lost, None if unknown."""
    if won is None:
        return None
    return stake * (price - 1.0) if won else -stake
