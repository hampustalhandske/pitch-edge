"""Removing the bookmaker margin ("de-vigging") from decimal odds of an n-outcome market.

A book's implied probabilities pi_i = 1 / odds_i sum to more than one (the overround). Each method
below maps them to fair probabilities p_i that sum to exactly one, differing in *where* the margin is
assumed to sit:

| method | assumption | parameter returned |
|---|---|---|
| `multiplicative` | margin proportional to pi_i: p_i = pi_i / sum(pi) | overround sum(pi) - 1 |
| `additive` | margin split equally: p_i = pi_i - (sum(pi) - 1) / n | overround sum(pi) - 1 |
| `power` | p_i = pi_i ** k, k solved so sum(p) = 1 (margin falls more on longshots) | exponent k |
| `shin` | Shin (1993) insider-trading model (margin falls more on longshots) | insider share z |

`research/STANDARDS.md` uses `power` for the sharp-close benchmark and `shin` as the sensitivity check.
All functions take an array of decimal odds shaped (rows, n_outcomes) — or a single row of shape
(n_outcomes,) — and are vectorised over rows. A row containing NaN or an infinite quote yields NaN
probabilities and a NaN parameter without affecting other rows; odds <= 1 raise ValueError.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import NamedTuple

import numpy as np
import pandas as pd

_MAX_ITER = 200


class DevigResult(NamedTuple):
    """Fair probabilities (same shape as the odds) and the method's per-row parameter (shape (rows,),
    or a 0-d array for a single row)."""

    probs: np.ndarray
    param: np.ndarray


def _implied(odds: Sequence[float] | Sequence[Sequence[float]] | np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    """Validate odds and return (implied probabilities 2-D, mask of complete rows, input-was-1-D)."""
    arr = np.asarray(odds, dtype=float)
    single = arr.ndim == 1
    if single:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"odds must be 1-D or 2-D, got shape {arr.shape}")
    if arr.shape[1] < 2:
        raise ValueError(f"a market needs at least two outcomes, got {arr.shape[1]}")
    finite = np.isfinite(arr)
    if np.any(arr[finite] <= 1.0):
        raise ValueError("decimal odds must be > 1.0")
    ok = finite.all(axis=1)
    pi = 1.0 / arr
    pi[~ok] = np.nan  # a row with a missing or infinite quote is not a complete market
    return pi, ok, single


def _pack(probs: np.ndarray, param: np.ndarray, single: bool) -> DevigResult:
    if single:
        return DevigResult(probs[0], np.asarray(param[0]))
    return DevigResult(probs, param)


def multiplicative(odds: Sequence[float] | Sequence[Sequence[float]] | np.ndarray) -> DevigResult:
    """Basic normalisation: p_i = pi_i / sum_j pi_j. Parameter: overround sum(pi) - 1."""
    pi, _, single = _implied(odds)
    total = pi.sum(axis=1, keepdims=True)
    return _pack(pi / total, total[:, 0] - 1.0, single)


def additive(odds: Sequence[float] | Sequence[Sequence[float]] | np.ndarray) -> DevigResult:
    """Equal margin per outcome: p_i = pi_i - (sum_j pi_j - 1) / n. Parameter: overround.

    Can return a negative probability for an extreme longshot in a high-margin book; it is returned as
    computed rather than clipped, so callers see that the method does not fit that row.
    """
    pi, _, single = _implied(odds)
    over = pi.sum(axis=1) - 1.0
    return _pack(pi - over[:, None] / pi.shape[1], over, single)


def power(odds: Sequence[float] | Sequence[Sequence[float]] | np.ndarray) -> DevigResult:
    """Power method: p_i = pi_i ** k with k solved so that sum_i pi_i ** k = 1.

    k > 1 for an overround book (k = 1 for a fair one, k < 1 for an underround one). Solved per row by
    Newton's method from k = 1 on f(k) = sum exp(k ln pi_i) - 1, which is convex and strictly
    decreasing, so the iteration converges monotonically. Reference: Vovk & Zhdanov (2009), "Prediction
    with expert advice for the Brier game", JMLR 10; Clarke, Kovalchik & Ingram (2017), "Adjusting
    bookmaker's odds to allow for overround", American Journal of Sports Science 5(6).
    """
    pi, ok, single = _implied(odds)
    log_pi = np.log(np.where(ok[:, None], pi, 0.5))
    k = np.ones(len(pi))
    for _ in range(_MAX_ITER):
        terms = np.exp(k[:, None] * log_pi)
        f = terms.sum(axis=1) - 1.0
        if np.all(np.abs(f) < 1e-15):
            break
        k = k - f / (terms * log_pi).sum(axis=1)
    probs = np.exp(k[:, None] * log_pi)
    probs = probs / probs.sum(axis=1, keepdims=True)  # remove the last ~1e-16 of residual
    probs[~ok] = np.nan
    k[~ok] = np.nan
    return _pack(probs, k, single)


def _shin_probs(z: np.ndarray, pi: np.ndarray, total: np.ndarray) -> np.ndarray:
    zc = z[:, None]
    return (np.sqrt(zc**2 + 4 * (1 - zc) * pi**2 / total) - zc) / (2 * (1 - zc))


def shin(odds: Sequence[float] | Sequence[Sequence[float]] | np.ndarray) -> DevigResult:
    """Shin's method: fair probabilities implied by a book that protects itself against a share z of
    insider money.

        p_i(z) = (sqrt(z^2 + 4 (1 - z) pi_i^2 / S) - z) / (2 (1 - z)),  S = sum_j pi_j

    with z the root of sum_i p_i(z) = 1. For n >= 3 this root is exactly the fixed point of the
    Jullien & Salanie iteration z = (sum_i sqrt(z^2 + 4 (1 - z) pi_i^2 / S) - 2) / (n - 2); it is found
    here by bisection on the strictly decreasing sum_i p_i(z) instead, which needs no starting value and
    also covers two-outcome markets, where that iteration is undefined. z > 0 for an overround book, 0
    for a fair one; an underround book gives z < 0, which has no insider interpretation (the row is
    still returned, flagged by its sign). A row whose root lies outside (-0.99, 1) yields NaN.

    References: Shin (1993), "Measuring the incidence of insider trading in a market for state-contingent
    claims", Economic Journal 103(420); Jullien & Salanie (1994), "Measuring the incidence of insider
    trading: a comment on Shin", Economic Journal 104(427); Strumbelj (2014), "On determining probability
    forecasts from betting odds", International Journal of Forecasting 30(4).
    """
    pi, ok, single = _implied(odds)
    pi = np.where(ok[:, None], pi, 0.5)
    total = pi.sum(axis=1, keepdims=True)
    lo = np.full(len(pi), -0.99)
    hi = np.full(len(pi), 1.0 - 1e-12)
    bracketed = (_shin_probs(lo, pi, total).sum(axis=1) >= 1.0) & (_shin_probs(hi, pi, total).sum(axis=1) <= 1.0)
    for _ in range(_MAX_ITER):
        mid = (lo + hi) / 2
        above = _shin_probs(mid, pi, total).sum(axis=1) > 1.0  # sum decreases in z: root is to the right
        lo = np.where(above, mid, lo)
        hi = np.where(above, hi, mid)
        if np.all(hi - lo < 1e-15):
            break
    z = (lo + hi) / 2
    probs = _shin_probs(z, pi, total)
    probs = probs / probs.sum(axis=1, keepdims=True)
    bad = ~ok | ~bracketed
    probs[bad] = np.nan
    z[bad] = np.nan
    return _pack(probs, z, single)


METHODS: tuple[str, ...] = ("multiplicative", "additive", "power", "shin")
_FUNCS: dict[str, Callable[..., DevigResult]] = {
    "multiplicative": multiplicative,
    "additive": additive,
    "power": power,
    "shin": shin,
}
PARAM_NAMES: dict[str, str] = {"multiplicative": "overround", "additive": "overround", "power": "k", "shin": "z"}


def devig(odds: Sequence[float] | Sequence[Sequence[float]] | np.ndarray, method: str = "power") -> DevigResult:
    """Dispatch to one of `METHODS` by name. Raises ValueError for an unknown method."""
    if method not in _FUNCS:
        raise ValueError(f"unknown de-vig method {method!r}; choose from {METHODS}")
    return _FUNCS[method](odds)


def devig_frame(df: pd.DataFrame, columns: Sequence[str], method: str = "power", prefix: str = "p_") -> pd.DataFrame:
    """De-vig the decimal-odds `columns` of `df` (one market per row, one column per outcome).

    Returns a frame on `df`'s index with one `<prefix><column>` probability column per outcome followed by
    the method's parameter column `<method>_<param>` (e.g. `power_k`, `shin_z`, `multiplicative_overround`).
    """
    res = devig(df[list(columns)].to_numpy(dtype=float), method=method)
    out = pd.DataFrame(res.probs, index=df.index, columns=[f"{prefix}{c}" for c in columns])
    out[f"{method}_{PARAM_NAMES[method]}"] = res.param
    return out
