"""Statistics for grading betting-market hypotheses on the `research/STANDARDS.md` evidence ladder.

Every function here is pure (no I/O, no global state; randomness only through an explicit `seed`) and
states its formula and reference in its docstring, so an auditor can re-derive any number by hand.

Contents:

- `block_bootstrap` / `paired_block_bootstrap_diff` — percentile CIs and one-sided p-values that
  respect within-block dependence (e.g. matches on the same match-day share news, weather and market
  moves), by resampling whole blocks.
- `probabilistic_sharpe_ratio`, `expected_max_sharpe`, `deflated_sharpe_ratio` — Sharpe-ratio
  significance with non-normal returns, and deflated for the number of trials searched.
- `pbo_cscv` — probability of backtest overfitting via combinatorially symmetric cross-validation.
- `benjamini_hochberg` — false-discovery-rate control across a family of hypotheses.
- `min_detectable_effect` — the smallest true mean effect a test of a given size can reliably detect.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from itertools import combinations
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from scipy.stats import norm

EULER_MASCHERONI = 0.5772156649015329

Stat = Callable[[np.ndarray], Any]


class BootstrapResult(NamedTuple):
    """Outcome of a block bootstrap.

    `estimate` is the statistic on the original sample; `lo`/`hi` the percentile CI at level
    `1 - alpha`; `p_gt0` the one-sided p-value for H1: statistic > 0 and `p_lt0` for H1: statistic < 0;
    `n` the number of observations, `n_blocks` the number of distinct blocks, `n_boot` the draws.
    """

    estimate: float
    lo: float
    hi: float
    p_gt0: float
    p_lt0: float
    n: int
    n_blocks: int
    n_boot: int


class PBOResult(NamedTuple):
    """Outcome of CSCV: `pbo` = share of splits whose in-sample winner ranks at or below the
    out-of-sample median (logit <= 0); `logits` = one logit per split; `n_combinations` = C(S, S/2)."""

    pbo: float
    logits: np.ndarray
    n_combinations: int


# --------------------------------------------------------------------------- block bootstrap


def _as_values(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return arr


def _block_codes(blocks: Sequence[Any] | np.ndarray | pd.Series, n: int) -> tuple[np.ndarray, int]:
    codes, uniques = pd.factorize(pd.Series(np.asarray(blocks)), sort=True)
    if len(codes) != n:
        raise ValueError(f"blocks has {len(codes)} entries but there are {n} values")
    if (codes < 0).any():
        raise ValueError("blocks contains missing labels")
    if len(uniques) < 2:
        raise ValueError(f"a block bootstrap needs at least 2 blocks, got {len(uniques)}")
    return codes.astype(np.int64), len(uniques)


def _summarise(estimate: float, boot: np.ndarray, alpha: float, n: int, n_blocks: int) -> BootstrapResult:
    n_boot = len(boot)
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    p_gt0 = (1 + int(np.sum(boot <= 0))) / (n_boot + 1)
    p_lt0 = (1 + int(np.sum(boot >= 0))) / (n_boot + 1)
    return BootstrapResult(float(estimate), float(lo), float(hi), p_gt0, p_lt0, n, n_blocks, n_boot)


def _check_boot_args(n_boot: int, alpha: float) -> None:
    if n_boot < 1:
        raise ValueError(f"n_boot must be >= 1, got {n_boot}")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")


def _bootstrap(
    columns: list[np.ndarray],
    codes: np.ndarray,
    n_blocks: int,
    stat: Stat,
    combine: Callable[[list[float]], float],
    n_boot: int,
    seed: int,
    alpha: float,
) -> BootstrapResult:
    """Shared engine: draw `n_blocks` block labels with replacement per replicate, evaluate `stat` on
    every column restricted to the drawn blocks, and combine the per-column results into one number."""
    n = len(codes)
    estimate = combine([float(stat(col)) for col in columns])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_blocks, size=(n_boot, n_blocks))
    block_n = np.bincount(codes, minlength=n_blocks).astype(float)

    if stat is np.mean:
        # Mean of a union of whole blocks = (sum of the drawn blocks' sums) / (sum of their sizes).
        sizes = block_n[draws].sum(axis=1)
        per_col = [np.bincount(codes, weights=col, minlength=n_blocks)[draws].sum(axis=1) / sizes for col in columns]
        boot = np.array([combine([float(pc[i]) for pc in per_col]) for i in range(n_boot)])
    else:
        order = np.argsort(codes, kind="stable")
        bounds = np.concatenate([[0], np.cumsum(block_n.astype(np.int64))])
        members = [order[bounds[b] : bounds[b + 1]] for b in range(n_blocks)]
        boot = np.empty(n_boot)
        for i, row in enumerate(draws):
            idx = np.concatenate([members[b] for b in row])
            boot[i] = combine([float(stat(col[idx])) for col in columns])
    return _summarise(estimate, boot, alpha, n, n_blocks)


def block_bootstrap(
    values: Sequence[float] | np.ndarray,
    blocks: Sequence[Any] | np.ndarray | pd.Series,
    stat: Stat = np.mean,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> BootstrapResult:
    """Block bootstrap of `stat(values)`, resampling whole blocks with replacement.

    Takes per-decision `values`, a same-length `blocks` label per value (e.g. the match-day), a
    statistic (default the mean), the number of draws, a seed and the CI level `1 - alpha`.

    Method: with B distinct blocks, each replicate draws B block labels uniformly with replacement and
    evaluates `stat` on the concatenation of every value in the drawn blocks (a block drawn twice
    contributes all its values twice). This is the cluster bootstrap of Davison & Hinkley (1997,
    *Bootstrap Methods and their Application*, sec. 3.8); it keeps the within-block dependence that an
    iid bootstrap would destroy, so the CI widens accordingly.

    CI: percentile interval [q(alpha/2), q(1 - alpha/2)] of the replicates (Efron & Tibshirani 1993,
    ch. 13). One-sided p-values invert the same percentile interval:
    p_gt0 = (1 + #{theta* <= 0}) / (n_boot + 1), p_lt0 = (1 + #{theta* >= 0}) / (n_boot + 1)
    (the +1 keeps a finite number of draws from ever reporting p = 0; Davison & Hinkley eq. 4.11). So
    the two-sided (1 - alpha) CI excludes zero on the positive side essentially when p_gt0 < alpha/2.

    Returns a `BootstrapResult`. Raises ValueError on empty input, NaN/inf values, or a length mismatch.
    """
    _check_boot_args(n_boot, alpha)
    arr = _as_values(values, "values")
    codes, n_blocks = _block_codes(blocks, len(arr))
    return _bootstrap([arr], codes, n_blocks, stat, lambda r: r[0], n_boot, seed, alpha)


def paired_block_bootstrap_diff(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
    blocks: Sequence[Any] | np.ndarray | pd.Series,
    stat: Stat = np.mean,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> BootstrapResult:
    """Block bootstrap of `stat(a) - stat(b)` where `a[i]` and `b[i]` are two strategies' metric on the
    same decision `i` (e.g. challenger vs champion CLV per bet).

    Both series are resampled with the same block draws, so noise shared by a block (the match-day)
    cancels in the difference — the paired comparison `research/STANDARDS.md` requires for the champion
    ratchet. Formula and CI/p-value conventions as in `block_bootstrap`; p_gt0 is the one-sided p-value
    for "a beats b". Raises ValueError if `a` and `b` differ in length.
    """
    _check_boot_args(n_boot, alpha)
    arr_a = _as_values(a, "a")
    arr_b = _as_values(b, "b")
    if len(arr_a) != len(arr_b):
        raise ValueError(f"a and b must be paired (same length), got {len(arr_a)} and {len(arr_b)}")
    codes, n_blocks = _block_codes(blocks, len(arr_a))
    return _bootstrap([arr_a, arr_b], codes, n_blocks, stat, lambda r: r[0] - r[1], n_boot, seed, alpha)


# --------------------------------------------------------------------------- Sharpe-ratio statistics


def probabilistic_sharpe_ratio(sr: float, n: int, skew: float, kurt: float, sr_benchmark: float = 0.0) -> float:
    """Probabilistic Sharpe ratio: the probability that the true Sharpe ratio exceeds `sr_benchmark`.

    Takes the observed (non-annualised, per-observation) Sharpe ratio `sr`, the number of observations
    `n`, their sample skewness, and their *non-excess* (Pearson) kurtosis `kurt` (3 for a normal).
    Pearson's inequality kurt >= 1 + skew^2 holds for every distribution; a smaller value (typically
    excess kurtosis passed by mistake) raises ValueError.

        PSR(SR*) = Phi( (SR - SR*) * sqrt(n - 1) / sqrt(1 - skew * SR + (kurt - 1) / 4 * SR^2) )

    Reference: Bailey & Lopez de Prado (2012), "The Sharpe Ratio Efficient Frontier", Journal of Risk
    15(2), eq. 2 / sec. 3. Raises ValueError when n < 2 or the variance term is not positive.
    """
    if n < 2:
        raise ValueError(f"PSR needs n >= 2 observations, got {n}")
    if kurt < 1.0 + skew**2 - 1e-12:
        raise ValueError(
            f"kurt={kurt:.4g} < 1 + skew^2 = {1 + skew**2:.4g} is impossible for non-excess kurtosis "
            "(was excess kurtosis passed?)"
        )
    var_term = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2
    if var_term <= 0:
        raise ValueError(f"non-positive PSR variance term {var_term:.4g} (check skew/kurt; kurt is non-excess)")
    return float(norm.cdf((sr - sr_benchmark) * math.sqrt(n - 1) / math.sqrt(var_term)))


def expected_max_sharpe(n_trials: int, var_trial_sr: float) -> float:
    """Expected maximum Sharpe ratio among `n_trials` independent trials whose true Sharpe is zero and
    whose estimated Sharpe ratios have cross-trial variance `var_trial_sr`.

        E[max SR] ~= sqrt(V) * ((1 - gamma) * Phi^-1(1 - 1/N) + gamma * Phi^-1(1 - 1/(N e)))

    with gamma the Euler-Mascheroni constant — the extreme-value approximation to the expected maximum
    of N iid standard normals. Reference: Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio",
    Journal of Portfolio Management 40(5), eq. 5 (proof in Bailey, Borwein, Lopez de Prado & Zhu 2014,
    "Pseudo-Mathematics and Financial Charlatanism", appendix). Returns 0 for a single trial (nothing was
    searched over). Raises ValueError for n_trials < 1 or negative variance.
    """
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")
    if var_trial_sr < 0:
        raise ValueError(f"var_trial_sr must be >= 0, got {var_trial_sr}")
    if n_trials == 1:
        return 0.0
    g = EULER_MASCHERONI
    z = (1 - g) * norm.ppf(1 - 1 / n_trials) + g * norm.ppf(1 - 1 / (n_trials * math.e))
    return float(math.sqrt(var_trial_sr) * z)


def deflated_sharpe_ratio(sr: float, n: int, skew: float, kurt: float, n_trials: int, var_trial_sr: float) -> float:
    """Deflated Sharpe ratio: PSR against the Sharpe ratio expected from the best of `n_trials` skill-less
    trials, i.e. DSR = PSR(SR0) with SR0 = `expected_max_sharpe(n_trials, var_trial_sr)`.

    Arguments as in `probabilistic_sharpe_ratio` (kurt non-excess) plus the number of trials searched
    and the variance of the Sharpe ratios across them. DSR >= 0.95 is the L3 bar in
    `research/STANDARDS.md`. Reference: Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio",
    Journal of Portfolio Management 40(5), eq. 4-5.
    """
    return probabilistic_sharpe_ratio(sr, n, skew, kurt, sr_benchmark=expected_max_sharpe(n_trials, var_trial_sr))


# --------------------------------------------------------------------------- PBO / CSCV


def pbo_cscv(
    perf_matrix: np.ndarray | pd.DataFrame,
    n_splits: int = 16,
    metric: Callable[[np.ndarray], np.ndarray] | None = None,
    weights: np.ndarray | pd.DataFrame | None = None,
) -> PBOResult:
    """Probability of backtest overfitting via combinatorially symmetric cross-validation (CSCV).

    Takes a T x N matrix: T chronological performance observations (e.g. per-match-day metric) for each
    of N strategy configurations (the registered grid), an even number of splits S, and a `metric`
    mapping a (rows x N) sub-matrix to N scores (default: the column mean; higher is better).

    Method (Bailey, Borwein, Lopez de Prado & Zhu 2017, "The Probability of Backtest Overfitting",
    Journal of Computational Finance 20(4), sec. 2):
    1. Split the rows into S contiguous groups (`np.array_split`, so sizes differ by at most one row).
    2. For each of the C(S, S/2) ways to pick S/2 groups as in-sample (IS), the rest out-of-sample (OOS):
       select n* = argmax of the IS metric; compute its OOS rank r among the N configurations
       (ascending, ties averaged) and its relative rank w = r / (N + 1); logit lambda = ln(w / (1 - w)).
    3. PBO = share of splits with lambda <= 0, i.e. the IS winner is at or below the OOS median.

    With `weights` (same shape as `perf_matrix`, e.g. decisions per match-day while `perf_matrix` holds
    each match-day's metric sum) the score is the weighted mean sum(perf) / sum(weights) over the rows of a
    half — the decision-weighted mean — and a configuration with zero weight in a half scores -inf there
    (it made no decisions, so it cannot be the in-sample winner and ranks last out of sample).

    Returns a `PBOResult`. Pure noise gives PBO ~ 0.5; a genuinely dominant configuration gives ~ 0.
    Raises ValueError for N < 2, odd or < 2 splits, fewer rows than splits, or weights that do not match.
    """
    perf = np.asarray(perf_matrix, dtype=float)
    if perf.ndim != 2:
        raise ValueError(f"perf_matrix must be 2-D (T x N), got shape {perf.shape}")
    t_rows, n_cfg = perf.shape
    if n_cfg < 2:
        raise ValueError("PBO needs at least two configurations to select among")
    if n_splits < 2 or n_splits % 2:
        raise ValueError(f"n_splits must be an even number >= 2, got {n_splits}")
    if t_rows < n_splits:
        raise ValueError(f"need at least n_splits={n_splits} rows, got {t_rows}")
    if not np.all(np.isfinite(perf)):
        raise ValueError("perf_matrix contains NaN or infinite values")

    w = None
    if weights is not None:
        if metric is not None:
            raise ValueError("pass either metric or weights, not both")
        w = np.asarray(weights, dtype=float)
        if w.shape != perf.shape or not np.all(np.isfinite(w)) or np.any(w < 0):
            raise ValueError("weights must be finite, non-negative and shaped like perf_matrix")

    parts = np.array_split(np.arange(t_rows), n_splits)
    combos = list(combinations(range(n_splits), n_splits // 2))
    if w is not None:
        part_sums = np.vstack([perf[p].sum(axis=0) for p in parts])
        part_w = np.vstack([w[p].sum(axis=0) for p in parts])
        mask = np.zeros((len(combos), n_splits))
        for i, c in enumerate(combos):
            mask[i, list(c)] = 1.0
        with np.errstate(divide="ignore", invalid="ignore"):
            is_w, oos_w = mask @ part_w, (1 - mask) @ part_w
            is_perf = np.where(is_w > 0, (mask @ part_sums) / is_w, -np.inf)
            oos_perf = np.where(oos_w > 0, ((1 - mask) @ part_sums) / oos_w, -np.inf)
    elif metric is None:
        part_sums = np.vstack([perf[p].sum(axis=0) for p in parts])  # S x N
        part_n = np.array([len(p) for p in parts], dtype=float)
        mask = np.zeros((len(combos), n_splits))
        for i, c in enumerate(combos):
            mask[i, list(c)] = 1.0
        is_perf = (mask @ part_sums) / (mask @ part_n)[:, None]
        oos_perf = ((1 - mask) @ part_sums) / ((1 - mask) @ part_n)[:, None]
    else:
        is_perf = np.empty((len(combos), n_cfg))
        oos_perf = np.empty((len(combos), n_cfg))
        for i, c in enumerate(combos):
            chosen = set(c)
            is_rows = np.concatenate([parts[j] for j in range(n_splits) if j in chosen])
            oos_rows = np.concatenate([parts[j] for j in range(n_splits) if j not in chosen])
            is_perf[i] = np.asarray(metric(perf[is_rows]), dtype=float)
            oos_perf[i] = np.asarray(metric(perf[oos_rows]), dtype=float)

    best = np.argmax(is_perf, axis=1)
    chosen_oos = oos_perf[np.arange(len(combos)), best][:, None]
    below = (oos_perf < chosen_oos).sum(axis=1)
    ties = (oos_perf == chosen_oos).sum(axis=1)  # includes the chosen configuration itself
    rank = below + (ties + 1) / 2.0  # average rank, 1-based, ascending
    w = rank / (n_cfg + 1)
    logits = np.log(w / (1 - w))
    return PBOResult(float(np.mean(logits <= 0)), logits, len(combos))


# --------------------------------------------------------------------------- multiple testing


def benjamini_hochberg(pvals: Sequence[float] | np.ndarray, q: float = 0.10) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg step-up procedure controlling the false discovery rate at level `q`.

    Takes m p-values (NaN entries are ignored: never rejected, adjusted value NaN, not counted in m).
    With sorted p-values p_(1) <= ... <= p_(m), reject H_(1..k) for the largest k with p_(k) <= k q / m.
    Adjusted p-values (q-values): q_(i) = min_{j >= i} min(1, m p_(j) / j); H_i is rejected iff
    q_i <= q.

    Returns `(rejected, qvalues)` as arrays in the input order. Reference: Benjamini & Hochberg (1995),
    "Controlling the False Discovery Rate", JRSS B 57(1), 289-300.
    """
    if not 0 < q < 1:
        raise ValueError(f"q must be in (0, 1), got {q}")
    p = np.asarray(pvals, dtype=float).reshape(-1)
    qvals = np.full(p.shape, np.nan)
    ok = ~np.isnan(p)
    if np.any((p[ok] < 0) | (p[ok] > 1)):
        raise ValueError("p-values must lie in [0, 1]")
    m = int(ok.sum())
    if m:
        idx = np.flatnonzero(ok)
        order = idx[np.argsort(p[idx], kind="stable")]
        raw = p[order] * m / np.arange(1, m + 1)
        qvals[order] = np.clip(np.minimum.accumulate(raw[::-1])[::-1], 0, 1)
    rejected = np.zeros(p.shape, dtype=bool)
    rejected[ok] = qvals[ok] <= q
    return rejected, qvals


# --------------------------------------------------------------------------- power


def min_detectable_effect(sd: float, n: int, alpha: float = 0.05, power: float = 0.8, one_sided: bool = True) -> float:
    """Minimum detectable effect of a test on a mean with `n` independent observations of standard
    deviation `sd`, at significance `alpha` and the given `power`:

        MDE = (z_{1 - alpha'} + z_{power}) * sd / sqrt(n),  alpha' = alpha (one-sided) or alpha/2

    Normal-approximation power formula (Cohen 1988, *Statistical Power Analysis for the Behavioral
    Sciences*, ch. 2; Bloom 1995, "Minimum Detectable Effects", Evaluation Review 19(5)). With clustered
    observations (several decisions per match-day) pass the effective sample size n / design-effect, or
    an sd inflated by sqrt(design-effect), or the MDE is optimistic.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if sd < 0:
        raise ValueError(f"sd must be >= 0, got {sd}")
    if not 0 < alpha < 1 or not 0 < power < 1:
        raise ValueError("alpha and power must lie in (0, 1)")
    z_alpha = norm.ppf(1 - alpha) if one_sided else norm.ppf(1 - alpha / 2)
    return float((z_alpha + norm.ppf(power)) * sd / math.sqrt(n))


def directional_p_value(p_gt0: float, p_lt0: float, direction: str) -> float:
    """The p-value for a registered direction from the two one-sided p-values: p_gt0 for `positive`,
    p_lt0 for `negative`, and min(1, 2 min(p_gt0, p_lt0)) for `two-sided` (the doubled smaller tail)."""
    if direction == "positive":
        return p_gt0
    if direction == "negative":
        return p_lt0
    if direction == "two-sided":
        return min(1.0, 2.0 * min(p_gt0, p_lt0))
    raise ValueError(f"unknown direction {direction!r}")
