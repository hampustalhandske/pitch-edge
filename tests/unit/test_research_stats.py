"""Known-value and property tests for `pitch_edge.research.stats`.

Expected values for the closed-form statistics are recomputed here from the published formulas with
`scipy.stats.norm`, independently of the implementation under test.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import norm

from pitch_edge.research.stats import (
    benjamini_hochberg,
    block_bootstrap,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    min_detectable_effect,
    paired_block_bootstrap_diff,
    pbo_cscv,
    probabilistic_sharpe_ratio,
)

pytestmark = pytest.mark.unit

EULER_GAMMA = 0.5772156649015329


# --------------------------------------------------------------------------- PSR / DSR


def _psr_reference(sr: float, n: int, skew: float, kurt: float, sr_star: float) -> float:
    """Bailey & Lopez de Prado (2012), eq. for PSR with non-excess kurtosis."""
    return float(norm.cdf((sr - sr_star) * math.sqrt(n - 1) / math.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr**2)))


def test_psr_matches_closed_form_hand_value():
    # z = 0.1 * sqrt(499) / sqrt(1 + 0.05 + 0.0075) = 2.17221...
    z = 0.1 * math.sqrt(499) / math.sqrt(1.0575)
    assert z == pytest.approx(2.17221, abs=1e-4)
    got = probabilistic_sharpe_ratio(sr=0.1, n=500, skew=-0.5, kurt=4.0)
    assert got == pytest.approx(norm.cdf(z), abs=1e-12)
    assert got == pytest.approx(0.98508, abs=1e-4)


@pytest.mark.parametrize(
    ("sr", "n", "skew", "kurt", "bench"),
    [(0.05, 1000, 0.0, 3.0, 0.0), (0.2, 60, -1.2, 9.0, 0.05), (-0.03, 250, 0.4, 3.5, 0.0), (0.1, 2000, 0.0, 3.0, 0.1)],
)
def test_psr_matches_reference(sr, n, skew, kurt, bench):
    assert probabilistic_sharpe_ratio(sr, n, skew, kurt, sr_benchmark=bench) == pytest.approx(
        _psr_reference(sr, n, skew, kurt, bench), abs=1e-12
    )


def test_psr_is_half_at_the_benchmark_and_monotone_in_n():
    assert probabilistic_sharpe_ratio(0.1, 300, 0.0, 3.0, sr_benchmark=0.1) == pytest.approx(0.5)
    assert probabilistic_sharpe_ratio(0.05, 2000, 0.0, 3.0) > probabilistic_sharpe_ratio(0.05, 200, 0.0, 3.0)


def test_psr_rejects_too_few_observations():
    with pytest.raises(ValueError):
        probabilistic_sharpe_ratio(0.1, 1, 0.0, 3.0)


def test_psr_rejects_impossible_non_excess_kurtosis():
    # Pearson's inequality: kurt >= 1 + skew^2 for any distribution. Passing *excess* kurtosis (0 for a
    # normal) where non-excess is expected usually violates it, so it is refused rather than misread.
    with pytest.raises(ValueError, match="kurt"):
        probabilistic_sharpe_ratio(0.08, 400, -0.5, 1.2)  # 1.2 < 1 + 0.25
    with pytest.raises(ValueError, match="kurt"):
        probabilistic_sharpe_ratio(0.08, 400, 0.0, 0.0)
    assert probabilistic_sharpe_ratio(0.08, 400, -0.5, 1.25) > 0  # boundary (two-point distribution) is allowed


def test_expected_max_sharpe_matches_formula_and_known_order_statistic():
    n, var = 100, 0.0025
    expected = math.sqrt(var) * ((1 - EULER_GAMMA) * norm.ppf(1 - 1 / n) + EULER_GAMMA * norm.ppf(1 - 1 / (n * math.e)))
    assert expected_max_sharpe(n, var) == pytest.approx(expected, abs=1e-12)
    # E[max of 1000 iid N(0,1)] = 3.2414 exactly; the approximation is within ~0.02 of it.
    assert expected_max_sharpe(1000, 1.0) == pytest.approx(3.2414, abs=0.05)
    assert expected_max_sharpe(1, 1.0) == 0.0


def test_dsr_matches_psr_at_expected_max_benchmark():
    sr, n, skew, kurt, n_trials, var = 0.12, 800, -0.3, 5.0, 50, 0.002
    sr0 = math.sqrt(var) * (
        (1 - EULER_GAMMA) * norm.ppf(1 - 1 / n_trials) + EULER_GAMMA * norm.ppf(1 - 1 / (n_trials * math.e))
    )
    expected = _psr_reference(sr, n, skew, kurt, sr0)
    assert deflated_sharpe_ratio(sr, n, skew, kurt, n_trials, var) == pytest.approx(expected, abs=1e-12)


def test_dsr_shrinks_with_more_trials_and_equals_psr_for_one_trial():
    kw = {"sr": 0.1, "n": 500, "skew": 0.0, "kurt": 3.0, "var_trial_sr": 0.003}
    one = deflated_sharpe_ratio(n_trials=1, **kw)
    assert one == pytest.approx(probabilistic_sharpe_ratio(0.1, 500, 0.0, 3.0))
    assert deflated_sharpe_ratio(n_trials=10, **kw) < one
    assert deflated_sharpe_ratio(n_trials=1000, **kw) < deflated_sharpe_ratio(n_trials=10, **kw)


# --------------------------------------------------------------------------- PBO


def test_pbo_exact_two_split_examples():
    # Column 0 wins the first half and loses the second; column 1 the reverse: always overfit.
    flip = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    res = pbo_cscv(flip, n_splits=2)
    assert res.n_combinations == 2
    assert res.pbo == 1.0
    np.testing.assert_allclose(res.logits, [math.log(0.5), math.log(0.5)])

    # Column 0 wins both halves: the IS winner is the OOS winner, rank 2 of 2 -> omega = 2/3.
    stable = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    res = pbo_cscv(stable, n_splits=2)
    assert res.pbo == 0.0
    np.testing.assert_allclose(res.logits, [math.log(2.0), math.log(2.0)])


def test_pbo_near_half_for_pure_noise():
    pbos = []
    for seed in range(6):
        rng = np.random.default_rng(seed)
        pbos.append(pbo_cscv(rng.normal(size=(640, 10)), n_splits=16).pbo)
    assert 0.35 < float(np.mean(pbos)) < 0.65


def test_pbo_near_zero_when_one_column_dominates():
    rng = np.random.default_rng(7)
    m = rng.normal(size=(640, 10))
    m[:, 3] += 1.0
    res = pbo_cscv(m, n_splits=16)
    assert res.n_combinations == math.comb(16, 8)
    assert len(res.logits) == res.n_combinations
    assert res.pbo < 0.02


def test_pbo_custom_metric_is_used():
    rng = np.random.default_rng(1)
    m = rng.normal(size=(320, 6))
    m[:, 0] += 1.0  # best mean ...
    m[:, 0] *= -1.0  # ... turned into the worst mean; with -mean as the metric it is the best again
    res = pbo_cscv(m, n_splits=8, metric=lambda x: -x.mean(axis=0))
    assert res.pbo < 0.05


def test_pbo_weighted_mean_metric():
    # perf holds per-row sums, weights per-row counts: the score is the decision-weighted mean.
    sums = np.array([[2.0, 0.0], [0.0, 3.0], [0.0, 3.0], [2.0, 0.0]])
    counts = np.array([[1.0, 0.0], [0.0, 10.0], [0.0, 10.0], [1.0, 0.0]])
    # Halves: rows {0,1} and {2,3}. Column 0 mean 2.0 in both halves, column 1 mean 0.3: always the winner.
    res = pbo_cscv(sums, n_splits=2, weights=counts)
    assert res.pbo == 0.0
    # Unweighted, column 1 has the larger per-row mean (1.5 vs 1.0), so weighting changes the winner.
    assert pbo_cscv(sums, n_splits=2).pbo == 0.0
    with pytest.raises(ValueError):  # weights must match perf
        pbo_cscv(sums, n_splits=2, weights=counts[:, :1])


def test_pbo_rejects_bad_shapes():
    with pytest.raises(ValueError):
        pbo_cscv(np.zeros((10, 1)), n_splits=2)  # one configuration: nothing to select among
    with pytest.raises(ValueError):
        pbo_cscv(np.zeros((10, 3)), n_splits=3)  # odd number of splits
    with pytest.raises(ValueError):
        pbo_cscv(np.zeros((8, 3)), n_splits=16)  # fewer rows than splits


# --------------------------------------------------------------------------- BH


def test_bh_all_rejected_hand_example():
    rej, q = benjamini_hochberg([0.01, 0.04, 0.03, 0.005], q=0.05)
    assert rej.tolist() == [True, True, True, True]
    np.testing.assert_allclose(q, [0.02, 0.04, 0.04, 0.02])


def test_bh_none_rejected_hand_example():
    rej, q = benjamini_hochberg([0.01, 0.02, 0.03, 0.04, 0.05, 0.5], q=0.05)
    assert not rej.any()
    np.testing.assert_allclose(q, [0.06, 0.06, 0.06, 0.06, 0.06, 0.5])


def test_bh_partial_rejection_hand_example():
    p = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205]
    rej, q = benjamini_hochberg(p, q=0.05)
    assert rej.tolist() == [True, True, False, False, False, False, False, False]
    np.testing.assert_allclose(q, [0.008, 0.032, 0.0672, 0.0672, 0.0672, 0.08, 0.0845714286, 0.205], rtol=1e-6)


def test_bh_step_up_rejects_p_values_above_their_own_threshold():
    # p_(1)=0.04 > 1/2 * 0.05 but p_(2)=0.045 <= 0.05, so the step-up rejects both.
    rej, _ = benjamini_hochberg([0.045, 0.04], q=0.05)
    assert rej.tolist() == [True, True]


def test_bh_nan_is_never_rejected_and_not_counted():
    rej, q = benjamini_hochberg([0.01, np.nan, 0.04], q=0.05)
    assert rej.tolist() == [True, False, True]
    assert np.isnan(q[1])
    np.testing.assert_allclose(q[[0, 2]], [0.02, 0.04])


def test_bh_empty():
    rej, q = benjamini_hochberg([], q=0.1)
    assert rej.shape == (0,) and q.shape == (0,)


def test_bh_matches_statsmodels_when_available():
    multitest = pytest.importorskip("statsmodels.stats.multitest")
    p = np.random.default_rng(3).uniform(size=40) ** 3
    rej, q = benjamini_hochberg(p, q=0.1)
    sm_rej, sm_q, _, _ = multitest.multipletests(p, alpha=0.1, method="fdr_bh")
    np.testing.assert_array_equal(rej, sm_rej)
    np.testing.assert_allclose(q, sm_q)


# --------------------------------------------------------------------------- block bootstrap


def _clustered(rng: np.random.Generator, n_blocks: int, per_block: int, mu: float, block_sd: float):
    blocks = np.repeat(np.arange(n_blocks), per_block)
    effects = rng.normal(0, block_sd, size=n_blocks)
    values = mu + effects[blocks] + rng.normal(0, 1.0, size=n_blocks * per_block)
    return values, blocks


def test_block_bootstrap_estimate_and_reproducibility():
    rng = np.random.default_rng(0)
    values, blocks = _clustered(rng, 50, 4, 0.3, 0.5)
    a = block_bootstrap(values, blocks, n_boot=500, seed=11)
    b = block_bootstrap(values, blocks, n_boot=500, seed=11)
    assert a == b
    assert a.estimate == pytest.approx(values.mean())
    assert a.lo < a.estimate < a.hi
    assert a.n == 200 and a.n_blocks == 50 and a.n_boot == 500


def test_block_bootstrap_coverage_is_close_to_nominal():
    rng = np.random.default_rng(2024)
    hits = 0
    sims = 400
    for i in range(sims):
        values, blocks = _clustered(rng, 60, 5, 0.0, 0.7)
        res = block_bootstrap(values, blocks, n_boot=1000, seed=i)
        hits += res.lo <= 0.0 <= res.hi
    # Nominal 0.95; the binomial SE over 400 simulations is ~0.011.
    assert 0.92 <= hits / sims <= 0.98


def test_block_bootstrap_is_wider_than_iid_under_clustering():
    rng = np.random.default_rng(5)
    values, blocks = _clustered(rng, 40, 10, 0.0, 1.0)
    clustered = block_bootstrap(values, blocks, n_boot=1000, seed=1)
    iid = block_bootstrap(values, np.arange(len(values)), n_boot=1000, seed=1)
    assert (clustered.hi - clustered.lo) > 1.8 * (iid.hi - iid.lo)


def test_block_bootstrap_resamples_whole_blocks():
    # Block b holds b+1 copies of the value b, so any resample made of whole blocks contains each
    # value a multiple of (b+1) times.
    blocks = np.concatenate([np.full(b + 1, b) for b in range(8)])
    values = blocks.astype(float)
    seen = []

    def stat(x: np.ndarray) -> float:
        vals, counts = np.unique(x, return_counts=True)
        for v, c in zip(vals, counts, strict=True):
            assert c % (int(v) + 1) == 0
        seen.append(len(x))
        return float(np.mean(x))

    block_bootstrap(values, blocks, stat=stat, n_boot=200, seed=3)
    assert len(seen) == 201  # the point estimate plus every draw
    assert len(set(seen)) > 1  # resample sizes vary because blocks have different sizes


def test_block_bootstrap_mean_fast_path_equals_generic_path():
    rng = np.random.default_rng(9)
    values, blocks = _clustered(rng, 30, 3, 0.1, 0.3)
    fast = block_bootstrap(values, blocks, stat=np.mean, n_boot=300, seed=4)
    slow = block_bootstrap(values, blocks, stat=lambda x: float(np.mean(x)), n_boot=300, seed=4)
    assert fast.lo == pytest.approx(slow.lo)
    assert fast.hi == pytest.approx(slow.hi)
    assert fast.p_gt0 == pytest.approx(slow.p_gt0)


def test_block_bootstrap_p_values():
    rng = np.random.default_rng(1)
    values, blocks = _clustered(rng, 100, 4, 0.5, 0.2)
    res = block_bootstrap(values, blocks, n_boot=2000, seed=0)
    assert res.p_gt0 < 0.01
    assert res.p_lt0 > 0.99
    assert res.p_gt0 == pytest.approx(1 / 2001)  # +1-corrected: never exactly zero
    neg = block_bootstrap(-values, blocks, n_boot=2000, seed=0)
    assert neg.p_lt0 < 0.01 and neg.p_gt0 > 0.99


def test_block_bootstrap_ci_and_p_value_agree():
    rng = np.random.default_rng(12)
    for _ in range(20):
        values, blocks = _clustered(rng, 40, 3, 0.15, 0.3)
        res = block_bootstrap(values, blocks, n_boot=1000, seed=0, alpha=0.05)
        # Percentile-inversion p-value: the CI excludes 0 on the positive side exactly when fewer than
        # alpha/2 of the draws are <= 0 (up to quantile interpolation and the +1 correction).
        if res.lo > 0:
            assert res.p_gt0 <= 0.027
        else:
            assert res.p_gt0 >= 0.024


def test_block_bootstrap_refuses_a_single_block():
    # One block: every replicate is the original sample, so the "CI" collapses to a point and p_lt0 or
    # p_gt0 reaches its floor no matter how weak the effect.
    with pytest.raises(ValueError, match="at least 2 blocks"):
        block_bootstrap(np.random.default_rng(0).normal(size=50), np.zeros(50), n_boot=100)
    with pytest.raises(ValueError, match="at least 2 blocks"):
        paired_block_bootstrap_diff(np.ones(5), np.zeros(5), np.zeros(5), n_boot=100)


def test_block_bootstrap_validates_inputs():
    with pytest.raises(ValueError):
        block_bootstrap([1.0, 2.0], [0], n_boot=10)
    with pytest.raises(ValueError):
        block_bootstrap([], [], n_boot=10)
    with pytest.raises(ValueError):
        block_bootstrap([1.0, np.nan], [0, 1], n_boot=10)


def test_paired_diff_cancels_shared_block_noise():
    rng = np.random.default_rng(4)
    n_blocks, per = 80, 5
    blocks = np.repeat(np.arange(n_blocks), per)
    shared = rng.normal(0, 3.0, size=n_blocks)[blocks]
    a = shared + 0.1 + rng.normal(0, 0.1, size=len(blocks))
    b = shared + rng.normal(0, 0.1, size=len(blocks))
    paired = paired_block_bootstrap_diff(a, b, blocks, n_boot=1000, seed=0)
    assert paired.estimate == pytest.approx(np.mean(a - b))
    assert paired.lo > 0
    unpaired_width = (
        block_bootstrap(a, blocks, n_boot=1000, seed=0).hi - block_bootstrap(a, blocks, n_boot=1000, seed=0).lo
    )
    assert (paired.hi - paired.lo) < unpaired_width / 5


def test_paired_diff_requires_equal_lengths():
    with pytest.raises(ValueError):
        paired_block_bootstrap_diff([1.0, 2.0], [1.0], [0, 1], n_boot=10)


# --------------------------------------------------------------------------- MDE


def test_min_detectable_effect_known_values():
    one = (norm.ppf(0.95) + norm.ppf(0.8)) * 1.0 / 10.0
    two = (norm.ppf(0.975) + norm.ppf(0.8)) * 1.0 / 10.0
    assert min_detectable_effect(1.0, 100) == pytest.approx(one)
    assert min_detectable_effect(1.0, 100) == pytest.approx(0.24865, abs=1e-4)
    assert min_detectable_effect(1.0, 100, one_sided=False) == pytest.approx(two)
    assert min_detectable_effect(2.0, 400, alpha=0.01, power=0.9) == pytest.approx(
        (norm.ppf(0.99) + norm.ppf(0.9)) * 2.0 / 20.0
    )


def test_min_detectable_effect_validates():
    with pytest.raises(ValueError):
        min_detectable_effect(1.0, 0)
    with pytest.raises(ValueError):
        min_detectable_effect(1.0, 10, power=1.0)
