"""Known-value checks for r2lib's de-vig and grading statistics. Run with pytest."""

import numpy as np
import pandas as pd
import r2lib as L


def test_multiplicative_and_power_symmetric():
    o = np.array([[2.0, 2.0], [1.9, 1.9]])
    for m in L.DEVIG:
        assert np.allclose(L.DEVIG[m](o), 0.5)


def test_shin_two_way_equals_additive():
    o = np.array([[1.5, 2.6]])
    q = 1 / o
    add = q - (q.sum() - 1) / 2
    assert np.allclose(L.devig_shin(o), add, atol=1e-9)


def test_power_shifts_margin_to_longshot():
    o = np.array([[1.4, 4.8, 8.0]])
    pm, pp = L.devig_multiplicative(o), L.devig_power(o)
    assert np.isclose(pp.sum(), 1) and pp[0, 0] > pm[0, 0] and pp[0, 2] < pm[0, 2]


def test_bh_known_values():
    adj, rej = L.bh(np.array([0.01, 0.04, 0.03, 0.2]), q=0.06)
    assert np.allclose(adj, [0.04, 0.16 / 3, 0.16 / 3, 0.2])
    assert list(rej) == [True, True, True, False]


def test_psr_half_at_benchmark():
    assert np.isclose(L.psr(0.1, 0.1, 500, 0.0, 3.0), 0.5)


def test_boot_mean_brackets_mean_and_ols_recovers_slope():
    rng = np.random.default_rng(0)
    day = pd.Series(np.repeat(np.arange(400), 5))
    x = rng.normal(1.0, 1.0, 2000)
    r = L.boot_mean(x, day)
    assert r["ci_lo"] < x.mean() < r["ci_hi"]
    z = rng.normal(size=2000)
    y = 0.5 + 2 * z + rng.normal(size=2000)
    o = L.boot_ols(y, [z], day)
    assert abs(o["coef"][1] - 2) < 0.1 and o["ci_lo"][1] < 2 < o["ci_hi"][1]


def test_pbo_low_for_dominant_config_high_for_noise():
    rng = np.random.default_rng(1)
    good = rng.normal(0, 1, (1600, 5))
    good[:, 0] += 0.5
    assert L.pbo_cscv(good) < 0.05
    assert L.pbo_cscv(rng.normal(0, 1, (1600, 5))) > 0.2
