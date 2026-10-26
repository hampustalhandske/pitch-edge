"""De-vig methods: probabilities sum to one, fair odds are recovered, and each method's parameter
behaves as its definition says."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pitch_edge.odds.utils import no_vig_probabilities
from pitch_edge.research.devig import METHODS, additive, devig, devig_frame, multiplicative, power, shin

pytestmark = pytest.mark.unit

BOOK = np.array([[1.80, 3.60, 4.50], [1.30, 5.50, 11.0], [2.60, 3.20, 2.90], [1.05, 13.0, 34.0]])


@pytest.mark.parametrize("method", METHODS)
def test_probabilities_sum_to_one(method):
    res = devig(BOOK, method=method)
    np.testing.assert_allclose(res.probs.sum(axis=1), 1.0, atol=1e-10)
    assert res.param.shape == (len(BOOK),)


@pytest.mark.parametrize("method", METHODS)
def test_fair_odds_are_recovered(method):
    fair = np.array([[0.5, 0.3, 0.2], [0.7, 0.2, 0.1]])
    res = devig(1.0 / fair, method=method)
    np.testing.assert_allclose(res.probs, fair, atol=1e-9)


def test_fair_odds_give_neutral_parameters():
    odds = 1.0 / np.array([[0.5, 0.3, 0.2]])
    np.testing.assert_allclose(multiplicative(odds).param, [0.0], atol=1e-12)
    np.testing.assert_allclose(power(odds).param, [1.0], atol=1e-9)
    np.testing.assert_allclose(shin(odds).param, [0.0], atol=1e-9)


def test_multiplicative_and_additive_known_values():
    pi = np.array([0.55, 0.30, 0.20])  # overround 0.05
    odds = 1.0 / pi[None, :]
    np.testing.assert_allclose(multiplicative(odds).probs[0], pi / 1.05)
    np.testing.assert_allclose(multiplicative(odds).param, [0.05])
    np.testing.assert_allclose(additive(odds).probs[0], pi - 0.05 / 3)
    np.testing.assert_allclose(additive(odds).param, [0.05])


def test_power_solves_its_defining_equation():
    res = power(BOOK)
    pi = 1.0 / BOOK
    k = res.param[:, None]
    np.testing.assert_allclose(res.probs, pi**k, rtol=1e-10)
    assert (res.param > 1).all()  # overround -> exponent above one


def test_power_handles_an_underround_book():
    odds = 1.0 / np.array([[0.5, 0.25, 0.2]])  # implied sum 0.95
    res = power(odds)
    assert res.param[0] < 1
    assert res.probs.sum() == pytest.approx(1.0)


def test_shin_z_positive_for_overround_and_satisfies_fixed_point():
    res = shin(BOOK)
    assert (res.param > 0).all()
    pi = 1.0 / BOOK
    total = pi.sum(axis=1, keepdims=True)
    z = res.param[:, None]
    # Shin (1993) implied probabilities for the solved z ...
    expected = (np.sqrt(z**2 + 4 * (1 - z) * pi**2 / total) - z) / (2 * (1 - z))
    np.testing.assert_allclose(res.probs, expected, rtol=1e-9)
    # ... and z is the Jullien-Salanie fixed point z = (sum sqrt(z^2 + 4(1-z) pi^2 / S) - 2) / (n - 2).
    n = BOOK.shape[1]
    fixed = (np.sqrt(z**2 + 4 * (1 - z) * pi**2 / total).sum(axis=1) - 2) / (n - 2)
    np.testing.assert_allclose(res.param, fixed, atol=1e-9)


def test_shin_equals_additive_for_two_outcomes():
    # Known identity: with two outcomes Shin's method removes the margin additively.
    odds = np.array([[1.40, 2.90], [1.20, 4.5], [1.05, 9.0], [1.9, 1.9]])
    np.testing.assert_allclose(shin(odds).probs, additive(odds).probs, atol=1e-10)


def test_infinite_odds_give_a_nan_row():
    for method in METHODS:
        res = devig(np.array([[1.5, np.inf, 3.0], [1.8, 3.6, 4.5]]), method=method)
        assert np.isnan(res.probs[0]).all() and np.isnan(res.param[0])
        assert res.probs[1].sum() == pytest.approx(1.0)


def test_shin_works_for_two_outcome_markets():
    res = shin(np.array([[1.90, 1.90], [1.40, 2.90]]))
    np.testing.assert_allclose(res.probs.sum(axis=1), 1.0, atol=1e-10)
    np.testing.assert_allclose(res.probs[0], [0.5, 0.5])
    assert (res.param > 0).all()


def test_power_and_shin_take_more_margin_from_the_longshot():
    odds = np.array([[1.30, 5.50, 11.0]])
    m = multiplicative(odds).probs[0]
    for method in (power, shin):
        p = method(odds).probs[0]
        assert p[2] < m[2]  # longshot shrunk more than proportionally
        assert p[0] > m[0]  # favourite keeps more


def test_one_dimensional_input_returns_one_row():
    res = devig([1.80, 3.60, 4.50], method="power")
    assert res.probs.shape == (3,)
    assert res.param.shape == ()
    assert res.probs.sum() == pytest.approx(1.0)


def test_nan_rows_propagate_without_breaking_other_rows():
    odds = np.array([[1.80, 3.60, 4.50], [np.nan, 3.0, 2.0]])
    for method in METHODS:
        res = devig(odds, method=method)
        assert np.isnan(res.probs[1]).all() and np.isnan(res.param[1])
        assert res.probs[0].sum() == pytest.approx(1.0)


def test_invalid_odds_raise():
    with pytest.raises(ValueError):
        devig([[1.0, 3.0, 4.0]], method="power")
    with pytest.raises(ValueError):
        devig([[2.0]], method="multiplicative")
    with pytest.raises(ValueError):
        devig([[2.0, 2.0]], method="nope")


def test_devig_frame_is_vectorised_row_by_row():
    df = pd.DataFrame(BOOK, columns=["h", "d", "a"], index=[10, 11, 12, 13])
    out = devig_frame(df, ["h", "d", "a"], method="shin")
    assert list(out.columns) == ["p_h", "p_d", "p_a", "shin_z"]
    assert list(out.index) == [10, 11, 12, 13]
    for i, row in enumerate(BOOK):
        np.testing.assert_allclose(out.iloc[i, :3].to_numpy(), shin(row[None, :]).probs[0])
    names = {m: devig_frame(df, ["h", "d", "a"], method=m).columns[-1] for m in METHODS}
    assert names == {
        "multiplicative": "multiplicative_overround",
        "additive": "additive_overround",
        "power": "power_k",
        "shin": "shin_z",
    }


def test_existing_no_vig_helper_is_the_multiplicative_method():
    probs = no_vig_probabilities(1.80, 3.60, 4.50)
    np.testing.assert_allclose(probs, multiplicative(np.array([[1.80, 3.60, 4.50]])).probs[0])
    assert isinstance(probs, tuple) and all(isinstance(p, float) for p in probs)
    with pytest.raises(ValueError):
        no_vig_probabilities(1.0, 3.0, 4.0)
    assert all(np.isnan(no_vig_probabilities(np.nan, 3.0, 4.0)))
