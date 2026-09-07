from __future__ import annotations

import pytest

from pitch_edge.odds.utils import detect_edges, implied_probability, no_vig_probabilities, overround

pytestmark = pytest.mark.unit


def test_implied_probability():
    assert implied_probability(2.0) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        implied_probability(1.0)


def test_overround_positive_for_real_book():
    assert overround(1.80, 3.60, 4.50) > 0


def test_no_vig_sums_to_one():
    probs = no_vig_probabilities(1.80, 3.60, 4.50)
    assert sum(probs) == pytest.approx(1.0) and all(0 < p < 1 for p in probs)


def test_detect_edges():
    edges = detect_edges({"home": 0.6, "draw": 0.2, "away": 0.2}, 2.2, 3.4, 3.3)
    home = next(e for e in edges if e.outcome == "home")
    assert home.has_positive_edge
    market = no_vig_probabilities(1.8, 3.6, 4.5)
    flat = detect_edges(dict(zip(("home", "draw", "away"), market, strict=True)), 1.8, 3.6, 4.5)
    assert all(abs(e.edge) < 1e-9 for e in flat)
