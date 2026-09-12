"""Model-probability-vs-market-probability edge computation.

Extracted from the deleted `agents/graph.py::edge_detector` node so the same no-vig/edge math is a
plain, reusable function rather than something only reachable by running a LangGraph node.
"""

from __future__ import annotations

from pitch_edge.odds.utils import no_vig_probabilities

OUTCOMES = ("home", "draw", "away")


def compute_edge(
    model_probs: dict[str, float],
    home_odds: float,
    draw_odds: float,
    away_odds: float,
) -> dict[str, dict[str, float]]:
    """model_probs: {"home": p, "draw": p, "away": p}. Returns, per outcome:
    {"model_probability", "market_probability", "edge", "decimal_odds"}.

    Raises ValueError if any odds are <= 1.0 (see `odds.utils.implied_probability`).
    """
    quotes = {"home": home_odds, "draw": draw_odds, "away": away_odds}
    mkt = dict(zip(OUTCOMES, no_vig_probabilities(home_odds, draw_odds, away_odds), strict=True))
    return {
        o: {
            "model_probability": float(model_probs[o]),
            "market_probability": mkt[o],
            "edge": float(model_probs[o]) - mkt[o],
            "decimal_odds": float(quotes[o]),
        }
        for o in OUTCOMES
    }
