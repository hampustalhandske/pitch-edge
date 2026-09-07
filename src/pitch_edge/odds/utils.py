"""No-vig (overround-removed) fair-probability math and edge detection.

Standard sports-trading vocabulary: a bookmaker's quoted decimal odds imply
probabilities that sum to > 1 (the overround / vig). Removing it via simple
normalization (there are fancier methods — Shin's method, power method — but
normalization is the standard baseline) gives the market's "fair" implied
probability, which is what a model's own probability should be compared
against to claim an edge.
"""

from __future__ import annotations

from dataclasses import dataclass


def implied_probability(decimal_odds: float) -> float:
    if decimal_odds <= 1.0:
        raise ValueError(f"decimal odds must be > 1.0, got {decimal_odds}")
    return 1.0 / decimal_odds


def overround(home_odds: float, draw_odds: float, away_odds: float) -> float:
    """Sum of implied probabilities minus 1; the bookmaker's margin."""
    return sum(implied_probability(o) for o in (home_odds, draw_odds, away_odds)) - 1.0


def no_vig_probabilities(home_odds: float, draw_odds: float, away_odds: float) -> tuple[float, float, float]:
    """Normalize implied probabilities to sum to 1 (basic no-vig method)."""
    raw = [implied_probability(o) for o in (home_odds, draw_odds, away_odds)]
    total = sum(raw)
    return tuple(p / total for p in raw)  # type: ignore[return-value]


@dataclass(frozen=True)
class Edge:
    outcome: str  # "home" | "draw" | "away"
    model_probability: float
    market_probability: float
    edge: float  # model_probability - market_probability
    decimal_odds: float

    @property
    def has_positive_edge(self) -> bool:
        return self.edge > 0


def detect_edges(
    model_probs: dict[str, float],
    home_odds: float,
    draw_odds: float,
    away_odds: float,
) -> list[Edge]:
    """Compare model probabilities against no-vig market probabilities per outcome.

    model_probs must have keys "home", "draw", "away" summing to ~1.
    """
    market_home, market_draw, market_away = no_vig_probabilities(home_odds, draw_odds, away_odds)
    market = {"home": market_home, "draw": market_draw, "away": market_away}
    odds = {"home": home_odds, "draw": draw_odds, "away": away_odds}

    edges = []
    for outcome in ("home", "draw", "away"):
        model_p = model_probs[outcome]
        market_p = market[outcome]
        edges.append(
            Edge(
                outcome=outcome,
                model_probability=model_p,
                market_probability=market_p,
                edge=model_p - market_p,
                decimal_odds=odds[outcome],
            )
        )
    return edges
