"""No-vig (overround-removed) fair-probability math and edge detection.

Standard sports-trading vocabulary: a bookmaker's quoted decimal odds imply
probabilities that sum to > 1 (the overround / vig). Removing it via simple
normalization gives the market's "fair" implied probability, which is what a
model's own probability should be compared against to claim an edge. The
normalization itself, and the power / Shin / additive alternatives, live in
`pitch_edge.research.devig`; this module is the scalar 1X2 convenience layer.
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
    """Normalize implied probabilities to sum to 1: the scalar form of `research.devig.multiplicative`
    (kept in plain Python because it is called once per quote inside loops; a test pins the two equal)."""
    raw = [implied_probability(o) for o in (home_odds, draw_odds, away_odds)]
    total = sum(raw)
    return raw[0] / total, raw[1] / total, raw[2] / total


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
