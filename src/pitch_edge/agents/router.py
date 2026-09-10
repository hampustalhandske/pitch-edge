"""Model router — picks whichever model has the best real walk-forward edge in a league.

Implements the project's actual design decision: models are trusted per-league only on the
strength of `backtest/report.py::league_table()` evidence (`data/backtest/main/by_league.csv`), never by
assumption or a single hardcoded default. The router always returns the model with the highest
`edge_bits` for that league among `available` — even when that edge is negative, i.e. the least-bad
model rather than none at all — so a league gets `None` only when there is no backtest evidence for
it whatsoever (the `by_league.csv` file or that league's rows are simply missing). Downstream
proposal ranking (`RiskManager.size()`) still ranks and prices strictly by each fixture's own edge,
so a bad model backing a fixture just means that fixture is unlikely to clear `min_edge` there —
the router choosing a model is not the same claim as a proposal being worth trusting.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pitch_edge.backtest.report import model_league_performance
from pitch_edge.models.base import MatchModel

logger = logging.getLogger(__name__)


def select_model_for_league(
    league_code: str, reports_dir: str | Path, available: list[MatchModel]
) -> MatchModel | None:
    performance = model_league_performance(reports_dir, league_code)
    if not performance:
        logger.info("no backtest evidence for league %s — no model trusted here", league_code)
        return None
    candidates = [(name, stats) for name, stats in performance.items() if name in {m.name for m in available}]
    if not candidates:
        logger.info(
            "no evidence for any available model in league %s (checked %s) — skipping inference",
            league_code,
            sorted(performance),
        )
        return None
    best_name, best_stats = max(candidates, key=lambda kv: kv[1]["edge_bits"])
    by_name = {m.name: m for m in available}
    logger.info(
        "league %s -> model %s (best available edge_bits=%.4f, n=%d)",
        league_code,
        best_name,
        best_stats["edge_bits"],
        best_stats["n"],
    )
    return by_name[best_name]
