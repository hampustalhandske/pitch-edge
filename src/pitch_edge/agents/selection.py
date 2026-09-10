"""Selection agent — data-quality screening, not "interestingness" picking.

Deliberately has NO LLM in it. Every fixture the deterministic scout finds already has a
well-defined edge-ranking step later (`RiskManager.size()`); this node's only job is judging
whether each fixture's *data* is trustworthy enough to run inference on at all — real vs synthetic
odds, confirmed vs provisional lineup, and whether the league has any backtest evidence at all for
its router-selected model (`agents/router.py::select_model_for_league` always returns the
best-available model for a league, even a negative-edge one — a fixture is only unroutable when
there is no evidence whatsoever). That is a small, fixed set of boolean facts the system already
computed — weighing them into ready/not-ready needs no model, local or hosted, per fixture. An LLM
call per fixture (the earlier design) does not scale: hundreds of fixtures means hundreds of
round-trips for a judgment this mechanical. The LLM's role in this project is now a single,
on-demand explanation over the already-computed results (see `agents/explainer.py`), never a
per-item loop.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field

from pitch_edge.agents.graph import SignalState
from pitch_edge.agents.router import select_model_for_league
from pitch_edge.data.storage import Warehouse
from pitch_edge.models import available_models as default_available_models
from pitch_edge.models.base import MatchModel
from pitch_edge.pipeline import live_quotes_from_odds_api


class FixtureReadiness(BaseModel):
    match_id: str
    ready: bool = Field(description="Data-quality judgment, NOT a betting/interest judgment")
    rationale: str
    concerns: list[str] = Field(default_factory=list)


def _fixture_concerns(fx: dict, real_odds: bool, model: MatchModel | None) -> list[str]:
    concerns = []
    if not real_odds:
        concerns.append("no live odds yet, only synthetic")
    if fx.get("lineup_source", "provisional") != "confirmed":
        concerns.append("lineup still provisional")
    if model is None:
        concerns.append("league has no backtest evidence for any available model")
    return concerns


def _judge(fx: dict, real_odds: bool, model: MatchModel | None, concerns: list[str]) -> FixtureReadiness:
    # The one fact that can never be worked around: no routable model means no prediction is
    # possible for this fixture, full stop. Everything else (synthetic odds, provisional lineup)
    # is a quality concern noted for the record but doesn't block readiness — RiskManager's own
    # edge/odds thresholds downstream are what actually decide if a fixture is worth a proposal.
    if model is None:
        return FixtureReadiness(
            match_id=fx["match_id"],
            ready=False,
            rationale="league has no backtest evidence for any available model — nothing to run",
            concerns=concerns,
        )
    rationale = (
        f"routable via {model.name}; noted concerns: {', '.join(concerns)}"
        if concerns
        else f"routable via {model.name}; no data-quality concerns"
    )
    return FixtureReadiness(match_id=fx["match_id"], ready=True, rationale=rationale, concerns=concerns)


def select_fixtures(
    state: SignalState,
    reports_dir: str | Path,
    wh: Warehouse | None = None,
    available: list[MatchModel] | None = None,
) -> dict:
    fixtures = state.get("fixtures", [])
    if not fixtures:
        return {"fixtures": [], "selected_fixtures": [], "log": [*state.get("log", []), "select_fixtures: 0 fixtures"]}

    available = available or default_available_models()
    fixtures_df = pd.DataFrame(fixtures)
    live_quotes = live_quotes_from_odds_api(wh, fixtures_df) if wh is not None else {}

    kept: list[dict] = []
    reviews: list[dict] = []
    for fx in fixtures:
        real_odds = fx["match_id"] in live_quotes
        model = select_model_for_league(fx.get("league_code", ""), reports_dir, available)
        concerns = _fixture_concerns(fx, real_odds, model)
        verdict = _judge(fx, real_odds, model, concerns)
        reviews.append(verdict.model_dump())
        if verdict.ready:
            kept.append(fx)

    log_line = f"select_fixtures: {len(kept)}/{len(fixtures)} ready ({len(fixtures) - len(kept)} screened out)"
    return {"fixtures": kept, "selected_fixtures": reviews, "log": [*state.get("log", []), log_line]}
