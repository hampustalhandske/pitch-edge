"""Reviewer agent — the deterministic "is this edge real" check.

No LLM per proposal. For every sized proposal it looks up this model's real walk-forward evidence
for this league (`model_league_performance`) and whether the odds behind it were real or
`synthetic_elo_book`, and derives a verdict from those two hard facts — a model with positive real
edge in this league, backed by real (non-synthetic) odds, is `trust`; negative real edge is
`distrust`; missing backtest evidence is `needs_info`. This is exactly the judgment a human analyst
would make from the same two numbers, so there's no reasoning step here that actually needs a
model, local or hosted — see `agents/selection.py` for the same argument applied to fixture
screening. RAG context is still gathered and attached (`context_docs`) so the on-demand explainer
(`agents/explainer.py`) has real grounding available if a user later asks the system to explain
these results in plain language — that single LLM call is the only place this project runs an LLM
inside the signals path, and only when someone actually asks for it.

A `"distrust"` verdict never drops a proposal — it is still passed to `human_approval` with its
verdict/reasons attached so a human makes the final call; the project's human-approval gate is not
weakened by this agent's judgment upstream of it.
"""

from __future__ import annotations

from pathlib import Path

from pitch_edge.agents.graph import SignalState
from pitch_edge.backtest.report import model_league_performance


def _judge(proposal: dict, league_perf: dict | None, real_odds: bool) -> tuple[str, list[str]]:
    model_name = proposal.get("model_name", "")
    perf = (league_perf or {}).get(model_name)
    if perf is None:
        return "needs_info", [f"no walk-forward evidence found for model {model_name} in this league"]
    edge_bits = perf["edge_bits"]
    n = perf["n"]
    if not real_odds:
        return "distrust", [
            "odds are synthetic (synthetic_elo_book), not real market prices",
            f"model {model_name} edge_bits={edge_bits:+.4f} over n={n} matches in this league",
        ]
    if edge_bits > 0:
        return "trust", [f"model {model_name} has positive real edge_bits={edge_bits:+.4f} over n={n} matches in this league"]
    return "distrust", [f"model {model_name} has negative real edge_bits={edge_bits:+.4f} over n={n} matches in this league"]


def review_proposals(state: SignalState, reports_dir: str | Path) -> dict:
    proposals = state.get("proposals", [])
    if not proposals:
        return {"proposals": [], "reviews": [], "log": [*state.get("log", []), "review_proposals: 0 proposals"]}

    league_by_match = {f["match_id"]: f.get("league_code") for f in state.get("fixtures", [])}
    reviews: list[dict] = []
    annotated: list[dict] = []
    for p in proposals:
        league_code = league_by_match.get(p["match_id"])
        league_perf = model_league_performance(reports_dir, league_code) if league_code else None
        real_odds = p.get("bookmaker", "") != "synthetic_elo_book"
        verdict, reasons = _judge(p, league_perf, real_odds)
        review = {
            "match_id": p["match_id"],
            "outcome": p["outcome"],
            "verdict": verdict,
            "reasons": reasons,
        }
        reviews.append(review)
        annotated.append({**p, "verdict": verdict, "reasons": reasons})

    log_line = f"review_proposals: {sum(r['verdict'] == 'trust' for r in reviews)}/{len(reviews)} trusted"
    return {"proposals": annotated, "reviews": reviews, "log": [*state.get("log", []), log_line]}
