"""Opt-in, genuinely tool-calling reviewer — an alternative to `agents.reviewer.review_proposals`.

The default reviewer (`agents/reviewer.py`) is deliberately deterministic: it already has both
numbers it needs (backtest evidence, real-vs-synthetic odds) handed to it, so there is no reasoning
step that benefits from an LLM. This module instead gives a local LLM (`agents/llm.py`, Ollama —
never a paid API) two tools, `backtest_evidence` and `search_context`, and lets it decide for itself
which to call before returning a verdict — the "agent decides its own tool calls in a loop" pattern,
built on `langgraph.prebuilt.create_react_agent`.

Off by default (`Settings.agentic_reviewer_tool_calling`, env `PITCH_EDGE_REVIEWER_TOOL_CALLING`,
CLI `agentic-signals --tool-reviewer`): it is slower (one or more LLM round-trips per proposal
instead of zero) and less deterministic than `review_proposals`, so it stays an explicit opt-in
rather than the default path. Same non-negotiables as everywhere else in this project: the LLM only
ever cites what a tool returns, never invents a number, and a 'distrust'/'needs_info' verdict is
still passed through to `human_approval` rather than dropping the proposal — this agent's judgment
never weakens or bypasses the human-approval gate.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from pitch_edge.agents.graph import SignalState
from pitch_edge.agents.llm import get_local_llm
from pitch_edge.agents.reviewer import _judge
from pitch_edge.backtest.report import model_league_performance
from pitch_edge.rag.index import VectorIndex

logger = logging.getLogger(__name__)


class ToolReviewVerdict(BaseModel):
    verdict: Literal["trust", "distrust", "needs_info"]
    reasons: list[str] = Field(description="Short reasons, grounded only in what a tool call returned")


def _make_tools(reports_dir: str | Path, index: VectorIndex) -> list:
    @tool
    def backtest_evidence(model_name: str, league_code: str) -> str:
        """Look up this model's real walk-forward edge_bits/log_loss/n in this league. Returns
        'no evidence' if none exists — never invent a number when that happens."""
        perf = model_league_performance(reports_dir, league_code)
        row = (perf or {}).get(model_name)
        if row is None:
            return f"no walk-forward evidence for model={model_name} in league={league_code}"
        return (
            f"model={model_name} league={league_code} edge_bits={row['edge_bits']:+.4f} "
            f"log_loss={row['log_loss']:.4f} n={row['n']}"
        )

    @tool
    def search_context(query: str) -> str:
        """Search the RAG index (news/match/prediction docs) for context relevant to this fixture,
        e.g. team news, injuries, head-to-head history."""
        hits = index.query(query, k=4)
        if not hits:
            return "no matching documents"
        return "\n".join(f"[{doc.doc_id}] {doc.text[:400]}" for doc, _score in hits)

    return [backtest_evidence, search_context]


_PROMPT = (
    "You are reviewing one proposed football betting signal for a human approver. Use the "
    "backtest_evidence tool to check this model's real walk-forward edge in this league, and "
    "search_context if team news would help judge plausibility. Never invent a number — only cite "
    "what a tool returns. Then decide: 'trust' (positive real edge_bits AND real market odds), "
    "'distrust' (negative edge_bits, or synthetic odds), or 'needs_info' (no backtest evidence "
    "found for this model/league). This verdict is advisory only — a human still approves or "
    "rejects the proposal either way.\n\n"
    "Proposal: {home} vs {away} [{match_id}], outcome={outcome}, model={model_name}, "
    "league={league_code}, bookmaker={bookmaker} (real market odds unless bookmaker is "
    "'synthetic_elo_book')."
)


def _build_agent(reports_dir: str | Path, index: VectorIndex, llm_model: str | None):
    llm = get_local_llm(model=llm_model)
    return create_react_agent(llm, tools=_make_tools(reports_dir, index), response_format=ToolReviewVerdict)


def _agentic_judge(agent, proposal: dict, league_code: str | None) -> tuple[str, list[str]]:
    prompt = _PROMPT.format(
        home=proposal.get("home_team", ""),
        away=proposal.get("away_team", ""),
        match_id=proposal["match_id"],
        outcome=proposal["outcome"],
        model_name=proposal.get("model_name", ""),
        league_code=league_code or "unknown",
        bookmaker=proposal.get("bookmaker", ""),
    )
    result = agent.invoke({"messages": [HumanMessage(prompt)]})
    parsed: ToolReviewVerdict = result["structured_response"]
    return parsed.verdict, parsed.reasons


def review_proposals_agentic(
    state: SignalState, reports_dir: str | Path, index: VectorIndex, llm_model: str | None = None
) -> dict:
    """Same input/output shape as `agents.reviewer.review_proposals` so it drops into
    `agents/orchestrator.py` as a straight swap. Falls back to the deterministic `_judge` per
    proposal on any LLM/tool failure (Ollama not running, malformed structured output) — a broken
    local LLM must never block a run, matching `agents/explainer.py`'s fallback policy."""
    proposals = state.get("proposals", [])
    if not proposals:
        return {
            "proposals": [],
            "reviews": [],
            "log": [*state.get("log", []), "review_proposals_agentic: 0 proposals"],
        }

    league_by_match = {f["match_id"]: f.get("league_code") for f in state.get("fixtures", [])}
    agent = _build_agent(reports_dir, index, llm_model)

    reviews: list[dict] = []
    annotated: list[dict] = []
    n_fallback = 0
    for p in proposals:
        league_code = league_by_match.get(p["match_id"])
        try:
            verdict, reasons = _agentic_judge(agent, p, league_code)
        except Exception as exc:  # noqa: BLE001 - LLM/tool failure must never block a run
            logger.warning(
                "review_proposals_agentic: LLM unavailable for %s (%s) — falling back to deterministic judge",
                p["match_id"],
                exc,
            )
            league_perf = model_league_performance(reports_dir, league_code) if league_code else None
            real_odds = p.get("bookmaker", "") != "synthetic_elo_book"
            verdict, reasons = _judge(p, league_perf, real_odds)
            n_fallback += 1
        review = {"match_id": p["match_id"], "outcome": p["outcome"], "verdict": verdict, "reasons": reasons}
        reviews.append(review)
        annotated.append({**p, "verdict": verdict, "reasons": reasons})

    fallback_note = f" ({n_fallback} fell back to deterministic judge)" if n_fallback else ""
    log_line = f"review_proposals_agentic: {sum(r['verdict'] == 'trust' for r in reviews)}/{len(reviews)} trusted{fallback_note}"
    return {"proposals": annotated, "reviews": reviews, "log": [*state.get("log", []), log_line]}
