"""Predictions explainer — the only place an LLM runs in the signals path, and only on request.

Everything analytical already happened deterministically before this module is ever touched:
`agents/router.py` picked each fixture's model from real backtest evidence, `edge_detector`/
`RiskManager` computed and sized every proposal, and `agents/reviewer.py` derived a trust/distrust
verdict from real numbers (edge_bits sign, real-vs-synthetic odds). This function's only job is
turning that already-complete analysis into a short natural-language explanation for a human to
read — one LLM call over the whole batch, not one per fixture or per proposal. It is invoked
exclusively by `pitch-edge predict` (see cli.py), i.e. only when a user actually asks the system
for predictions — never automatically inside `agentic-signals` or any other pipeline stage.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from pitch_edge.agents.llm import get_local_llm

logger = logging.getLogger(__name__)


class ProposalNote(BaseModel):
    match_id: str
    outcome: str
    take: str = Field(description="One short sentence on this specific proposal, grounded in its own numbers")


class SignalsExplanation(BaseModel):
    overview: str = Field(description="2-3 sentences summarizing the batch of proposals as a whole")
    notes: list[ProposalNote] = Field(default_factory=list)
    backend: str = "template"


def _compact_summary(proposals: list[dict], reviews: list[dict]) -> str:
    reviews_by_key = {(r["match_id"], r["outcome"]): r for r in reviews}
    lines = []
    for p in proposals:
        r = reviews_by_key.get((p["match_id"], p["outcome"]), {})
        lines.append(
            f"- {p['home_team']} vs {p['away_team']} [{p['match_id']}], outcome={p['outcome']}: "
            f"model {p.get('model_name', '')} {p['model_probability']:.1%} vs market {p['market_probability']:.1%} "
            f"(edge {p['edge']:+.1%}) at odds {p['decimal_odds']}, stake {p['stake']:.2f}; "
            f"reviewer verdict={r.get('verdict', 'unknown')} ({'; '.join(r.get('reasons', []))})"
        )
    return "\n".join(lines)


def _prompt(proposals: list[dict], reviews: list[dict]) -> str:
    return (
        "Below is a fully-computed batch of proposed signals from a football prediction pipeline. "
        "Every number (model probability, market probability, edge, stake, reviewer verdict) was "
        "already calculated deterministically — you are not estimating anything, only explaining "
        "what is given, in plain language for a human who will approve or reject each one.\n\n"
        f"{_compact_summary(proposals, reviews)}\n\n"
        "Fill in BOTH fields of the response schema:\n"
        "1. `overview`: 2-3 sentences summarizing the batch (how many proposals, general quality "
        "per the reviewer verdicts).\n"
        f"2. `notes`: exactly {len(proposals)} entries, one per proposal listed above (same "
        "match_id/outcome), each a short grounded sentence (cite only the numbers given above, "
        "never invent a number). Do not leave `notes` empty.\n"
        "Never suggest an action beyond describing what the numbers show — this is not a betting "
        "instruction."
    )


def _template_explanation(proposals: list[dict], reviews: list[dict]) -> SignalsExplanation:
    reviews_by_key = {(r["match_id"], r["outcome"]): r for r in reviews}
    n_trust = sum(1 for r in reviews if r.get("verdict") == "trust")
    overview = (
        f"{len(proposals)} proposal(s), {n_trust} with a 'trust' reviewer verdict "
        f"(template summary — no LLM configured)."
    )
    notes = [
        ProposalNote(
            match_id=p["match_id"],
            outcome=p["outcome"],
            take=(
                f"{p['home_team']} vs {p['away_team']} {p['outcome']}: edge {p['edge']:+.1%} at "
                f"{p['decimal_odds']}, reviewer={reviews_by_key.get((p['match_id'], p['outcome']), {}).get('verdict', 'unknown')}."
            ),
        )
        for p in proposals
    ]
    return SignalsExplanation(overview=overview, notes=notes, backend="template")


def explain_signals(proposals: list[dict], reviews: list[dict], llm_model: str | None = None) -> SignalsExplanation:
    """One LLM call over the whole batch (falls back to a plain template if Ollama isn't running —
    this must never block a user from seeing their proposals)."""
    if not proposals:
        return SignalsExplanation(overview="No proposals to explain.", notes=[], backend="template")
    try:
        llm = get_local_llm(model=llm_model)
        structured = llm.with_structured_output(SignalsExplanation)
        result = structured.invoke(_prompt(proposals, reviews))
        parsed = result if isinstance(result, SignalsExplanation) else SignalsExplanation.model_validate(result)
        return SignalsExplanation(
            overview=parsed.overview, notes=parsed.notes, backend=f"ollama:{llm_model or 'default'}"
        )
    except Exception as exc:  # noqa: BLE001 - LLM unavailable must never block seeing real proposals
        logger.warning("explain_signals: LLM unavailable (%s) — falling back to template", exc)
        return _template_explanation(proposals, reviews)
