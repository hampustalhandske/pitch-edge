"""Local-LLM query parsing for the `ask` agent.

`parse_ask_intent` structures a free-text question into one of the `ask` agent's three
standardized shapes (top N bets / one fixture / fixtures on a day), using the same local Ollama
model as the rest of the project (`agents/llm.py`) — no API key needed.
"""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class AskIntent(BaseModel):
    """Structured-output target for `parse_ask_intent` — the `ask` agent's three standardized
    question shapes, plus a catch-all for anything else."""

    intent: Literal["top_bets", "fixture", "fixtures_on_day", "unrecognized"]
    n: int | None = Field(default=None, description="For 'top_bets': how many bets were asked for")
    home_team: str | None = Field(default=None, description="For 'fixture': the first team named")
    away_team: str | None = Field(default=None, description="For 'fixture': the second team named")
    day: str | None = Field(default=None, description="For 'fixtures_on_day': the date named, as ISO YYYY-MM-DD")


class TopBetsQuery(BaseModel):
    n: int


class FixtureQuery(BaseModel):
    home_team: str
    away_team: str


class FixturesOnDayQuery(BaseModel):
    day: str


_ASK_INTENT_PROMPT = (
    "Classify this question about football betting signals into exactly one of three shapes, or "
    "'unrecognized' if it fits none of them:\n"
    "1. 'top_bets': asking for the N best/top bets right now (extract n; default 4 if unspecified "
    "but clearly this shape, e.g. 'give me the top bets').\n"
    "2. 'fixture': asking for a prediction on one specific match between two named teams (extract "
    "home_team and away_team).\n"
    "3. 'fixtures_on_day': asking what fixtures/matches are happening on a given day (extract day "
    "as an ISO date if one is stated or clearly implied; otherwise use 'unrecognized').\n"
    "Never guess team names or a day that isn't actually in the question. If the question asks "
    "anything else (general chit-chat, a question about a concept, no specific bets/fixture/day), "
    "use 'unrecognized'.\n\n"
    "Question: {question!r}"
)


def parse_ask_intent(question: str, llm=None) -> TopBetsQuery | FixtureQuery | FixturesOnDayQuery | None:
    """Structures `question` into one of the `ask` agent's three standardized intents, or `None`
    if the local LLM can't confidently place it into one of them (missing required fields, or the
    LLM itself unreachable/malformed) — the caller renders `None` as "I can't understand that."""
    try:
        if llm is None:
            from pitch_edge.agents.llm import get_llm_for

            llm = get_llm_for("fast")
        # method="json_schema" — the function-calling default trips a tool-name hallucination on
        # Groq's gpt-oss models (`Tool call validation failed: ... which was not in request.tools`);
        # JSON-schema-constrained decoding is confirmed reliable and fast on both Ollama and Groq.
        structured = llm.with_structured_output(AskIntent, method="json_schema")
        parsed: AskIntent = structured.invoke(_ASK_INTENT_PROMPT.format(question=question))
    except Exception as exc:  # noqa: BLE001 - local LLM unavailable must never raise
        logger.info("parse_ask_intent: local LLM unavailable (%s)", exc)
        return None
    if parsed.intent == "top_bets":
        return TopBetsQuery(n=parsed.n or 4)
    if parsed.intent == "fixture" and parsed.home_team and parsed.away_team:
        return FixtureQuery(home_team=parsed.home_team, away_team=parsed.away_team)
    if parsed.intent == "fixtures_on_day" and parsed.day:
        return FixturesOnDayQuery(day=parsed.day)
    return None
