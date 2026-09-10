"""Local-LLM query parsing for the RAG layer.

Turns a free-text question ("how does the model view Arsenal vs Chelsea this weekend?") into
structured entities (team names, a restated topic) so retrieval can be pointed at the right
documents instead of relying on the raw question text alone — the same
`gather_fixture_context`-style team-focused query already used by the agentic pipeline
(`rag/fixture_context.py`), reused here for the interactive `rag` CLI/dashboard question box.

Uses the same local Ollama model as the selection/reviewer agents (`agents/llm.py`) — no API key
needed. Falls back to treating the raw question as the topic (no team names extracted) if the
local model is unreachable or returns something unusable; retrieval then proceeds exactly as it
did before this module existed.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ParsedQuery(BaseModel):
    home_team: str | None = Field(default=None, description="First/home team named in the question, if any")
    away_team: str | None = Field(default=None, description="Second/away team named in the question, if any")
    topic: str = Field(description="The question restated concisely, for use as a fallback search query")


def parse_query(question: str, llm=None) -> ParsedQuery:
    """Best-effort structured extraction. Never raises — degrades to `ParsedQuery(topic=question)`
    (i.e. a no-op) if the local LLM is unreachable or its output can't be parsed."""
    try:
        if llm is None:
            from pitch_edge.agents.llm import get_local_llm

            llm = get_local_llm()
        structured = llm.with_structured_output(ParsedQuery)
        parsed = structured.invoke(
            "Extract structured information from this football question for a search system.\n"
            f"Question: {question!r}\n"
            "home_team/away_team: the specific club names mentioned, exactly as written, or null if "
            "none are named (a general question about a league, a model, or a concept has no teams).\n"
            "topic: the question restated concisely, to use as a fallback search query."
        )
        return ParsedQuery(home_team=parsed.home_team, away_team=parsed.away_team, topic=parsed.topic or question)
    except Exception as exc:  # noqa: BLE001 - local LLM unavailable must never break RAG retrieval
        logger.info("parse_query: local LLM unavailable (%s); using the raw question", exc)
        return ParsedQuery(topic=question)


def retrieval_query_text(parsed: ParsedQuery) -> str:
    """The text to actually search with: a team-focused query when both teams were named
    (matches `rag/fixture_context.py::gather_fixture_context`'s phrasing), else the topic."""
    if parsed.home_team and parsed.away_team:
        return f"{parsed.home_team} vs {parsed.away_team} team news injuries form"
    return parsed.topic
