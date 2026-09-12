"""Stage 1 of the `ask` agent: a fixture's circumstance-slice values and retrieved context docs.

Deterministic, except for one narrow, bounded LLM call (`squad_value_news_hint`) that reads
retrieved news text for clear evidence of a squad-value imbalance (injuries, suspensions,
rotation) the structured feature store doesn't already have. It never invents a new feature slot,
never touches a probability, and its output is handed to Stage 2 as a separately-tagged note —
never silently blended into the structured circumstance values a model was actually trained on.
"""

from __future__ import annotations

import logging

import pandas as pd
from pydantic import BaseModel, Field

from pitch_edge.backtest.slices import fixture_slice_values
from pitch_edge.rag.documents import Document
from pitch_edge.rag.fixture_context import gather_fixture_context
from pitch_edge.rag.index import VectorIndex

logger = logging.getLogger(__name__)


class SquadValueHint(BaseModel):
    has_clear_evidence: bool = Field(description="True only if the text clearly describes a squad-value imbalance")
    note: str = Field(default="", description="One short sentence citing what the text said, empty if no evidence")


def gather_context(index: VectorIndex, home: str, away: str, match_id: str, as_of: pd.Timestamp, k: int = 6) -> list[Document]:
    return gather_fixture_context(index, home, away, match_id, k=k, as_of=str(as_of.date()))


def squad_value_news_hint(context_docs: list[Document], llm=None) -> str | None:
    """`None` unless retrieved news text gives clear evidence of a squad-value imbalance the
    structured feature store might be missing. Never invents a number — the note is prose, not a
    probability or a feature value, and is surfaced to Stage 2 as an explicitly unconfirmed hint."""
    if not context_docs:
        return None
    try:
        if llm is None:
            from pitch_edge.agents.llm import get_llm_for

            llm = get_llm_for("fast")
        structured = llm.with_structured_output(SquadValueHint, method="json_schema")
        text = "\n".join(d.text for d in context_docs[:6])
        hint: SquadValueHint = structured.invoke(
            "From this retrieved team news, is there clear evidence of a squad-value imbalance "
            "between the two sides (key injuries, suspensions, or heavy rotation for one side but "
            "not the other)? Only say yes if the text clearly supports it; otherwise say no. "
            "Never guess.\n\n" + text
        )
    except Exception as exc:  # noqa: BLE001 - a broken local LLM must never block an answer
        logger.info("squad_value_news_hint: local LLM unavailable (%s)", exc)
        return None
    return hint.note.strip() or None if hint.has_clear_evidence else None


def structure_context(
    fixture_row: pd.Series,
    index: VectorIndex,
    as_of: pd.Timestamp,
    home: str,
    away: str,
    match_id: str,
    llm=None,
) -> tuple[dict[str, str], list[Document], str | None]:
    """Returns `(slice_values, context_docs, squad_value_hint)` for one fixture. `slice_values`
    always comes from the structured feature store (`fixture_slice_values`); `squad_value_hint` is
    only populated from retrieved news when the structured `squad_value_gap_bucket` is missing."""
    docs = gather_context(index, home, away, match_id, as_of)
    slices = fixture_slice_values(fixture_row)
    hint = None
    if "squad_value_gap_bucket" not in slices:
        hint = squad_value_news_hint(docs, llm=llm)
    return slices, docs, hint
