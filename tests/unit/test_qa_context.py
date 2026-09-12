"""Stage 1 of the `ask` agent: structured slice values, retrieved context, and the narrow
squad-value news hint (never invents a feature, degrades to no hint on any doubt)."""

from __future__ import annotations

import pandas as pd
import pytest

from pitch_edge.agents.qa_context import SquadValueHint, squad_value_news_hint, structure_context
from pitch_edge.rag.documents import Document

pytestmark = pytest.mark.unit


class _StubStructuredLLM:
    def __init__(self, result):
        self._result = result

    def invoke(self, prompt):
        return self._result


class _StubLLM:
    def __init__(self, result):
        self._result = result

    def with_structured_output(self, model, method=None):
        return _StubStructuredLLM(self._result)


class _RaisingLLM:
    def with_structured_output(self, model, method=None):
        class _Raiser:
            def invoke(self, prompt):
                raise RuntimeError("ollama not running")

        return _Raiser()


class _FakeIndex:
    def __init__(self, hits):
        self._hits = hits

    def query(self, text, k=6, where=None):
        return self._hits


def _doc(doc_id: str, text: str, doc_type: str = "news") -> Document:
    return Document(doc_id, text, {"type": doc_type})


def test_squad_value_news_hint_returns_none_without_docs():
    assert squad_value_news_hint([], llm=_StubLLM(SquadValueHint(has_clear_evidence=True, note="x"))) is None


def test_squad_value_news_hint_returns_note_on_clear_evidence():
    docs = [_doc("news:1", "Three starters ruled out for the away side.")]
    llm = _StubLLM(SquadValueHint(has_clear_evidence=True, note="Three away starters ruled out"))
    assert squad_value_news_hint(docs, llm=llm) == "Three away starters ruled out"


def test_squad_value_news_hint_returns_none_without_clear_evidence():
    docs = [_doc("news:1", "Both sides at full strength.")]
    llm = _StubLLM(SquadValueHint(has_clear_evidence=False, note=""))
    assert squad_value_news_hint(docs, llm=llm) is None


def test_squad_value_news_hint_degrades_when_llm_unavailable():
    docs = [_doc("news:1", "Some news.")]
    assert squad_value_news_hint(docs, llm=_RaisingLLM()) is None


def test_structure_context_skips_the_llm_when_structured_value_present():
    fixture_row = pd.Series({"league_code": "E0", "sv_missing_pct_diff": 0.2})
    index = _FakeIndex([(_doc("news:1", "Injury news"), 1.0)])
    llm = _StubLLM(SquadValueHint(has_clear_evidence=True, note="should not be used"))

    slices, docs, hint = structure_context(fixture_row, index, pd.Timestamp("2024-01-01"), "A", "B", "m1", llm=llm)

    assert slices["squad_value_gap_bucket"] == "large_gap"
    assert hint is None


def test_structure_context_uses_the_hint_only_when_structured_value_missing():
    fixture_row = pd.Series({"league_code": "E0"})
    index = _FakeIndex([(_doc("news:1", "Two starters missing for the home side"), 1.0)])
    llm = _StubLLM(SquadValueHint(has_clear_evidence=True, note="Two home starters missing"))

    slices, docs, hint = structure_context(fixture_row, index, pd.Timestamp("2024-01-01"), "A", "B", "m1", llm=llm)

    assert "squad_value_gap_bucket" not in slices
    assert hint == "Two home starters missing"
