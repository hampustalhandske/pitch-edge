"""Local-LLM query parsing: extracts team names for retrieval, degrades to raw-question topic
when the local LLM is unreachable or returns nothing usable — never breaks RAG."""

from __future__ import annotations

import pytest

from pitch_edge.rag.query_parser import ParsedQuery, parse_query, retrieval_query_text

pytestmark = pytest.mark.unit


class _StubStructuredLLM:
    def __init__(self, result):
        self._result = result

    def invoke(self, prompt):
        return self._result


class _StubLLM:
    def __init__(self, result):
        self._result = result

    def with_structured_output(self, model):
        return _StubStructuredLLM(self._result)


class _RaisingLLM:
    def with_structured_output(self, model):
        class _Raiser:
            def invoke(self, prompt):
                raise RuntimeError("ollama not running")

        return _Raiser()


def test_parse_query_extracts_team_names():
    llm = _StubLLM(ParsedQuery(home_team="Arsenal", away_team="Chelsea", topic="Arsenal vs Chelsea outlook"))
    parsed = parse_query("How does the model view Arsenal vs Chelsea this weekend?", llm=llm)
    assert parsed.home_team == "Arsenal" and parsed.away_team == "Chelsea"
    assert retrieval_query_text(parsed) == "Arsenal vs Chelsea team news injuries form"


def test_parse_query_no_teams_falls_back_to_topic():
    llm = _StubLLM(ParsedQuery(home_team=None, away_team=None, topic="what is fractional Kelly staking"))
    parsed = parse_query("What is fractional Kelly staking?", llm=llm)
    assert parsed.home_team is None
    assert retrieval_query_text(parsed) == "what is fractional Kelly staking"


def test_parse_query_degrades_when_llm_unavailable():
    parsed = parse_query("Arsenal vs Chelsea?", llm=_RaisingLLM())
    assert parsed.home_team is None and parsed.away_team is None
    assert retrieval_query_text(parsed) == "Arsenal vs Chelsea?"
