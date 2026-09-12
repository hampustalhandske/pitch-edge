"""Local-LLM query parsing: structures a question into the `ask` agent's three standardized
intents, degrading to `None` ("I can't understand that") on anything else or an unreachable LLM."""

from __future__ import annotations

import pytest

from pitch_edge.rag.query_parser import (
    AskIntent,
    FixtureQuery,
    FixturesOnDayQuery,
    TopBetsQuery,
    parse_ask_intent,
)

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


def test_parse_ask_intent_top_bets():
    llm = _StubLLM(AskIntent(intent="top_bets", n=6))
    parsed = parse_ask_intent("give me the top 6 bets", llm=llm)
    assert parsed == TopBetsQuery(n=6)


def test_parse_ask_intent_top_bets_defaults_n_to_four():
    llm = _StubLLM(AskIntent(intent="top_bets", n=None))
    parsed = parse_ask_intent("what are the best bets", llm=llm)
    assert parsed == TopBetsQuery(n=4)


def test_parse_ask_intent_fixture():
    llm = _StubLLM(AskIntent(intent="fixture", home_team="Liverpool", away_team="Arsenal"))
    parsed = parse_ask_intent("Liverpool vs Arsenal?", llm=llm)
    assert parsed == FixtureQuery(home_team="Liverpool", away_team="Arsenal")


def test_parse_ask_intent_fixture_missing_a_team_is_unrecognized():
    llm = _StubLLM(AskIntent(intent="fixture", home_team="Liverpool", away_team=None))
    assert parse_ask_intent("Liverpool vs someone?", llm=llm) is None


def test_parse_ask_intent_fixtures_on_day():
    llm = _StubLLM(AskIntent(intent="fixtures_on_day", day="2024-06-01"))
    parsed = parse_ask_intent("what's on 2024-06-01", llm=llm)
    assert parsed == FixturesOnDayQuery(day="2024-06-01")


def test_parse_ask_intent_unrecognized():
    llm = _StubLLM(AskIntent(intent="unrecognized"))
    assert parse_ask_intent("what's the weather like", llm=llm) is None


def test_parse_ask_intent_degrades_to_none_when_llm_unavailable():
    assert parse_ask_intent("top 4 bets", llm=_RaisingLLM()) is None
