"""The `ask` LangGraph: intent routing, the pure fixtures-on-day lookup, and an end-to-end
top-N-bets pass with the judge's LLM stubbed out (matching this repo's established pattern of
stubbing the agent-building call rather than `create_react_agent` itself)."""

from __future__ import annotations

import pandas as pd
import pytest

from pitch_edge.agents import qa_graph as qg
from pitch_edge.agents.qa_graph import Answer, FixtureNote, ask_question, build_qa_graph
from pitch_edge.models.base import MatchModel
from pitch_edge.rag.query_parser import AskIntent

pytestmark = pytest.mark.unit


class _FixedModel(MatchModel):
    def __init__(self, name: str, home: float, draw: float, away: float):
        self.name = name
        self._p = (home, draw, away)

    def fit(self, train):
        return self

    def predict_proba(self, X):
        h, d, a = self._p
        return pd.DataFrame({"home": h, "draw": d, "away": a}, index=X.index)


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


class _FakeMessage:
    def __init__(self, content: str):
        self.content = content


class _StubJudgeAgent:
    def __init__(self, transcript: str = "Arsenal vs Chelsea: good_model shows a real edge here, backed by real evidence."):
        self._transcript = transcript
        self.calls = 0

    def invoke(self, payload, config=None):
        self.calls += 1
        return {"messages": [_FakeMessage(self._transcript)]}


class _FakeIndex:
    def query(self, text, k=6, where=None):
        return []


def _features_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "match_id": ["m1", "m2"],
            "date": ["2024-06-01", "2024-06-03"],
            "home_team": ["Arsenal", "Liverpool"],
            "away_team": ["Chelsea", "ManCity"],
            "league_code": ["E0", "E0"],
            "referee": ["M. Oliver", "A. Taylor"],
            "sv_missing_pct_diff": [0.01, 0.01],
            "PSH": [1.8, 2.5],
            "PSD": [3.5, 3.2],
            "PSA": [4.5, 2.8],
        }
    )


def _slice_evidence() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": "good_model",
                "slice_dim": "league_code",
                "slice_value": "E0",
                "n": 200,
                "edge_bits": 0.03,
                "q_value": 0.01,
                "significant": True,
                "checkpoint_date": "2023-01-01",
            },
            {
                "model": "bad_model",
                "slice_dim": "league_code",
                "slice_value": "E0",
                "n": 200,
                "edge_bits": -0.01,
                "q_value": 0.5,
                "significant": False,
                "checkpoint_date": "2023-01-01",
            },
        ]
    )


def _models() -> list[MatchModel]:
    return [_FixedModel("good_model", 0.7, 0.2, 0.1), _FixedModel("bad_model", 0.3, 0.3, 0.4)]


def test_unrecognized_question_returns_cannot_answer(monkeypatch):
    graph = build_qa_graph(_features_frame(), _slice_evidence(), _FakeIndex(), _models(), llm=_StubLLM(AskIntent(intent="unrecognized")))
    state = ask_question(graph, "what's the weather like", pd.Timestamp("2024-06-01"))
    assert state.intent == "unrecognized"
    assert state.message and "top N bets" in state.message
    assert state.answer is None


def test_fixtures_on_day_lists_without_predicting():
    llm = _StubLLM(AskIntent(intent="fixtures_on_day", day="2024-06-01"))
    graph = build_qa_graph(_features_frame(), _slice_evidence(), _FakeIndex(), _models(), llm=llm)
    state = ask_question(graph, "what's on 2024-06-01", pd.Timestamp("2024-01-01"))
    assert state.intent == "fixtures_on_day"
    assert "Arsenal vs Chelsea" in state.message
    assert state.answer is None
    assert state.predictions == []  # no model ever ran


def test_top_bets_selects_the_significant_model_and_calls_judge_once(monkeypatch):
    # m2 (Liverpool vs ManCity) has the larger good_model edge of the two candidates, so it — not
    # m1 — is the one that survives ranking down to the top 1.
    stub_answer = Answer(
        overview="One clear edge found.",
        notes=[FixtureNote(match_id="m2", take="Liverpool vs ManCity: good_model shows a real edge here.")],
    )
    stub_agent = _StubJudgeAgent()
    monkeypatch.setattr(qg, "_build_judge_agent", lambda llm, tools: stub_agent)
    monkeypatch.setattr(qg, "_structure_answer", lambda llm, transcript, n_fixtures: stub_answer)

    llm = _StubLLM(AskIntent(intent="top_bets", n=1))
    graph = build_qa_graph(_features_frame(), _slice_evidence(), _FakeIndex(), _models(), llm=llm)
    state = ask_question(graph, "top 1 bet", pd.Timestamp("2024-05-25"))

    assert state.intent == "top_bets"
    assert stub_agent.calls == 1
    assert state.trusted == {"m1": "good_model", "m2": "good_model"}
    assert state.answer is not None
    assert state.answer.backend == "llm"
    assert state.answer.citations_grounded
    # Team names are backfilled deterministically, not trusted to the LLM's own repetition of them.
    assert state.answer.notes[0].home_team == "Liverpool"
    assert state.answer.notes[0].away_team == "ManCity"


def test_judge_falls_back_to_template_when_llm_unavailable(monkeypatch):
    def _raise(llm, tools):
        raise RuntimeError("ollama not running")

    monkeypatch.setattr(qg, "_build_judge_agent", _raise)
    llm = _StubLLM(AskIntent(intent="top_bets", n=2))
    graph = build_qa_graph(_features_frame(), _slice_evidence(), _FakeIndex(), _models(), llm=llm)
    state = ask_question(graph, "top bets", pd.Timestamp("2024-05-25"))

    assert state.answer is not None
    assert state.answer.backend == "template"
    assert len(state.answer.notes) == 2
