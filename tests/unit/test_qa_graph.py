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
    def __init__(
        self, transcript: str = "Arsenal vs Chelsea: good_model shows a real edge here, backed by real evidence."
    ):
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
    graph = build_qa_graph(
        _features_frame(), _slice_evidence(), _FakeIndex(), _models(), llm=_StubLLM(AskIntent(intent="unrecognized"))
    )
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


# --------------------------------------------------------------------------- evidence ladder in the answer


def _evidence(floor: str = "L4", validated: bool = False) -> dict:
    return {
        "research": {
            "display_floor": floor,
            "validated_edge_exists": validated,
            "families": [
                {
                    "family": "fam-a",
                    "best_rung": "L2",
                    "champion": {"id": "fam-a-x-v1", "rung": "L2"},
                    "hypotheses": [{"id": "fam-a-x-v1", "rung": "L2", "headline": {"estimate": 0.0208}}],
                }
            ],
        },
        "paper": {"available": True, "strategies": [{"strategy_id": "soft-vs-sharp-live", "n_settled": 0}]},
    }


class _RecordingJudgeAgent(_StubJudgeAgent):
    def __init__(self):
        super().__init__()
        self.prompts: list[str] = []

    def invoke(self, payload, config=None):
        self.prompts.append(payload["messages"][0].content)
        return super().invoke(payload, config)


def test_judge_prompt_states_the_evidence_rules():
    prompt = qg._JUDGE_PROMPT.lower()
    assert "evidence rung" in prompt
    assert "no validated edge exists" in prompt
    assert "model-vs-market disagreements, not validated edges" in prompt
    assert "never invent a number" in prompt
    assert "research_status" in prompt and "paper_status" in prompt


def test_judge_gets_the_status_tools_and_each_candidate_its_evidence_status(monkeypatch):
    agent = _RecordingJudgeAgent()
    seen_tools: list[str] = []

    def _build(llm, tools):
        seen_tools.extend(t.name for t in tools)
        return agent

    monkeypatch.setattr(qg, "_build_judge_agent", _build)
    monkeypatch.setattr(qg, "_structure_answer", lambda llm, transcript, n_fixtures: Answer(overview="ok", notes=[]))
    graph = build_qa_graph(
        _features_frame(),
        _slice_evidence(),
        _FakeIndex(),
        _models(),
        llm=_StubLLM(AskIntent(intent="top_bets", n=2)),
        evidence_loader=_evidence,
    )
    state = ask_question(graph, "top 2 bets", pd.Timestamp("2024-05-25"))

    assert {"research_status", "paper_status"} <= set(seen_tools)
    assert state.evidence == _evidence()
    assert [c["evidence"]["status"] for c in state.candidates] == ["no registered evidence / below floor"] * 2
    assert all(c["evidence"]["model"] == "good_model" for c in state.candidates)
    (prompt,) = agent.prompts
    assert "display floor L4" in prompt
    assert "no strategy is at or above it" in prompt.lower()
    assert prompt.count("model-vs-market disagreement, not a validated edge") >= 2  # one per fixture line


def test_numbers_quoted_from_the_status_tools_count_as_grounded(monkeypatch):
    answer = Answer(
        overview="No validated edge exists.",
        notes=[
            FixtureNote(match_id="m2", take="The best family is only at L2 (estimate 0.0208); not a validated edge.")
        ],
    )
    monkeypatch.setattr(qg, "_build_judge_agent", lambda llm, tools: _StubJudgeAgent())
    monkeypatch.setattr(qg, "_structure_answer", lambda llm, transcript, n_fixtures: answer)
    graph = build_qa_graph(
        _features_frame(),
        _slice_evidence(),
        _FakeIndex(),
        _models(),
        llm=_StubLLM(AskIntent(intent="top_bets", n=1)),
        evidence_loader=_evidence,
    )
    state = ask_question(graph, "top 1 bet", pd.Timestamp("2024-05-25"))
    assert state.answer is not None
    assert state.answer.citations_grounded, state.answer.ungrounded_numbers


def test_template_fallback_says_no_validated_edge_exists(monkeypatch):
    def _raise(llm, tools):
        raise RuntimeError("no llm")

    monkeypatch.setattr(qg, "_build_judge_agent", _raise)
    graph = build_qa_graph(
        _features_frame(),
        _slice_evidence(),
        _FakeIndex(),
        _models(),
        llm=_StubLLM(AskIntent(intent="top_bets", n=2)),
        evidence_loader=_evidence,
    )
    state = ask_question(graph, "top bets", pd.Timestamp("2024-05-25"))
    assert state.answer is not None and state.answer.backend == "template"
    assert "no validated edge exists" in state.answer.overview.lower()
    assert all("not a validated edge" in n.take for n in state.answer.notes)


def test_a_broken_evidence_loader_falls_back_to_no_validated_edge(monkeypatch):
    def _broken():
        raise OSError("ledger unreadable")

    monkeypatch.setattr(qg, "_build_judge_agent", lambda llm, tools: (_ for _ in ()).throw(RuntimeError("no llm")))
    graph = build_qa_graph(
        _features_frame(),
        _slice_evidence(),
        _FakeIndex(),
        _models(),
        llm=_StubLLM(AskIntent(intent="top_bets", n=1)),
        evidence_loader=_broken,
    )
    state = ask_question(graph, "top 1 bet", pd.Timestamp("2024-05-25"))
    assert state.evidence["research"]["display_floor"] == "L4"
    assert state.evidence["research"]["validated_edge_exists"] is False
    assert state.candidates[0]["evidence"]["validated_edge"] is False


# --------------------------------------------------------------------------- data freshness line

_LAST_DATES = {"E0": pd.Timestamp("2024-04-20")}


def test_top_bets_answer_carries_the_stale_data_line(monkeypatch):
    from pitch_edge.agents.qa_data import data_freshness_notes

    monkeypatch.setattr(
        qg, "freshness_for", lambda wh, leagues, as_of: data_freshness_notes(_LAST_DATES, leagues, as_of)
    )
    monkeypatch.setattr(qg, "_build_judge_agent", lambda llm, tools: (_ for _ in ()).throw(RuntimeError("no llm")))
    llm = _StubLLM(AskIntent(intent="top_bets", n=2))
    graph = build_qa_graph(_features_frame(), _slice_evidence(), _FakeIndex(), _models(), llm=llm)
    state = ask_question(graph, "top bets", pd.Timestamp("2024-05-25"))

    assert state.answer is not None
    assert state.data_notes == ["Data for E0 ends 2024-04-20 (35 days before 2024-05-25)"]


def test_top_bets_answer_omits_the_line_when_data_is_fresh(monkeypatch):
    from pitch_edge.agents.qa_data import data_freshness_notes

    fresh = {"E0": pd.Timestamp("2024-05-22")}
    monkeypatch.setattr(qg, "freshness_for", lambda wh, leagues, as_of: data_freshness_notes(fresh, leagues, as_of))
    monkeypatch.setattr(qg, "_build_judge_agent", lambda llm, tools: (_ for _ in ()).throw(RuntimeError("no llm")))
    llm = _StubLLM(AskIntent(intent="top_bets", n=2))
    graph = build_qa_graph(_features_frame(), _slice_evidence(), _FakeIndex(), _models(), llm=llm)
    state = ask_question(graph, "top bets", pd.Timestamp("2024-05-25"))

    assert state.answer is not None
    assert state.data_notes == []


def test_fixtures_on_day_reads_the_warehouse_for_the_stale_data_line(warehouse):
    warehouse.upsert(
        "matches",
        pd.DataFrame(
            {
                "match_id": ["old"],
                "date": pd.to_datetime(["2024-04-20 15:00"]),
                "league_code": ["E0"],
                "home_team": ["H"],
                "away_team": ["A"],
            }
        ),
        keys=["match_id"],
    )
    llm = _StubLLM(AskIntent(intent="fixtures_on_day", day="2024-06-01"))
    graph = build_qa_graph(_features_frame(), _slice_evidence(), _FakeIndex(), _models(), llm=llm, wh=warehouse)
    stale = ask_question(graph, "what's on 2024-06-01", pd.Timestamp("2024-05-25"))
    assert stale.data_notes == ["Data for E0 ends 2024-04-20 (35 days before 2024-05-25)"]
    fresh = ask_question(graph, "what's on 2024-06-01", pd.Timestamp("2024-04-25"))
    assert fresh.data_notes == []


def test_a_day_without_fixtures_still_carries_the_stale_data_line(warehouse):
    warehouse.upsert(
        "matches",
        pd.DataFrame(
            {
                "match_id": ["old"],
                "date": pd.to_datetime(["2024-04-20 15:00"]),
                "league_code": ["E0"],
                "home_team": ["H"],
                "away_team": ["A"],
            }
        ),
        keys=["match_id"],
    )
    llm = _StubLLM(AskIntent(intent="fixtures_on_day", day="2031-01-01"))
    graph = build_qa_graph(_features_frame(), _slice_evidence(), _FakeIndex(), _models(), llm=llm, wh=warehouse)
    state = ask_question(graph, "what's on 2031-01-01", pd.Timestamp("2024-05-25"))
    assert state.message == "No fixtures found for 2031-01-01."
    assert state.data_notes == ["Data for E0 ends 2024-04-20 (35 days before 2024-05-25)"]
