"""Predictions explainer: the single, on-demand LLM call over an already-computed batch of
proposals. Never runs automatically, never per-fixture — see agents/explainer.py."""

from __future__ import annotations

import pytest

from pitch_edge.agents.explainer import SignalsExplanation, explain_signals

pytestmark = pytest.mark.unit


def _proposal(match_id="m1"):
    return {
        "match_id": match_id,
        "home_team": "A",
        "away_team": "B",
        "outcome": "home",
        "model_probability": 0.55,
        "market_probability": 0.45,
        "edge": 0.10,
        "decimal_odds": 2.0,
        "stake": 10.0,
        "model_name": "gbdt",
    }


def _review(match_id="m1", verdict="trust"):
    return {"match_id": match_id, "outcome": "home", "verdict": verdict, "reasons": ["positive real edge_bits"]}


class _StubStructured:
    def __init__(self, response):
        self._response = response
        self.calls = 0

    def invoke(self, prompt):
        self.calls += 1
        return self._response


class _StubLLM:
    def __init__(self, response):
        self._structured = _StubStructured(response)

    def with_structured_output(self, model):
        return self._structured


def test_explain_signals_no_proposals_short_circuits(monkeypatch):
    called = []
    monkeypatch.setattr(
        "pitch_edge.agents.explainer.get_local_llm", lambda model=None: called.append(1) or _StubLLM(None)
    )
    out = explain_signals([], [], llm_model=None)
    assert out.overview == "No proposals to explain."
    assert called == []  # never even constructs an LLM client for an empty batch


def test_explain_signals_makes_exactly_one_call_for_many_proposals(monkeypatch):
    proposals = [_proposal(f"m{i}") for i in range(5)]
    reviews = [_review(f"m{i}") for i in range(5)]
    response = SignalsExplanation(overview="5 proposals, all trusted.", notes=[], backend="ignored")
    stub = _StubLLM(response)
    monkeypatch.setattr("pitch_edge.agents.explainer.get_local_llm", lambda model=None: stub)
    out = explain_signals(proposals, reviews, llm_model="llama3.1:8b")
    assert stub._structured.calls == 1  # one call for the whole batch, not one per proposal
    assert out.overview == "5 proposals, all trusted."
    assert out.backend == "ollama:llama3.1:8b"


def test_explain_signals_falls_back_to_template_when_llm_unavailable(monkeypatch):
    monkeypatch.setattr(
        "pitch_edge.agents.explainer.get_local_llm",
        lambda model=None: (_ for _ in ()).throw(RuntimeError("ollama not running")),
    )
    out = explain_signals([_proposal()], [_review()], llm_model=None)
    assert out.backend == "template"
    assert "1 proposal" in out.overview
    assert len(out.notes) == 1
