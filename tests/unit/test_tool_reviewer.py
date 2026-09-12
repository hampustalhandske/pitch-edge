"""Tool-calling reviewer: the verdict comes from an LLM agent calling its own tools
(`create_react_agent`) instead of being handed pre-fetched evidence.
"""

from __future__ import annotations

import pandas as pd
import pytest

from pitch_edge.agents.tool_reviewer import ToolReviewVerdict, review_proposals_agentic

pytestmark = pytest.mark.unit


def _proposal(bookmaker="PS", model_name="gbdt"):
    return {
        "match_id": "m1",
        "home_team": "A",
        "away_team": "B",
        "outcome": "home",
        "model_probability": 0.55,
        "market_probability": 0.45,
        "edge": 0.10,
        "decimal_odds": 2.0,
        "stake": 10.0,
        "bookmaker": bookmaker,
        "model_name": model_name,
    }


def _state(**proposal_kwargs):
    return {
        "proposals": [_proposal(**proposal_kwargs)],
        "fixtures": [{"match_id": "m1", "league_code": "E0"}],
        "log": [],
    }


def _write_by_league(tmp_path, edge_bits=0.03):
    pd.DataFrame(
        [
            {
                "model": "gbdt",
                "league_code": "E0",
                "n": 400,
                "log_loss": 0.6,
                "market_log_loss": 0.63,
                "edge_bits": edge_bits,
            }
        ]
    ).to_csv(tmp_path / "by_league.csv", index=False)


class _StubAgent:
    def __init__(self, response):
        self._response = response
        self.calls = 0

    def invoke(self, payload):
        self.calls += 1
        if isinstance(self._response, Exception):
            raise self._response
        return {"structured_response": self._response}


def test_review_proposals_agentic_no_proposals_never_builds_agent(monkeypatch, tmp_path):
    built = []
    monkeypatch.setattr(
        "pitch_edge.agents.tool_reviewer._build_agent",
        lambda *a, **k: built.append(1) or _StubAgent(None),
    )
    out = review_proposals_agentic({"proposals": [], "log": []}, reports_dir=tmp_path, index=object())
    assert out["proposals"] == [] and out["reviews"] == []
    assert built == []


def test_review_proposals_agentic_uses_tool_agent_verdict(monkeypatch, tmp_path):
    _write_by_league(tmp_path, edge_bits=0.03)
    stub = _StubAgent(ToolReviewVerdict(verdict="trust", reasons=["backtest_evidence tool: edge_bits=+0.0300"]))
    monkeypatch.setattr("pitch_edge.agents.tool_reviewer._build_agent", lambda *a, **k: stub)
    out = review_proposals_agentic(_state(bookmaker="PS"), reports_dir=tmp_path, index=object())
    assert stub.calls == 1
    assert out["proposals"][0]["verdict"] == "trust"
    assert out["reviews"][0]["verdict"] == "trust"
    assert "review_proposals_agentic: 1/1 trusted" in out["log"][-1]


def test_review_proposals_agentic_falls_back_to_deterministic_judge_on_llm_failure(monkeypatch, tmp_path):
    _write_by_league(tmp_path, edge_bits=0.03)
    stub = _StubAgent(RuntimeError("ollama not running"))
    monkeypatch.setattr("pitch_edge.agents.tool_reviewer._build_agent", lambda *a, **k: stub)
    out = review_proposals_agentic(_state(bookmaker="PS"), reports_dir=tmp_path, index=object())
    # deterministic _judge: positive edge_bits + real odds -> trust, even though the LLM "failed"
    assert out["proposals"][0]["verdict"] == "trust"
    assert "fell back to deterministic judge" in out["log"][-1]


def test_review_proposals_agentic_never_drops_a_distrusted_proposal(monkeypatch, tmp_path):
    _write_by_league(tmp_path, edge_bits=0.03)
    stub = _StubAgent(ToolReviewVerdict(verdict="distrust", reasons=["synthetic odds"]))
    monkeypatch.setattr("pitch_edge.agents.tool_reviewer._build_agent", lambda *a, **k: stub)
    out = review_proposals_agentic(_state(bookmaker="synthetic_elo_book"), reports_dir=tmp_path, index=object())
    assert len(out["proposals"]) == 1
    assert out["proposals"][0]["verdict"] == "distrust"
