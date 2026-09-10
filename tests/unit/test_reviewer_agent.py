"""Reviewer agent: deterministic trust/distrust from real backtest evidence + real-vs-synthetic
odds, never an LLM call — see agents/reviewer.py for why the judgment doesn't need one. Never
drops a proposal, even on distrust."""

from __future__ import annotations

import pandas as pd
import pytest

from pitch_edge.agents.reviewer import review_proposals

pytestmark = pytest.mark.unit


def _proposal(bookmaker="PS", model_name="gbdt"):
    return {
        "match_id": "m1",
        "date": "2025-01-01",
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
        "rationale": "edge",
    }


def _state(**proposal_kwargs):
    return {
        "proposals": [_proposal(**proposal_kwargs)],
        "fixtures": [{"match_id": "m1", "league_code": "E0"}],
        "log": [],
    }


def _write_by_league(tmp_path, edge_bits=0.03):
    pd.DataFrame(
        [{"model": "gbdt", "league_code": "E0", "n": 400, "log_loss": 0.6, "market_log_loss": 0.63, "edge_bits": edge_bits}]
    ).to_csv(tmp_path / "by_league.csv", index=False)


def test_positive_edge_bits_and_real_odds_trust(tmp_path):
    _write_by_league(tmp_path, edge_bits=0.03)
    out = review_proposals(_state(bookmaker="PS"), reports_dir=tmp_path)
    assert len(out["proposals"]) == 1
    assert out["proposals"][0]["verdict"] == "trust"
    assert "positive real edge_bits" in out["reviews"][0]["reasons"][0]


def test_negative_edge_bits_distrust_but_never_dropped(tmp_path):
    _write_by_league(tmp_path, edge_bits=-0.02)
    out = review_proposals(_state(bookmaker="PS"), reports_dir=tmp_path)
    assert len(out["proposals"]) == 1  # never silently dropped
    assert out["proposals"][0]["verdict"] == "distrust"


def test_synthetic_odds_distrust_even_with_positive_edge_bits(tmp_path):
    _write_by_league(tmp_path, edge_bits=0.03)
    out = review_proposals(_state(bookmaker="synthetic_elo_book"), reports_dir=tmp_path)
    assert out["proposals"][0]["verdict"] == "distrust"
    assert any("synthetic" in r for r in out["reviews"][0]["reasons"])


def test_no_backtest_evidence_needs_info(tmp_path):
    # by_league.csv written for a different model name -> no evidence for "gbdt".
    pd.DataFrame(
        [{"model": "dixon_coles", "league_code": "E0", "n": 400, "log_loss": 0.6, "market_log_loss": 0.63, "edge_bits": 0.03}]
    ).to_csv(tmp_path / "by_league.csv", index=False)
    out = review_proposals(_state(model_name="gbdt"), reports_dir=tmp_path)
    assert out["proposals"][0]["verdict"] == "needs_info"


def test_review_proposals_no_proposals(tmp_path):
    out = review_proposals({"proposals": [], "log": []}, reports_dir=tmp_path)
    assert out["proposals"] == [] and out["reviews"] == []
