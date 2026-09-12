"""Stage 2 read-only tools: typed args, argument validation before any lookup, bounded search."""

from __future__ import annotations

import pandas as pd
import pytest

from pitch_edge.agents.evidence_tools import make_evidence_tools
from pitch_edge.rag.documents import Document

pytestmark = pytest.mark.unit


class _FakeIndex:
    def query(self, text, k=6, where=None):
        return [(Document("news:1", "some context " * 50), 1.0)]


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"match_id": "m1", "model": "gbdt", "p_home": 0.5, "p_draw": 0.3, "p_away": 0.2},
            {"match_id": "m1", "model": "dixon_coles", "p_home": 0.45, "p_draw": 0.3, "p_away": 0.25},
        ]
    )


def _slice_evidence() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": "gbdt",
                "slice_dim": "league_code",
                "slice_value": "E0",
                "n": 200,
                "edge_bits": 0.02,
                "q_value": 0.01,
                "significant": True,
                "checkpoint_date": "2023-01-01",
            }
        ]
    )


def _tools(as_of=None):
    as_of = as_of if as_of is not None else pd.Timestamp("2024-01-01")
    tools = make_evidence_tools(_predictions(), _slice_evidence(), _FakeIndex(), as_of, ["gbdt", "dixon_coles"])
    return {t.name: t for t in tools}


def test_get_model_predictions_returns_every_model_for_the_fixture():
    tools = _tools()
    out = tools["get_model_predictions"].invoke({"match_id": "m1"})
    assert "model=gbdt" in out and "model=dixon_coles" in out


def test_get_model_predictions_unknown_fixture():
    tools = _tools()
    out = tools["get_model_predictions"].invoke({"match_id": "unknown"})
    assert "no predictions found" in out


def test_get_backtest_evidence_returns_significance():
    tools = _tools()
    out = tools["get_backtest_evidence"].invoke({"model_name": "gbdt", "slice_dim": "league_code", "slice_value": "E0"})
    assert "edge_bits=+0.0200" in out and "significant" in out


def test_get_backtest_evidence_rejects_unknown_model_before_any_lookup():
    tools = _tools()
    out = tools["get_backtest_evidence"].invoke(
        {"model_name": "not_a_real_model", "slice_dim": "league_code", "slice_value": "E0"}
    )
    assert "unknown model" in out


def test_get_backtest_evidence_rejects_unknown_slice_dim():
    tools = _tools()
    out = tools["get_backtest_evidence"].invoke({"model_name": "gbdt", "slice_dim": "not_a_dim", "slice_value": "E0"})
    assert "unknown slice_dim" in out


def test_get_backtest_evidence_respects_as_of_checkpoint_cutoff():
    tools = _tools(as_of=pd.Timestamp("2022-01-01"))  # before the only checkpoint we have
    out = tools["get_backtest_evidence"].invoke({"model_name": "gbdt", "slice_dim": "league_code", "slice_value": "E0"})
    assert "no backtest evidence available" in out


def test_search_context_truncates_returned_text():
    tools = _tools()
    out = tools["search_context"].invoke({"query": "team news"})
    assert out.startswith("[news:1]")
    assert len(out) < 500
