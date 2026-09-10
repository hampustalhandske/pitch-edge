"""The agentic orchestrator: additive graph, same human-approval gate, agents/graph.py untouched.

Fully deterministic — no LLM anywhere in the graph (see agents/selection.py and
agents/reviewer.py), so these tests exercise the real code paths with no stubbing needed."""

from __future__ import annotations

import pandas as pd
import pytest

from pitch_edge.agents.graph import GraphDependencies
from pitch_edge.agents.orchestrator import AgenticSignalPipeline, build_agentic_graph
from pitch_edge.agents.risk import RiskLimits, RiskManager

pytestmark = pytest.mark.integration

FIXTURES = [
    {
        "match_id": "m1",
        "date": "2030-01-01",
        "home_team": "A",
        "away_team": "B",
        "league_code": "E0",
        "lineup_source": "provisional",
    },
    {
        "match_id": "m2",
        "date": "2030-01-01",
        "home_team": "C",
        "away_team": "D",
        "league_code": "E0",
        "lineup_source": "provisional",
    },
]


class _StubIndex:
    def query(self, text, k=6, where=None):
        return []


def _deps(sink_store: list) -> GraphDependencies:
    return GraphDependencies(
        scout=lambda: FIXTURES,
        featurize=lambda rows: rows,
        infer=lambda rows: [{"match_id": r["match_id"], "p_home": 0.62, "p_draw": 0.2, "p_away": 0.18} for r in rows],
        fetch_odds=lambda rows: [
            {"match_id": r["match_id"], "bookmaker": "PS", "home": 2.1, "draw": 3.4, "away": 3.6} for r in rows
        ],
        risk=RiskManager(RiskLimits(min_edge=0.02)),
        sink=lambda alerts: sink_store.extend(alerts),
        model_name="gbdt",
    )


def test_agentic_graph_interrupts_before_human_approval(tmp_path):
    pd.DataFrame(
        [{"model": "gbdt", "league_code": "E0", "n": 400, "log_loss": 0.6, "market_log_loss": 0.63, "edge_bits": 0.03}]
    ).to_csv(tmp_path / "by_league.csv", index=False)
    store: list = []
    pipe = AgenticSignalPipeline(_deps(store), reports_dir=tmp_path, index=_StubIndex(), wh=None)
    tid, state = pipe.run_to_gate()
    assert state["proposals"]
    assert "alerts" not in state and store == []
    assert len(state["selected_fixtures"]) == 2
    assert all(f["ready"] for f in state["selected_fixtures"])  # real edge_bits evidence for E0
    assert len(state["reviews"]) == len(state["proposals"])
    assert state["reviews"][0]["verdict"] == "trust"  # positive edge_bits + real odds
    snapshot = pipe.graph.get_state({"configurable": {"thread_id": tid}})
    assert snapshot.next == ("human_approval",)


def test_agentic_graph_resume_still_requires_human_identifier(tmp_path):
    pd.DataFrame(
        [{"model": "gbdt", "league_code": "E0", "n": 400, "log_loss": 0.6, "market_log_loss": 0.63, "edge_bits": 0.03}]
    ).to_csv(tmp_path / "by_league.csv", index=False)
    store: list = []
    pipe = AgenticSignalPipeline(_deps(store), reports_dir=tmp_path, index=_StubIndex(), wh=None)
    tid, _ = pipe.run_to_gate()
    with pytest.raises(ValueError):
        pipe.resume_with_decisions(tid, {}, approved_by="")
    final = pipe.resume_with_decisions(tid, {"m1|home": "approved", "m2|home": "rejected"}, approved_by="human")
    assert [a["match_id"] for a in final["alerts"]] == ["m1"]
    assert len(store) == 1 and store[0]["status"] == "approved_paper"


def test_agentic_graph_screens_out_leagues_with_no_backtest_evidence(tmp_path):
    # No by_league.csv written -> select_model_for_league finds no evidence for E0 -> not ready.
    store: list = []
    pipe = AgenticSignalPipeline(_deps(store), reports_dir=tmp_path, index=_StubIndex(), wh=None)
    _, state = pipe.run_to_gate()
    assert state["proposals"] == []
    assert all(not f["ready"] for f in state["selected_fixtures"])


def test_build_agentic_graph_always_has_the_interrupt(tmp_path):
    g = build_agentic_graph(_deps([]), reports_dir=tmp_path, index=_StubIndex(), wh=None)
    cfg = {"configurable": {"thread_id": "x"}}
    g.invoke({"run_id": "x", "log": []}, cfg)
    assert g.get_state(cfg).next == ("human_approval",)
