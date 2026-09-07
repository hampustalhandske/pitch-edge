"""The LangGraph pipeline's human-approval gate is structural, not a flag."""

from __future__ import annotations

import pytest

from pitch_edge.agents.graph import GraphDependencies, SignalPipeline, build_graph, make_nodes
from pitch_edge.agents.risk import RiskLimits, RiskManager

pytestmark = pytest.mark.integration

FIXTURES = [
    {"match_id": "m1", "date": "2030-01-01", "home_team": "A", "away_team": "B"},
    {"match_id": "m2", "date": "2030-01-01", "home_team": "C", "away_team": "D"},
]


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
        model_name="test",
    )


def test_graph_interrupts_before_human_approval():
    store: list = []
    pipe = SignalPipeline(_deps(store))
    tid, state = pipe.run_to_gate()
    assert state["proposals"] and all(p["outcome"] == "home" for p in state["proposals"])
    assert "alerts" not in state and store == []
    snapshot = pipe.graph.get_state({"configurable": {"thread_id": tid}})
    assert snapshot.next == ("human_approval",)


def test_resume_requires_human_identifier_and_filters_rejections():
    store: list = []
    pipe = SignalPipeline(_deps(store))
    tid, _ = pipe.run_to_gate()
    with pytest.raises(ValueError):
        pipe.resume_with_decisions(tid, {}, approved_by="")
    final = pipe.resume_with_decisions(tid, {"m1|home": "approved", "m2|home": "rejected"}, approved_by="human")
    assert [a["match_id"] for a in final["alerts"]] == ["m1"]
    assert len(store) == 1 and store[0]["status"] == "approved_paper"


def test_log_alerts_node_refuses_without_approver():
    nodes = make_nodes(_deps([]))
    with pytest.raises(PermissionError):
        nodes["log_alerts"]({"proposals": [{"match_id": "m1", "outcome": "home"}]})


def test_nodes_are_individually_testable():
    nodes = make_nodes(_deps([]))
    state = {"log": []}
    state.update(nodes["scout"](state))
    state.update(nodes["features"](state))
    state.update(nodes["inference"](state))
    state.update(nodes["odds"](state))
    state.update(nodes["edge_detector"](state))
    assert len(state["edges"]) == 6
    state.update(nodes["risk_manager"](state))
    assert len(state["proposals"]) == 2
    assert len(state["log"]) == 6


def test_build_graph_always_has_the_interrupt():
    g = build_graph(_deps([]))
    cfg = {"configurable": {"thread_id": "x"}}
    g.invoke({"run_id": "x", "log": []}, cfg)
    assert g.get_state(cfg).next == ("human_approval",)
