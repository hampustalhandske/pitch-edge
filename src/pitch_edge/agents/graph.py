"""LangGraph signal pipeline with a mandatory human-approval gate.

    scout -> features -> inference -> odds -> edge_detector -> risk_manager
          -> [HUMAN APPROVAL GATE] -> log_alerts

The gate is a `interrupt_before=["human_approval"]` on a checkpointed graph:
execution *stops* there and can only continue when a human writes a decision
into state and resumes. There is no flag, config or test mode that removes
the interrupt — `build_graph` always compiles with it, and `log_alerts`
refuses to write anything not carrying an explicit human decision. Every
node reads/writes state only, so each is unit-testable in isolation and every
run is replayable from the checkpoint store.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any, TypedDict

import pandas as pd
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from pitch_edge.agents.risk import Proposal, RiskManager
from pitch_edge.odds.utils import no_vig_probabilities


class SignalState(TypedDict, total=False):
    run_id: str
    as_of: str
    fixtures: list[dict]  # upcoming fixtures with features
    predictions: list[dict]  # per fixture: match_id, p_home, p_draw, p_away, model
    odds: list[dict]  # per fixture: match_id, bookmaker, home, draw, away
    edges: list[dict]
    proposals: list[dict]
    decisions: dict[str, str]  # match_id|outcome -> "approved" | "rejected" (written by a human)
    approved_by: str
    alerts: list[dict]
    log: list[str]
    errors: list[str]
    # Additive keys for the agentic orchestrator (agents/orchestrator.py) — the deterministic graph
    # in this module never reads or writes them.
    selected_fixtures: list[dict]  # per-fixture FixtureReadiness verdicts from agents/selection.py
    context_docs: dict[str, list[dict]]  # match_id -> [{doc_id, text, metadata}] gathered for it
    reviews: list[dict]  # per-proposal EdgeReview verdicts from agents/reviewer.py


class GraphDependencies:
    """Injected callables so nodes are pure with respect to I/O."""

    def __init__(
        self,
        scout: Callable[[], list[dict]],
        featurize: Callable[[list[dict]], list[dict]],
        infer: Callable[[list[dict]], list[dict]],
        fetch_odds: Callable[[list[dict]], list[dict]],
        risk: RiskManager,
        sink: Callable[[list[dict]], None],
        model_name: str = "ensemble",
    ):
        self.scout, self.featurize, self.infer, self.fetch_odds = scout, featurize, infer, fetch_odds
        self.risk, self.sink, self.model_name = risk, sink, model_name


def _log(state: SignalState, msg: str) -> list[str]:
    return [*state.get("log", []), f"{datetime.now(UTC).isoformat(timespec='seconds')} {msg}"]


def make_nodes(deps: GraphDependencies) -> dict[str, Callable[[SignalState], dict[str, Any]]]:
    def scout(state: SignalState) -> dict:
        fixtures = deps.scout()
        return {"fixtures": fixtures, "log": _log(state, f"scout: {len(fixtures)} fixtures")}

    def features(state: SignalState) -> dict:
        fx = deps.featurize(state.get("fixtures", []))
        return {"fixtures": fx, "log": _log(state, f"features: {len(fx)} rows")}

    def inference(state: SignalState) -> dict:
        preds = deps.infer(state.get("fixtures", []))
        return {"predictions": preds, "log": _log(state, f"inference: {len(preds)} predictions")}

    def odds(state: SignalState) -> dict:
        quotes = deps.fetch_odds(state.get("fixtures", []))
        return {"odds": quotes, "log": _log(state, f"odds: {len(quotes)} quotes")}

    def edge_detector(state: SignalState) -> dict:
        preds = {p["match_id"]: p for p in state.get("predictions", [])}
        fixtures = {f["match_id"]: f for f in state.get("fixtures", [])}
        edges = []
        for q in state.get("odds", []):
            p = preds.get(q["match_id"])
            f = fixtures.get(q["match_id"], {})
            if not p or min(q["home"], q["draw"], q["away"]) <= 1.0:
                continue
            mkt = dict(
                zip(("home", "draw", "away"), no_vig_probabilities(q["home"], q["draw"], q["away"]), strict=True)
            )
            for o in ("home", "draw", "away"):
                edges.append(
                    {
                        "match_id": q["match_id"],
                        "date": f.get("date", state.get("as_of")),
                        "home_team": f.get("home_team", ""),
                        "away_team": f.get("away_team", ""),
                        "outcome": o,
                        "model_probability": float(p[f"p_{o}"]),
                        "market_probability": mkt[o],
                        "edge": float(p[f"p_{o}"]) - mkt[o],
                        "decimal_odds": float(q[o]),
                        "bookmaker": q.get("bookmaker", ""),
                        # The router-based agentic `inference` node stamps each prediction with the
                        # actual per-fixture model it used (`p["model"]`); the deterministic pipeline's
                        # single-model `inference` node doesn't, so this falls back to deps.model_name
                        # — same value it always carried, no behaviour change there.
                        "model_name": p.get("model", deps.model_name),
                    }
                )
        return {
            "edges": edges,
            "log": _log(state, f"edge_detector: {sum(e['edge'] > 0 for e in edges)} positive edges"),
        }

    def risk_manager(state: SignalState) -> dict:
        edges = pd.DataFrame(state.get("edges", []))
        proposals: list[Proposal] = deps.risk.size(edges, deps.model_name) if not edges.empty else []
        return {
            "proposals": [asdict(p) for p in proposals],
            "log": _log(state, f"risk_manager: {len(proposals)} proposals (halted={deps.risk.halted()})"),
        }

    def human_approval(state: SignalState) -> dict:
        # Reached only after a human resumed the interrupted graph. Missing decisions = rejected.
        decisions = state.get("decisions") or {}
        approved = [
            p for p in state.get("proposals", []) if decisions.get(f"{p['match_id']}|{p['outcome']}") == "approved"
        ]
        return {
            "proposals": approved,
            "log": _log(state, f"human_approval: {len(approved)} approved by {state.get('approved_by', 'unknown')}"),
        }

    def log_alerts(state: SignalState) -> dict:
        if not state.get("approved_by"):
            raise PermissionError("log_alerts reached without a human approver in state — refusing to emit alerts")
        now = datetime.now(UTC).replace(tzinfo=None).isoformat(timespec="seconds")
        alerts = [
            {
                **p,
                "trade_id": uuid.uuid4().hex,
                "status": "approved_paper",
                "approved_by": state["approved_by"],
                "approved_at": now,
                "run_id": state.get("run_id", ""),
            }
            for p in state.get("proposals", [])
        ]
        deps.sink(alerts)
        return {"alerts": alerts, "log": _log(state, f"log_alerts: {len(alerts)} paper trades logged")}

    return {
        "scout": scout,
        "features": features,
        "inference": inference,
        "odds": odds,
        "edge_detector": edge_detector,
        "risk_manager": risk_manager,
        "human_approval": human_approval,
        "log_alerts": log_alerts,
    }


def build_graph(deps: GraphDependencies, checkpointer=None):
    nodes = make_nodes(deps)
    g: StateGraph = StateGraph(SignalState)
    for name, fn in nodes.items():
        g.add_node(name, fn)  # type: ignore[call-overload]
    g.set_entry_point("scout")
    for a, b in [
        ("scout", "features"),
        ("features", "inference"),
        ("inference", "odds"),
        ("odds", "edge_detector"),
        ("edge_detector", "risk_manager"),
        ("risk_manager", "human_approval"),
        ("human_approval", "log_alerts"),
    ]:
        g.add_edge(a, b)
    g.add_edge("log_alerts", END)
    # The gate is structural: compile() always interrupts before human_approval.
    return g.compile(checkpointer=checkpointer or MemorySaver(), interrupt_before=["human_approval"])


class SignalPipeline:
    """Convenience wrapper: run to the gate, inspect proposals, resume with human decisions."""

    def __init__(self, deps: GraphDependencies, checkpointer=None):
        self.graph = build_graph(deps, checkpointer)

    def run_to_gate(self, thread_id: str | None = None) -> tuple[str, SignalState]:
        thread_id = thread_id or uuid.uuid4().hex
        cfg = {"configurable": {"thread_id": thread_id}}
        init: SignalState = {
            "run_id": thread_id,
            "as_of": datetime.now(UTC).replace(tzinfo=None).isoformat(timespec="seconds"),
            "log": [],
        }
        self.graph.invoke(init, cfg)
        return thread_id, self.graph.get_state(cfg).values

    def pending_proposals(self, thread_id: str) -> list[dict]:
        return self.graph.get_state({"configurable": {"thread_id": thread_id}}).values.get("proposals", [])

    def resume_with_decisions(self, thread_id: str, decisions: dict[str, str], approved_by: str) -> SignalState:
        if not approved_by:
            raise ValueError("approved_by (a human identifier) is required to resume past the approval gate")
        cfg = {"configurable": {"thread_id": thread_id}}
        self.graph.update_state(cfg, {"decisions": decisions, "approved_by": approved_by})
        self.graph.invoke(None, cfg)
        return self.graph.get_state(cfg).values
