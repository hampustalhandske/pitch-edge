"""The genuinely agentic signals graph — additive, does not touch `agents/graph.py`.

    select_fixtures -> gather_context -> inference -> odds -> edge_detector -> risk_manager
        -> review_proposals -> [interrupt] human_approval -> log_alerts

`inference`, `odds`, `edge_detector`, `risk_manager`, `human_approval` and `log_alerts` are the
exact node functions from `agents.graph.make_nodes(deps)`, unmodified — only `select_fixtures`,
`gather_context` and `review_proposals` are new. The human-approval gate
(`interrupt_before=["human_approval"]`) is identical to the deterministic graph's: it is not
negotiable and is not being changed by adding agent judgment upstream of it.

No LLM runs anywhere in this graph. `select_fixtures` and `review_proposals` are both deterministic
(see their own docstrings) — routing, edge detection, and risk sizing all run on the real resources
this project already has (the model router, the feature store, real/synthetic odds, backtest
evidence). `gather_context` still retrieves RAG documents per fixture, purely so that evidence is
available if a user later asks the system to explain the results — that single, on-demand LLM call
lives in `agents/explainer.py` and is never invoked as part of running this graph.

The `inference` node's model is chosen per-league by `select_model_for_league` rather than one
hardcoded model — `deps.infer`/`deps.model_name` must therefore be *rebuilt per league* inside this
graph, since the deterministic `GraphDependencies.infer` closure only knows a single model.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from pitch_edge.agents.graph import GraphDependencies, SignalState, make_nodes
from pitch_edge.agents.reviewer import review_proposals
from pitch_edge.agents.router import select_model_for_league
from pitch_edge.agents.selection import select_fixtures
from pitch_edge.data.storage import Warehouse
from pitch_edge.models.base import MatchModel
from pitch_edge.rag.fixture_context import gather_fixture_context
from pitch_edge.rag.index import VectorIndex

logger = logging.getLogger(__name__)


def _log(state: SignalState, msg: str) -> list[str]:
    return [*state.get("log", []), f"{datetime.now(UTC).isoformat(timespec='seconds')} {msg}"]


def _make_router_infer(available: list[MatchModel], reports_dir: str | Path):
    """Replaces `deps.infer`: picks a model per-league via `select_model_for_league` instead of
    running one hardcoded model over every fixture. Fixtures whose league has no trusted model are
    dropped here and logged, never silently defaulted to some other model. `available` must already
    be fit — this function only selects and predicts."""

    def infer(rows: list[dict]) -> list[dict]:
        if not rows:
            return []
        preds: list[dict] = []
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        for league_code, grp in df.groupby(df.get("league_code", pd.Series(dtype=str))):
            model = select_model_for_league(str(league_code), reports_dir, available)
            if model is None:
                logger.info(
                    "inference: skipping %d fixture(s) in league %s — no trusted model (see select_model_for_league log above)",
                    len(grp),
                    league_code,
                )
                continue
            probs = model.predict_proba(grp)
            for mid, h, d, a in zip(grp["match_id"], probs["home"], probs["draw"], probs["away"], strict=True):
                preds.append(
                    {"match_id": mid, "model": model.name, "p_home": float(h), "p_draw": float(d), "p_away": float(a)}
                )
        return preds

    return infer


def build_agentic_graph(
    deps: GraphDependencies,
    reports_dir: str | Path,
    index: VectorIndex,
    wh: Warehouse | None = None,
    available_models: list[MatchModel] | None = None,
    checkpointer=None,
    context_as_of: str | None = None,
):
    """`available_models` must already be fit (e.g. on `load_feature_frame(wh, ...)`) — this graph
    only selects among them per-league and calls `predict_proba`. Without them (or without `wh`),
    the router is skipped and `deps.infer`'s single model is used instead, matching the
    deterministic graph's behaviour.

    `context_as_of` (an ISO date string) is threaded into `gather_fixture_context` to exclude any
    retrieved doc dated on or after it — required by `backtest/replay.py` so a production RAG index
    that already contains real match-report docs for the replay window can't leak the real outcome
    into the explainer's context. Live `agentic-signals` runs leave it unset (nothing to hide)."""
    base_nodes = make_nodes(deps)

    def select_fixtures_node(state: SignalState) -> dict:
        # This graph's entry point folds in scouting (deterministic, unchanged) before screening
        # what was scouted — there is no separate "scout"/"features" node in this graph.
        fixtures = deps.featurize(deps.scout())
        scouted_state: SignalState = {**state, "fixtures": fixtures}
        return select_fixtures(scouted_state, reports_dir, wh=wh, available=available_models)

    def gather_context_node(state: SignalState) -> dict:
        fixtures = state.get("fixtures", [])
        context_docs: dict[str, list[dict]] = {}
        for fx in fixtures:
            docs = gather_fixture_context(
                index, fx.get("home_team", ""), fx.get("away_team", ""), fx["match_id"], as_of=context_as_of
            )
            context_docs[fx["match_id"]] = [{"doc_id": d.doc_id, "text": d.text, "metadata": d.metadata} for d in docs]
        return {
            "context_docs": context_docs,
            "log": _log(state, f"gather_context: {len(context_docs)} fixtures enriched"),
        }

    def review_proposals_node(state: SignalState) -> dict:
        return review_proposals(state, reports_dir)

    # `inference` is rebuilt to route per-league instead of the deterministic graph's single model.
    infer_fn = _make_router_infer(available_models, reports_dir) if available_models else deps.infer
    router_deps = GraphDependencies(
        deps.scout, deps.featurize, infer_fn, deps.fetch_odds, deps.risk, deps.sink, deps.model_name
    )
    router_nodes = make_nodes(router_deps)

    g: StateGraph = StateGraph(SignalState)
    g.add_node("select_fixtures", select_fixtures_node)  # type: ignore[arg-type]
    g.add_node("gather_context", gather_context_node)  # type: ignore[arg-type]
    g.add_node("inference", router_nodes["inference"])  # type: ignore[arg-type]
    g.add_node("odds", base_nodes["odds"])  # type: ignore[arg-type]
    g.add_node("edge_detector", base_nodes["edge_detector"])  # type: ignore[arg-type]
    g.add_node("risk_manager", base_nodes["risk_manager"])  # type: ignore[arg-type]
    g.add_node("review_proposals", review_proposals_node)  # type: ignore[arg-type]
    g.add_node("human_approval", base_nodes["human_approval"])  # type: ignore[arg-type]
    g.add_node("log_alerts", base_nodes["log_alerts"])  # type: ignore[arg-type]
    g.set_entry_point("select_fixtures")
    for a, b in [
        ("select_fixtures", "gather_context"),
        ("gather_context", "inference"),
        ("inference", "odds"),
        ("odds", "edge_detector"),
        ("edge_detector", "risk_manager"),
        ("risk_manager", "review_proposals"),
        ("review_proposals", "human_approval"),
        ("human_approval", "log_alerts"),
    ]:
        g.add_edge(a, b)
    g.add_edge("log_alerts", END)
    # Same gate as the deterministic graph — structural, never removed.
    return g.compile(checkpointer=checkpointer or MemorySaver(), interrupt_before=["human_approval"])


class AgenticSignalPipeline:
    """Mirrors `agents.graph.SignalPipeline`'s run-to-gate / resume convenience API."""

    def __init__(
        self,
        deps: GraphDependencies,
        reports_dir: str | Path,
        index: VectorIndex,
        wh: Warehouse | None = None,
        available_models: list[MatchModel] | None = None,
        checkpointer=None,
        context_as_of: str | None = None,
    ):
        self.graph = build_agentic_graph(
            deps,
            reports_dir,
            index,
            wh=wh,
            available_models=available_models,
            checkpointer=checkpointer,
            context_as_of=context_as_of,
        )

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
