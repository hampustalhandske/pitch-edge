"""The `ask` agent: a LangGraph `StateGraph` answering exactly three standardized questions —
"top N bets", "prediction on team A vs team B", and "what's on day X" — as of an explicit
point in time (see `backtest/as_of.py`).

    START -> parse_intent -+-> cannot_answer -> END              (question didn't structure)
                            +-> list_fixtures -> END              (fixtures_on_day: pure lookup)
                            +-> gather -> judge -> END             (top_bets / fixture)
                                            ^  |
                                            +--+ (one-shot reflection retry on ungrounded numbers)

`parse_intent` and `judge` are the only two LLM-touching nodes. `gather` is deterministic: it
resolves candidate fixtures, fits every model fresh on data strictly before `as_of` and predicts
them (`agents/qa_data.py`), computes each candidate's no-vig market edge, and picks which model to
trust per fixture from real backtest evidence matching that fixture's own circumstances
(`backtest/slices.py::select_trusted_model`) — never an LLM guess. `judge` is a bounded
tool-calling agent (`agents/evidence_tools.py`) that narrates and, when it has good reason from the
evidence, may prefer a different model's number for a fixture — it never computes a probability or
an edge itself, only ever picks among and explains numbers `gather` already produced.
"""

from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd
from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from pitch_edge.agents.evidence_tools import make_evidence_tools
from pitch_edge.agents.llm import get_llm_for
from pitch_edge.agents.qa_context import structure_context
from pitch_edge.agents.qa_data import market_edges, resolve_fixture, run_models, top_bets_candidates
from pitch_edge.backtest.slices import fixture_slice_values, select_trusted_model
from pitch_edge.models.base import MatchModel
from pitch_edge.rag.documents import Document
from pitch_edge.rag.generate import verify_citations
from pitch_edge.rag.index import VectorIndex
from pitch_edge.rag.query_parser import FixtureQuery, TopBetsQuery, parse_ask_intent

logger = logging.getLogger(__name__)

JUDGE_RECURSION_LIMIT = 25  # confirmed against real Groq calls: 8 was too tight for a 4-fixture
# batch (each fixture can cost 1-2 tool round-trips before a final answer), cutting the loop off
# mid-reasoning and leaving `_structure_answer` nothing real to restate
CANNOT_ANSWER_MESSAGE = "I can't understand that — try 'top N bets', '<team> vs <team>', or 'what's on <day>'."


class FixtureNote(BaseModel):
    match_id: str
    home_team: str = ""
    away_team: str = ""
    take: str = Field(description="One short grounded sentence on this fixture, citing only given numbers/doc_ids")
    citations: list[str] = Field(default_factory=list)


class Answer(BaseModel):
    overview: str = Field(description="2-3 sentences summarizing the answer as a whole")
    notes: list[FixtureNote] = Field(default_factory=list)
    citations_grounded: bool = True
    ungrounded_numbers: list[str] = Field(default_factory=list)
    backend: str = "template"


class QAState(BaseModel):
    question: str
    as_of: datetime
    intent: str = "unrecognized"
    n: int | None = None
    home_team: str | None = None
    away_team: str | None = None
    day: str | None = None
    candidates: list[dict] = Field(default_factory=list)
    predictions: list[dict] = Field(default_factory=list)
    edges: list[dict] = Field(default_factory=list)
    trusted: dict[str, str] = Field(default_factory=dict)
    context_docs: dict[str, list[dict]] = Field(default_factory=dict)
    message: str | None = None
    answer: Answer | None = None
    retry_count: int = 0


_JUDGE_PROMPT = (
    "Below is a fully-computed batch of football fixtures. Every number (each model's "
    "probability, the market's no-vig probability, the edge, which model the backtest evidence "
    "says to trust here) was already calculated deterministically — you are not estimating "
    "anything, only explaining what is given, in plain language.\n\n"
    "{summary}\n"
    "{doc_block}\n"
    "Fill in BOTH fields of the response schema:\n"
    "1. `overview`: 2-3 sentences summarizing the batch (how many fixtures, whether any model "
    "shows a real, significant edge here or whether none of them clearly beat the market).\n"
    "2. `notes`: exactly {n_fixtures} entries, one per fixture listed above (same match_id), each "
    "a short grounded sentence citing only the numbers/context given above — never invent a "
    "number or a doc_id — plus its `citations` list (doc_ids actually used, empty if none). If a "
    "fixture's backtest evidence says NOT significant, say the evidence is inconclusive rather "
    "than implying confidence. Use the get_model_predictions/get_backtest_evidence/search_context "
    "tools if you need to double check a number or look for more context before writing a note.\n"
    "Never suggest an action beyond describing what the numbers show — this is not a betting "
    "instruction."
)


def _best_edge_row(match_id: str, edges: list[dict], trusted_model: str | None) -> dict | None:
    """The trusted model's best-outcome edge row for this fixture, or (if no model is trusted here)
    the best edge row across all models. `None` if no prediction exists for this fixture at all."""
    rows = [e for e in edges if e["match_id"] == match_id]
    if trusted_model:
        rows = [r for r in rows if r["model"] == trusted_model] or rows
    return max(rows, key=lambda r: r["edge"]) if rows else None


def _fixture_summary_line(fixture: dict, edges: list[dict], trusted_model: str | None) -> str:
    best = _best_edge_row(fixture["match_id"], edges, trusted_model)
    if best is None:
        return f"- {fixture['match_id']} {fixture.get('home_team', '')} vs {fixture.get('away_team', '')}: no prediction available"
    trust_note = f"trusted model: {trusted_model}" if trusted_model else "no model with matching backtest evidence"
    return (
        f"- {fixture['match_id']} {fixture.get('home_team', '')} vs {fixture.get('away_team', '')} "
        f"({fixture.get('league_code', '?')}): model={best['model']} outcome={best['outcome']} "
        f"model_probability={best['model_probability']:.1%} market_probability={best['market_probability']:.1%} "
        f"edge={best['edge']:+.1%} decimal_odds={best['decimal_odds']:.2f} ({trust_note})"
    )


def _build_judge_agent(llm, tools: list):
    # No `response_format` here on purpose: LangGraph's built-in structured-final-response step
    # forces tool_choice, which Groq's gpt-oss models reject ("Tool call validation failed" / "Tool
    # choice is required, but model did not call a tool") — confirmed by hand against the real API,
    # not assumed. `_structure_answer` below does that step itself with method="json_schema"
    # instead, which is confirmed reliable on both Ollama and Groq.
    return create_react_agent(llm, tools=tools)


def _structure_answer(llm, transcript: str, n_fixtures: int) -> Answer:
    structured = llm.with_structured_output(Answer, method="json_schema")
    return structured.invoke(
        "Restate the following analysis in the required schema — do not add, remove, or change any "
        f"number, doc_id, or conclusion, only structure it. `notes` must have exactly {n_fixtures} "
        f"entries.\n\n{transcript}"
    )


def _template_answer(candidates: list[dict], edges: list[dict], trusted: dict[str, str]) -> Answer:
    notes = [
        FixtureNote(
            match_id=f["match_id"],
            home_team=f.get("home_team", ""),
            away_team=f.get("away_team", ""),
            take=_fixture_summary_line(f, edges, trusted.get(f["match_id"])),
        )
        for f in candidates
    ]
    return Answer(
        overview=f"{len(candidates)} fixture(s) considered; no LLM available, showing the raw computed numbers.",
        notes=notes,
        backend="template",
    )


# --------------------------------------------------------------------------------- node: parse_intent
def parse_intent(state: QAState, llm=None) -> dict:
    parsed = parse_ask_intent(state.question, llm=llm)
    if parsed is None:
        return {"intent": "unrecognized", "message": CANNOT_ANSWER_MESSAGE}
    if isinstance(parsed, TopBetsQuery):
        return {"intent": "top_bets", "n": parsed.n}
    if isinstance(parsed, FixtureQuery):
        return {"intent": "fixture", "home_team": parsed.home_team, "away_team": parsed.away_team}
    return {"intent": "fixtures_on_day", "day": parsed.day}


# --------------------------------------------------------------------------------- node: list_fixtures
def list_fixtures(state: QAState, features: pd.DataFrame) -> dict:
    day = pd.Timestamp(state.day)
    d = pd.to_datetime(features["date"])
    rows = features[(d.dt.date == day.date())]
    if rows.empty:
        return {"message": f"No fixtures found for {state.day}."}
    lines = [f"- {r['home_team']} vs {r['away_team']} ({r.get('league_code', '?')})" for _, r in rows.iterrows()]
    return {"message": f"Fixtures on {state.day}:\n" + "\n".join(lines)}


# --------------------------------------------------------------------------------- node: gather
def gather(
    state: QAState,
    features: pd.DataFrame,
    slice_evidence: pd.DataFrame,
    index: VectorIndex,
    models: list[MatchModel],
    top_bets_window_days: int,
    llm=None,
) -> dict:
    as_of = pd.Timestamp(state.as_of)
    if state.intent == "top_bets":
        candidates = top_bets_candidates(features, as_of, window_days=top_bets_window_days)
    else:
        row = resolve_fixture(features, as_of, state.home_team or "", state.away_team or "")
        candidates = row if row is not None else features.iloc[0:0]

    if candidates.empty:
        return {"candidates": [], "message": "No fixtures with real market odds were found for this question."}

    # Deterministic and cheap: fit/predict/edge for every candidate in the window, and pick which
    # model to trust per fixture from real backtest evidence. No LLM, no RAG call yet.
    predictions = run_models(features, candidates, as_of, models)
    edges = market_edges(candidates, predictions)
    edge_records = edges.to_dict(orient="records")
    model_names = [m.name for m in models]

    trusted: dict[str, str] = {}
    for _, fx in candidates.iterrows():
        slices = fixture_slice_values(fx)
        best, _matches = select_trusted_model(slice_evidence, slices, model_names, as_of)
        if best:
            trusted[fx["match_id"]] = best

    if state.intent == "top_bets":
        ranked = sorted(
            candidates.to_dict(orient="records"),
            key=lambda f: (_best_edge_row(f["match_id"], edge_records, trusted.get(f["match_id"])) or {}).get(
                "edge", -1.0
            ),
            reverse=True,
        )[: state.n or 4]
    else:
        ranked = candidates.to_dict(orient="records")

    # RAG context (and the narrow squad-value LLM hint) only for the fixtures that survived
    # ranking — never for the whole candidate window, which can be dozens of fixtures wide.
    context_docs: dict[str, list[dict]] = {}
    for f in ranked:
        fx = candidates[candidates["match_id"] == f["match_id"]].iloc[0]
        _slices, docs, _hint = structure_context(
            fx, index, as_of, fx["home_team"], fx["away_team"], fx["match_id"], llm=llm
        )
        context_docs[f["match_id"]] = [{"doc_id": d.doc_id, "text": d.text} for d in docs]

    return {
        "candidates": ranked,
        "predictions": predictions.to_dict(orient="records"),
        "edges": edge_records,
        "trusted": trusted,
        "context_docs": context_docs,
    }


# --------------------------------------------------------------------------------- node: judge
def judge(
    state: QAState,
    predictions: pd.DataFrame,
    slice_evidence: pd.DataFrame,
    index: VectorIndex,
    model_names: list[str],
    llm=None,
) -> dict:
    if not state.candidates:
        return {"answer": Answer(overview="No fixtures to report on.", backend="template")}

    summary = "\n".join(
        _fixture_summary_line(f, state.edges, state.trusted.get(f["match_id"])) for f in state.candidates
    )
    docs = []
    seen: set[str] = set()
    for f in state.candidates:
        for d in state.context_docs.get(f["match_id"], []):
            if d["doc_id"] not in seen:
                seen.add(d["doc_id"])
                docs.append(d)
    doc_block = ""
    if docs:
        doc_lines = "\n".join(f"[{d['doc_id']}] {d['text']}" for d in docs)
        doc_block = f"\nRetrieved context (cite only these doc_ids, never invent one):\n{doc_lines}\n"
    prompt = _JUDGE_PROMPT.format(summary=summary, doc_block=doc_block, n_fixtures=len(state.candidates))
    if state.retry_count > 0 and state.answer:
        prompt += (
            f"\n\nYour previous answer used numbers not found above: {state.answer.ungrounded_numbers}. "
            "Rewrite it using ONLY the numbers given above."
        )

    try:
        agent_llm = llm or get_llm_for("deep")
        tools = make_evidence_tools(predictions, slice_evidence, index, pd.Timestamp(state.as_of), model_names)
        agent = _build_judge_agent(agent_llm, tools)
        result = agent.invoke({"messages": [HumanMessage(prompt)]}, config={"recursion_limit": JUDGE_RECURSION_LIMIT})
        transcript = str(result["messages"][-1].content)
        if len(transcript) < 40:
            # The react loop hit `recursion_limit` mid-reasoning (e.g. a tool call it never got a
            # turn to use) and left nothing real to restate — sending this to `_structure_answer`
            # anyway just wastes a call asking the model to reformat text it was never given.
            raise RuntimeError(f"judge agent's final message looks incomplete: {transcript!r}")
        answer = _structure_answer(agent_llm, transcript, len(state.candidates))
        answer.backend = "llm"
        # Team names are already known deterministically per match_id — fill them in rather than
        # rely on the LLM to repeat them correctly into the structured schema.
        fixtures_by_id = {f["match_id"]: f for f in state.candidates}
        for note in answer.notes:
            fx = fixtures_by_id.get(note.match_id)
            if fx:
                note.home_team = note.home_team or fx.get("home_team", "")
                note.away_team = note.away_team or fx.get("away_team", "")
    except Exception as exc:  # noqa: BLE001 - a broken local LLM must never block an answer
        logger.warning("judge: LLM/tool failure (%s) — falling back to template", exc)
        return {"answer": _template_answer(state.candidates, state.edges, state.trusted)}

    # The computed summary counts as a grounding source alongside the retrieved docs (`verify_citations`
    # otherwise only knows about retrieved-doc text), and a citation must name a doc_id actually given.
    grounding_docs = [Document(d["doc_id"], d["text"]) for d in docs] + [Document("computed", summary)]
    grounded_doc_ids = {d["doc_id"] for d in docs}
    ungrounded: list[str] = []
    for note in answer.notes:
        _ok, _cited, missing_numbers = verify_citations(note.take, grounding_docs)
        ungrounded.extend(missing_numbers)
        ungrounded.extend(cid for cid in note.citations if cid not in grounded_doc_ids)
    answer.citations_grounded = not ungrounded
    answer.ungrounded_numbers = sorted(set(ungrounded))
    return {"answer": answer, "retry_count": state.retry_count + 1}


def _judge_needs_retry(state: QAState) -> str:
    """Retries `judge` exactly once (`retry_count` counts attempts: 0 -> 1 after the first call,
    so a second, ungrounded pass at `retry_count == 1` is allowed one more try, then stops)."""
    if state.answer and not state.answer.citations_grounded and state.retry_count < 2:
        return "retry"
    return "done"


def _route_intent(state: QAState) -> str:
    if state.intent == "unrecognized":
        return "cannot_answer"
    if state.intent == "fixtures_on_day":
        return "list_fixtures"
    return "gather"


def cannot_answer(state: QAState) -> dict:
    return {"message": state.message or CANNOT_ANSWER_MESSAGE}


def build_qa_graph(
    features: pd.DataFrame,
    slice_evidence: pd.DataFrame,
    index: VectorIndex,
    models: list[MatchModel],
    llm=None,
    top_bets_window_days: int = 14,
    checkpointer=None,
):
    """Compiles the graph above, closing over the fixed environment (`features`, `slice_evidence`,
    `index`, `models`) so `graph.invoke(QAState(question=..., as_of=...))` only needs the
    per-question inputs."""
    model_names = [m.name for m in models]
    g: StateGraph = StateGraph(QAState)
    g.add_node("parse_intent", lambda s: parse_intent(s, llm=llm))
    g.add_node("list_fixtures", lambda s: list_fixtures(s, features))
    g.add_node(
        "gather",
        lambda s: gather(s, features, slice_evidence, index, models, top_bets_window_days, llm=llm),
    )
    g.add_node("judge", lambda s: judge(s, pd.DataFrame(s.predictions), slice_evidence, index, model_names, llm=llm))
    g.add_node("cannot_answer", cannot_answer)
    g.set_entry_point("parse_intent")
    g.add_conditional_edges(
        "parse_intent",
        _route_intent,
        {"cannot_answer": "cannot_answer", "list_fixtures": "list_fixtures", "gather": "gather"},
    )
    g.add_edge("list_fixtures", END)
    g.add_edge("cannot_answer", END)
    g.add_edge("gather", "judge")
    g.add_conditional_edges("judge", _judge_needs_retry, {"retry": "judge", "done": END})
    if checkpointer is None:
        import sqlite3

        from langgraph.checkpoint.sqlite import SqliteSaver

        checkpointer = SqliteSaver(sqlite3.connect(":memory:", check_same_thread=False))
    return g.compile(checkpointer=checkpointer)


def ask_question(graph, question: str, as_of: pd.Timestamp, thread_id: str | None = None) -> QAState:
    """Runs the compiled graph once and returns the final `QAState` (its `.answer` or `.message`
    is what a caller renders). Each call gets its own checkpoint thread unless `thread_id` is given."""
    import uuid

    cfg = {"configurable": {"thread_id": thread_id or uuid.uuid4().hex}}
    result = graph.invoke(QAState(question=question, as_of=pd.Timestamp(as_of)), cfg)
    return QAState.model_validate(result)
