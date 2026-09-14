"""Stage 2 of the `ask` agent's read-only tool set: what each model predicted, what the real
backtest evidence says about trusting a model in this fixture's circumstances, and any further
retrieved context — everything the `judge` node's tool-calling agent may look up before writing
its answer. Every tool validates its own arguments against a closed set before touching any data,
and every lookup is bounded (a fixed `k`, a capped query length) — an LLM-generated argument can
only ever read, never reach an unbounded query or an unknown model/slice.
"""

from __future__ import annotations

import pandas as pd
from langchain_core.tools import tool

from pitch_edge.backtest.slices import SLICE_DIMENSIONS, nearest_checkpoint
from pitch_edge.rag.index import VectorIndex

MAX_QUERY_CHARS = 200
MAX_SEARCH_K = 8


def make_evidence_tools(
    predictions: pd.DataFrame,
    slice_evidence: pd.DataFrame,
    index: VectorIndex,
    as_of: pd.Timestamp,
    valid_models: list[str],
    live_market: dict[str, dict] | None = None,
) -> list:
    valid_model_set = set(valid_models)
    valid_dim_set = set(SLICE_DIMENSIONS)
    live_market = live_market or {}

    @tool
    def get_model_predictions(match_id: str) -> str:
        """Every model's predicted home/draw/away probability for this fixture."""
        rows = predictions[predictions["match_id"] == match_id]
        if rows.empty:
            return f"no predictions found for match_id={match_id}"
        return "\n".join(
            f"model={r['model']} p_home={r['p_home']:.3f} p_draw={r['p_draw']:.3f} p_away={r['p_away']:.3f}"
            for _, r in rows.iterrows()
        )

    @tool
    def get_backtest_evidence(model_name: str, slice_dim: str, slice_value: str) -> str:
        """Real walk-forward evidence for how well `model_name` has performed when `slice_dim`
        (e.g. 'league_code', 'referee', 'rest_days_gap_bucket') equals `slice_value`, at the most
        recent checkpoint strictly before this question's as-of date. Returns 'no evidence' if
        `model_name`/`slice_dim` isn't recognized or nothing matches — never invent a number."""
        if model_name not in valid_model_set:
            return f"unknown model={model_name!r} — not one of the models available for this question"
        if slice_dim not in valid_dim_set:
            return f"unknown slice_dim={slice_dim!r} — not one of {sorted(valid_dim_set)}"
        cp = nearest_checkpoint(slice_evidence, as_of)
        if cp.empty:
            return "no backtest evidence available before this question's as-of date"
        row = cp[(cp["model"] == model_name) & (cp["slice_dim"] == slice_dim) & (cp["slice_value"] == slice_value)]
        if row.empty:
            return f"no evidence for model={model_name} {slice_dim}={slice_value} (too few matches, or never seen)"
        r = row.iloc[0]
        sig = "significant" if bool(r["significant"]) else "NOT significant"
        return (
            f"model={model_name} {slice_dim}={slice_value} n={int(r['n'])} edge_bits={r['edge_bits']:+.4f} "
            f"({sig}, q={r['q_value']:.3f}) as of checkpoint {r['checkpoint_date']}"
        )

    @tool
    def get_price_history(match_id: str) -> str:
        """The most recent real Polymarket no-vig price for this fixture (strictly before this
        question's as-of time), and how many ticks were seen. Returns 'no live market data' if
        this fixture has no mapped Polymarket market or no trades yet."""
        m = live_market.get(match_id)
        if not m:
            return "no live market data for this fixture"
        return (
            f"Polymarket, {m['n_ticks']} ticks, latest at {m['latest_ts']}: "
            f"p_home={m['p_home']:.3f} p_draw={m['p_draw']:.3f} p_away={m['p_away']:.3f}"
        )

    as_of_str = str(as_of.date())

    @tool
    def search_context(query: str) -> str:
        """Search retrieved news/match/prediction context for this question. `query` is truncated
        to a short phrase. Documents dated on/after this question's as-of date are dropped —
        same client-side post-filter as `rag/fixture_context.py::gather_fixture_context` (the
        `gather` node's own RAG lookup), applied here too since this tool previously had none,
        meaning `judge` could otherwise be handed a real future-dated match report."""
        q = query[:MAX_QUERY_CHARS]
        # Over-fetch since the as_of filter below may drop some hits.
        hits = index.query(q, k=MAX_SEARCH_K * 3)
        kept = [
            (doc, score)
            for doc, score in hits
            if not any(doc.metadata.get(key, "") >= as_of_str for key in ("date", "published_at") if doc.metadata.get(key))
        ][:MAX_SEARCH_K]
        if not kept:
            return "no matching documents"
        return "\n".join(f"[{doc.doc_id}] {doc.text[:400]}" for doc, _score in kept)

    return [get_model_predictions, get_backtest_evidence, get_price_history, search_context]
