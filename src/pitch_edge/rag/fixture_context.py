"""Per-fixture RAG retrieval for the agentic signals pipeline.

`VectorIndex.query` is a cheap, stateless, per-call operation (the embedding model and
cross-encoder are loaded once at `HybridRetriever` construction, not per call) — safe to call once
per fixture in a loop rather than only as a batch/offline job.
"""

from __future__ import annotations

from pitch_edge.rag.documents import Document
from pitch_edge.rag.index import VectorIndex

_CONTEXT_DOC_TYPES = {"news", "match", "prediction"}
_DATE_METADATA_KEYS = ("date", "published_at")


def gather_fixture_context(
    index: VectorIndex, home: str, away: str, match_id: str, k: int = 6, as_of: str | None = None
) -> list[Document]:
    """News mentioning either team + past head-to-head match docs + this match_id's own prediction
    doc if indexed.

    Both the doc-type restriction and (when given) the `as_of` date cutoff are applied on the
    returned documents rather than via the index's own `where` clause: the BM25/tfidf fallback
    backend (`force_tfidf=True`, used offline/in tests) only matches `where` values by exact
    equality, so an operator filter like `{"$in": [...]}` silently matches nothing there even though
    it works against a real Chroma store — client-side filtering behaves the same on every backend.

    `as_of` (an ISO date string) excludes any retrieved doc dated on or after it — required for the
    T0 replay-eval harness (`backtest/replay.py`), whose production `VectorIndex` already contains
    real match-report/prediction docs (with the real score baked into the text) for fixtures still
    "in the future" from the replay's point of view. Without this, the RAG context handed to
    `agents/reviewer.py` could leak the real outcome before it is supposed to be revealed."""
    query = f"{home} vs {away} team news injuries form"
    hits = index.query(query, k=k * 4)
    docs = [doc for doc, _score in hits if doc.metadata.get("type") in _CONTEXT_DOC_TYPES]
    if as_of:
        docs = [
            d
            for d in docs
            if not any(d.metadata.get(key, "") >= as_of for key in _DATE_METADATA_KEYS if d.metadata.get(key))
        ]
    return docs[:k]
