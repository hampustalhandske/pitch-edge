"""Vector index façade used by the pipeline, CLI and dashboard.

Backed by `HybridRetriever` (LangChain Chroma + HuggingFace embeddings ∪ BM25, RRF-fused,
cross-encoder re-ranked). `force_tfidf=True` keeps a tiny pure-sparse mode for tests and
fully offline machines. The public API (`add`, `query`, `count`, `backend`) is unchanged.
"""

from __future__ import annotations

from pathlib import Path

from pitch_edge.rag.documents import Document
from pitch_edge.rag.retrieval import HybridRetriever, RetrievalHit


class VectorIndex:
    def __init__(
        self,
        persist_dir: str | Path | None = None,
        collection: str = "pitch_edge",
        embedding_model: str | None = None,
        force_tfidf: bool = False,
        rerank: bool = True,
    ):
        self._r = HybridRetriever(
            persist_dir=persist_dir,
            collection=collection,
            embedding_model=embedding_model,
            dense=not force_tfidf,
            rerank=rerank and not force_tfidf,
        )
        self.persist_dir = self._r.persist_dir

    @property
    def backend(self) -> str:
        return self._r.backend

    def add(self, docs: list[Document]) -> None:
        self._r.add(docs)

    def query(self, text: str, k: int = 6, where: dict | None = None) -> list[tuple[Document, float]]:
        hits = self._r.query(text, k=k, where=where)
        return [(h.document, h.rerank_score if h.rerank_score is not None else h.score) for h in hits]

    def query_hits(self, text: str, k: int = 6, where: dict | None = None) -> list[RetrievalHit]:
        return self._r.query(text, k=k, where=where)

    def count(self) -> int:
        return self._r.count()
