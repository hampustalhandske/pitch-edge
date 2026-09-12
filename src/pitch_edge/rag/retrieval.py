"""Hybrid retrieval on LangChain: dense (Chroma + HuggingFace embeddings) ∪ sparse (BM25),
fused with reciprocal-rank fusion, then re-ranked by a cross-encoder.

Why hybrid: football questions mix exact tokens (team names, "2-1", "2024-03-10") where BM25
wins, with paraphrase ("why does the model like the away side") where dense embeddings win.
RRF needs no score calibration between the two, and the cross-encoder re-reads the top
candidates with the query — the standard, well-tested three-stage recipe.

Everything degrades gracefully: no embedding model → BM25 only; no cross-encoder → RRF order.

The embedder is pinned to `device="cpu"` — confirmed reproducible crash on Apple Silicon
otherwise: PyTorch's MPS (Metal) backend raises a low-level driver assertion (`failed assertion
_status < MTLCommandBufferStatusCommitted ... setCurrentCommandEncoder`) when the embedder and the
cross-encoder both run forward passes on the GPU in the same process. It's a small (~22M-parameter)
model, so CPU inference stays fast for the handful of documents a single retrieval call embeds.

The cross-encoder is deliberately left on its default device (MPS when available) rather than
also forced to CPU: `cross-encoder/ms-marco-MiniLM-L-6-v2` was confirmed, at the raw model level,
to produce `NaN` logits on CPU in this environment (not a dtype issue — reproduced with weights
already `float32`) while producing correct scores on MPS. Forcing it to CPU would silently corrupt
ranking rather than fix anything, so instead `query()` checks every rerank score for `NaN` and
degrades to the pre-rerank RRF order if any turn up, the same way a missing cross-encoder already
degrades — an honest fallback instead of a wrong one.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from pitch_edge.config import get_settings
from pitch_edge.rag.documents import Document

logger = logging.getLogger(__name__)

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass
class RetrievalHit:
    document: Document
    score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None
    rerank_score: float | None = None
    sources: list[str] = field(default_factory=list)


def _to_lc(doc: Document):
    from langchain_core.documents import Document as LCDocument

    meta = {
        k: (v if isinstance(v, str | int | float | bool) else str(v)) for k, v in doc.metadata.items() if v is not None
    }
    meta["doc_id"] = doc.doc_id
    return LCDocument(page_content=doc.text, metadata=meta)


def _from_lc(lc) -> Document:
    meta = dict(lc.metadata)
    doc_id = str(meta.pop("doc_id", ""))
    return Document(doc_id, lc.page_content, meta)


def reciprocal_rank_fusion(rankings: Iterable[list[str]], k: int = 60) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return scores


class HybridRetriever:
    """Chroma (dense) + BM25 (sparse) + cross-encoder reranking behind one `add` / `query` API."""

    def __init__(
        self,
        persist_dir: str | Path | None = None,
        collection: str = "pitch_edge",
        embedding_model: str | None = None,
        reranker_model: str | None = RERANKER_MODEL,
        dense: bool = True,
        rerank: bool = True,
    ):
        settings = get_settings()
        self.persist_dir = Path(persist_dir) if persist_dir else settings.vector_dir
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._docs: dict[str, Document] = {}
        self._bm25 = None
        self._bm25_ids: list[str] = []
        self._vs = None
        self._reranker = None
        self.backend = "bm25"
        if dense:
            try:
                from langchain_chroma import Chroma
                from langchain_huggingface import HuggingFaceEmbeddings

                emb = HuggingFaceEmbeddings(
                    model_name=embedding_model or settings.embedding_model, model_kwargs={"device": "cpu"}
                )
                self._vs = Chroma(
                    collection_name=collection,
                    embedding_function=emb,
                    persist_directory=str(self.persist_dir),
                    collection_metadata={"hnsw:space": "cosine"},
                )
                self.backend = "hybrid"
                self._load_existing_into_bm25()
            except Exception as exc:  # noqa: BLE001 - offline / model unavailable
                logger.warning("Dense retriever unavailable (%s); BM25 only", exc)
                self._vs = None
        if rerank and reranker_model:
            try:
                from sentence_transformers import CrossEncoder

                self._reranker = CrossEncoder(reranker_model)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Cross-encoder unavailable (%s); skipping rerank", exc)
                self._reranker = None

    # ----------------------------------------------------------------- state
    def _load_existing_into_bm25(self) -> None:
        """Rehydrate the sparse index from the persisted Chroma collection so restarts keep hybrid search."""
        if self._vs is None:
            return
        try:
            raw = self._vs.get(include=["documents", "metadatas"])
        except Exception:  # noqa: BLE001
            return
        docs = [
            Document(str(m.get("doc_id", i)), t, dict(m))
            for i, (t, m) in enumerate(zip(raw.get("documents", []), raw.get("metadatas", []), strict=False))
        ]
        if docs:
            self._docs.update({d.doc_id: d for d in docs})
            self._rebuild_bm25()

    def _rebuild_bm25(self) -> None:
        from rank_bm25 import BM25Okapi

        self._bm25_ids = list(self._docs)
        corpus = [_tokenize(self._docs[i].text) for i in self._bm25_ids]
        self._bm25 = BM25Okapi(corpus) if corpus else None

    def add(self, docs: list[Document]) -> None:
        new: list[Document] = []
        seen: set[str] = set()
        for d in docs:  # first occurrence wins, within the batch and against the existing index
            if d.doc_id in self._docs or d.doc_id in seen:
                continue
            seen.add(d.doc_id)
            new.append(d)
        if not new:
            return
        for d in new:
            self._docs[d.doc_id] = d
        self._rebuild_bm25()
        if self._vs is not None:
            for i in range(0, len(new), 256):
                chunk = new[i : i + 256]
                self._vs.add_documents([_to_lc(d) for d in chunk], ids=[d.doc_id for d in chunk])

    def count(self) -> int:
        return len(self._docs)

    # ----------------------------------------------------------------- query
    def query(self, text: str, k: int = 6, where: dict | None = None, candidates: int = 30) -> list[RetrievalHit]:
        if not self._docs:
            return []
        rankings: list[list[str]] = []
        dense_rank: dict[str, int] = {}
        sparse_rank: dict[str, int] = {}
        if self._vs is not None:
            try:
                flt = dict(where or {}) or None
                res = self._vs.similarity_search(text, k=candidates, filter=flt)
                ids = [str(r.metadata.get("doc_id")) for r in res]
                dense_rank = {d: i for i, d in enumerate(ids)}
                rankings.append(ids)
            except Exception as exc:  # noqa: BLE001
                logger.info("dense search failed (%s)", exc)
        if self._bm25 is not None:
            q_tokens = _tokenize(text)
            scores = self._bm25.get_scores(q_tokens)
            order = sorted(range(len(scores)), key=lambda i: -scores[i])
            ids = [self._bm25_ids[i] for i in order if scores[i] > 0][:candidates]
            if not ids:
                # tiny corpora: BM25 idf collapses to 0 for common terms — fall back to raw token overlap
                q_set = set(q_tokens)
                overlap = {d: len(q_set & set(_tokenize(self._docs[d].text))) for d in self._bm25_ids}
                ids = [d for d in sorted(overlap, key=lambda d: -overlap[d]) if overlap[d] > 0][:candidates]
            if where:
                ids = [i for i in ids if all(self._docs[i].metadata.get(k_) == v for k_, v in where.items())]
            sparse_rank = {d: i for i, d in enumerate(ids)}
            rankings.append(ids)
        fused = reciprocal_rank_fusion(rankings)
        if not fused:
            return []
        top = sorted(fused, key=lambda d: -fused[d])[: max(k * 3, 10)]
        hits = [
            RetrievalHit(
                self._docs[d],
                fused[d],
                dense_rank.get(d),
                sparse_rank.get(d),
                sources=[s for s, r in (("dense", dense_rank), ("bm25", sparse_rank)) if d in r],
            )
            for d in top
            if d in self._docs
        ]
        if self._reranker is not None and hits:
            try:
                pairs = [(text, h.document.text) for h in hits]
                rs = [float(s) for s in self._reranker.predict(pairs)]
                if any(s != s for s in rs):  # NaN != NaN — see module docstring
                    raise ValueError("cross-encoder returned NaN score(s)")
            except Exception as exc:  # noqa: BLE001 - a broken reranker must fall back, not corrupt order
                logger.warning("Cross-encoder rerank failed (%s); keeping RRF order", exc)
            else:
                for h, s in zip(hits, rs, strict=True):
                    h.rerank_score = s
                hits.sort(key=lambda h: -(h.rerank_score if h.rerank_score is not None else -1e9))
        return hits[:k]


def _tokenize(text: str) -> list[str]:
    import re

    return re.findall(r"[a-z0-9åäöéü']+", text.lower())
