"""Retrieval evaluation: synthetic QA generated from the corpus itself, scored by hit@k and MRR.

For every sampled match / prediction / news document we know the one document that answers a
templated question about it ("Who won X vs Y on <date>?"), so retrieval quality can be measured
without a human-labelled set. This runs in CI on the synthetic league and on demand
(`pitch-edge rag-eval`) against the real index; results go to `reports/rag_eval.csv`.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import pandas as pd

from pitch_edge.rag.documents import Document
from pitch_edge.rag.index import VectorIndex


@dataclass
class QAItem:
    question: str
    expected_doc_id: str
    kind: str


def synthetic_questions(docs: list[Document], n: int = 60, seed: int = 0) -> list[QAItem]:
    rng = random.Random(seed)
    items: list[QAItem] = []
    matches = [d for d in docs if d.metadata.get("type") == "match"]
    preds = [d for d in docs if d.metadata.get("type") == "prediction"]
    news = [d for d in docs if d.metadata.get("type") == "news"]
    for d in rng.sample(matches, min(n // 2, len(matches))):
        m = d.metadata
        items += [
            QAItem(
                f"Who won {m.get('home_team')} vs {m.get('away_team')} on {m.get('date')}?", d.doc_id, "match_result"
            ),
            QAItem(
                f"{m.get('home_team')} {m.get('away_team')} {m.get('date')} shots corners referee",
                d.doc_id,
                "match_stats",
            ),
        ]
    for d in rng.sample(preds, min(n // 4, len(preds))):
        m = d.metadata
        items.append(
            QAItem(
                f"What does the {m.get('model')} model say about {m.get('home_team')} against {m.get('away_team')} on {m.get('date')}?",
                d.doc_id,
                "prediction",
            )
        )
    for d in rng.sample(news, min(n // 4, len(news))):
        words = [w for w in d.text.split()[:8] if len(w) > 3]
        items.append(QAItem(" ".join(words), d.doc_id, "news"))
    return items[:n]


def evaluate_retrieval(index: VectorIndex, items: list[QAItem], k: int = 5) -> pd.DataFrame:
    rows = []
    for it in items:
        hits = index.query(it.question, k=k)
        ids = [d.doc_id for d, _ in hits]
        rank = ids.index(it.expected_doc_id) + 1 if it.expected_doc_id in ids else None
        rows.append(
            {
                "kind": it.kind,
                "question": it.question,
                "expected": it.expected_doc_id,
                "rank": rank,
                "hit@1": int(rank == 1),
                f"hit@{k}": int(rank is not None),
                "rr": (1.0 / rank) if rank else 0.0,
            }
        )
    return pd.DataFrame(rows)


def summarize_eval(results: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    if results.empty:
        return results
    g = (
        results.groupby("kind")
        .agg(n=("rr", "size"), hit_at_1=("hit@1", "mean"), hit_at_k=(f"hit@{k}", "mean"), mrr=("rr", "mean"))
        .reset_index()
    )
    total = pd.DataFrame(
        [
            {
                "kind": "ALL",
                "n": len(results),
                "hit_at_1": results["hit@1"].mean(),
                "hit_at_k": results[f"hit@{k}"].mean(),
                "mrr": results["rr"].mean(),
            }
        ]
    )
    return pd.concat([g, total], ignore_index=True)
