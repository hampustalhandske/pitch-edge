"""gather_fixture_context: retrieves news/match/prediction docs, and `as_of` must exclude anything
dated on or after it — the only thing preventing a replay-eval run from leaking a real future score
out of a production index into the reviewer's prompt."""

from __future__ import annotations

import pytest

from pitch_edge.rag.documents import Document
from pitch_edge.rag.fixture_context import gather_fixture_context
from pitch_edge.rag.index import VectorIndex

pytestmark = pytest.mark.unit


@pytest.fixture
def seeded_index(tmp_path):
    idx = VectorIndex(persist_dir=tmp_path, force_tfidf=True)
    idx.add(
        [
            Document(
                "match:past1",
                "Alpha 2-1 Beta (Premier, 2024-01-01, season 2023/24).",
                {"type": "match", "match_id": "past1", "date": "2024-01-01"},
            ),
            Document(
                "match:future1",
                "Alpha 3-0 Beta (Premier, 2024-06-15, season 2023/24). Alpha dominant.",
                {"type": "match", "match_id": "future1", "date": "2024-06-15"},
            ),
            Document(
                "news:1",
                "Alpha vs Beta team news: no injuries reported.",
                {"type": "news", "published_at": "2024-06-10"},
            ),
        ]
    )
    return idx


def test_gather_fixture_context_without_as_of_returns_everything(seeded_index):
    docs = gather_fixture_context(seeded_index, "Alpha", "Beta", "future1", k=6)
    ids = {d.doc_id for d in docs}
    assert "match:future1" in ids  # without as_of, the real future score is visible


def test_gather_fixture_context_as_of_excludes_same_or_later_dated_docs(seeded_index):
    docs = gather_fixture_context(seeded_index, "Alpha", "Beta", "future1", k=6, as_of="2024-06-15")
    ids = {d.doc_id for d in docs}
    assert "match:future1" not in ids  # dated exactly on as_of -> excluded
    assert "news:1" in ids  # published_at 2024-06-10 is strictly before as_of -> kept
    assert "match:past1" in ids  # strictly before as_of -> kept


def test_gather_fixture_context_as_of_keeps_strictly_earlier_docs(seeded_index):
    docs = gather_fixture_context(seeded_index, "Alpha", "Beta", "future1", k=6, as_of="2024-06-11")
    ids = {d.doc_id for d in docs}
    assert "news:1" in ids  # published_at 2024-06-10 is strictly before as_of
    assert "match:future1" not in ids
    assert "match:past1" in ids
