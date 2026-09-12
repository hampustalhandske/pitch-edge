from __future__ import annotations

import pandas as pd
import pytest

from pitch_edge.rag.documents import (
    Document,
    match_documents,
    news_documents,
    prediction_documents,
    statsbomb_documents,
)
from pitch_edge.rag.eval import evaluate_retrieval, summarize_eval, synthetic_questions
from pitch_edge.rag.generate import GroundedGenerator, scouting_report, verify_citations
from pitch_edge.rag.index import VectorIndex
from pitch_edge.rag.retrieval import HybridRetriever, reciprocal_rank_fusion

pytestmark = pytest.mark.unit


# ----------------------------------------------------------------------------- documents
def test_match_documents_include_stats_and_odds(synthetic_league_matches):
    docs = match_documents(synthetic_league_matches, limit=5)
    assert len(docs) == 5
    assert "Shots" in docs[0].text and "Pinnacle" in docs[0].text and "referee" in docs[0].text
    assert docs[0].metadata["type"] == "match"


def test_prediction_documents_cite_drivers():
    preds = pd.DataFrame(
        {
            "match_id": ["m1"],
            "date": [pd.Timestamp("2024-01-01")],
            "home_team": ["A"],
            "away_team": ["B"],
            "p_home": [0.5],
            "p_draw": [0.25],
            "p_away": [0.25],
            "mkt_home": [0.45],
            "mkt_draw": [0.27],
            "mkt_away": [0.28],
            "edge_home": [0.05],
            "edge_draw": [-0.02],
            "edge_away": [-0.03],
        }
    )
    feats = pd.DataFrame({"match_id": ["m1"], "elo_diff": [120.0], "form_diff_r5": [0.8], "away_rest_days": [3.0]})
    d = prediction_documents(preds, "gbdt", feats)[0]
    assert "Largest edge: home (+5.0%)" in d.text and "Elo gap +120" in d.text and "away rest 3d" in d.text


def test_news_and_statsbomb_documents(statsbomb_events_df):
    news = pd.DataFrame(
        {
            "item_id": ["n1"],
            "feed": ["bbc"],
            "published_at": [pd.Timestamp("2024-01-01")],
            "title": ["T"],
            "summary": ["S"],
            "link": [""],
            "team": ["A"],
            "sentiment": [0.3],
        }
    )
    assert news_documents(news)[0].metadata["team"] == "A"
    sbm = pd.DataFrame(
        {
            "statsbomb_match_id": [999],
            "home_team": ["Home FC"],
            "away_team": ["Away FC"],
            "home_goals": [1.0],
            "away_goals": [0.0],
            "league": ["Test"],
            "date": [pd.Timestamp("2020-01-01")],
        }
    )
    d = statsbomb_documents(statsbomb_events_df, sbm)[0]
    assert "xG" in d.text and d.text.startswith("Home FC 1-0 Away FC")


# ----------------------------------------------------------------------------- retrieval
def test_reciprocal_rank_fusion_prefers_agreement():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]])
    assert fused["a"] == pytest.approx(fused["b"]) and fused["a"] > fused["c"] > 0 and "d" in fused


def test_sparse_only_index_roundtrip_filter_and_dedupe(tmp_path):
    idx = VectorIndex(persist_dir=tmp_path, force_tfidf=True)
    idx.add(
        [
            Document("a", "Arsenal beat Chelsea with a late goal", {"type": "match"}),
            Document("b", "Model gbdt likes Liverpool away", {"type": "prediction"}),
            Document("a", "duplicate id ignored", {"type": "match"}),
        ]
    )
    assert idx.count() == 2 and idx.backend == "bm25"
    assert idx.query("Arsenal Chelsea", k=1)[0][0].doc_id == "a"
    assert [d.doc_id for d, _ in idx.query("Liverpool", k=5, where={"type": "prediction"})] == ["b"]
    hits = idx.query_hits("Arsenal late goal", k=2)
    assert hits[0].sources == ["bm25"] and hits[0].dense_rank is None


def test_hybrid_retriever_degrades_without_dense(tmp_path, monkeypatch):
    import pitch_edge.rag.retrieval as r

    monkeypatch.setattr(r, "RERANKER_MODEL", "definitely/not-a-model")
    hr = HybridRetriever(
        persist_dir=tmp_path, embedding_model="definitely/not-a-model", reranker_model="definitely/not-a-model"
    )
    assert hr.backend == "bm25"
    hr.add([Document("x", "Malmö FF won the Allsvenskan title", {"type": "match"})])
    assert hr.query("who won allsvenskan")[0].document.doc_id == "x"


def test_retrieval_eval_harness(synthetic_league_matches, tmp_path):
    docs = match_documents(synthetic_league_matches, limit=60)
    idx = VectorIndex(persist_dir=tmp_path, force_tfidf=True)
    idx.add(docs)
    items = synthetic_questions(docs, n=20, seed=1)
    assert items and all(it.kind in ("match_result", "match_stats") for it in items)
    res = evaluate_retrieval(idx, items, k=5)
    summ = summarize_eval(res, k=5)
    assert set(summ["kind"]) >= {"ALL"} and 0 <= summ.loc[summ["kind"] == "ALL", "mrr"].iloc[0] <= 1
    assert res[res["kind"] == "match_result"]["hit@5"].mean() > 0.5  # team names + date are exact tokens


# ----------------------------------------------------------------------------- generation
def test_verify_citations_flags_ungrounded_numbers():
    docs = [Document("pred:gbdt:m1", "Model gbdt: P(home)=55.0%, P(draw)=25.0%.", {"type": "prediction"})]
    ok, cited, missing = verify_citations("Home is favoured at 55.0% [pred:gbdt:m1].", docs)
    assert ok and cited == ["pred:gbdt:m1"] and missing == []
    ok2, _, missing2 = verify_citations("Home is favoured at 61% [pred:gbdt:m1].", docs)
    assert not ok2 and missing2 == ["61"]


def test_template_generator_is_grounded_and_cites():
    gen = GroundedGenerator(force_template=True)
    docs = [
        Document("pred:gbdt:m1", "Model gbdt: P(home)=55.0%.", {"type": "prediction"}),
        Document("match:m0", "A 2-1 B.", {"type": "match"}),
    ]
    ans = gen.answer("why home?", docs)
    assert (
        ans.backend == "template"
        and "[pred:gbdt:m1]" in ans.text
        and set(ans.citations) == {"pred:gbdt:m1", "match:m0"}
    )
    assert ans.verified and "nothing here is a betting instruction" in ans.text
    assert gen.answer("q", []).citations == []
    assert scouting_report(gen, "Team A", docs).backend == "template"


def _fake_anthropic(monkeypatch, text: str, stop_reason: str = "end_turn"):
    calls: dict = {}

    class _Block:
        type = "text"

        def __init__(self, t):
            self.text = t

    class _Resp:
        def __init__(self):
            self.stop_reason = stop_reason
            self.content = [_Block(text)]
            self.model = "claude-fable-5-1"

    class _Messages:
        def create(self, **kwargs):
            calls.update(kwargs)
            return _Resp()

    class _Beta:
        def __init__(self):
            self.messages = _Messages()

    class _Client:
        def __init__(self, *a, **k):
            self.messages = _Messages()
            self.beta = _Beta()

    import anthropic

    monkeypatch.setattr(anthropic, "Anthropic", _Client)
    return calls


class _FakeLocalLLM:
    model = "llama3.1:8b"

    def __init__(self, text):
        self._text = text

    def invoke(self, messages):
        class _Msg:
            def __init__(self, content):
                self.content = content

        return _Msg(self._text)


def test_local_llm_is_tried_before_claude(monkeypatch):
    """Backend precedence is local-first: a reachable local model must win even when a Claude
    key is also configured, and Claude's client must never be constructed/called in that case."""
    monkeypatch.setattr(
        "pitch_edge.agents.llm.get_llm",
        lambda model=None: _FakeLocalLLM("Home is favoured at 55.0% [pred:gbdt:m1]."),
    )

    def _boom(*a, **k):
        raise AssertionError("Claude client must not be constructed when the local LLM succeeds")

    import anthropic

    monkeypatch.setattr(anthropic, "Anthropic", _boom)
    gen = GroundedGenerator(api_key="test-key")
    assert gen.backend == "local"
    docs = [Document("pred:gbdt:m1", "Model gbdt: P(home)=55.0%.", {"type": "prediction"})]
    ans = gen.answer("why?", docs)
    assert ans.backend == "local" and ans.citations == ["pred:gbdt:m1"] and ans.verified


def test_local_llm_failure_falls_back_to_claude(monkeypatch):
    calls = _fake_anthropic(monkeypatch, "Home is favoured [pred:gbdt:m1].")

    class _RaisingLLM:
        model = "llama3.1:8b"

        def invoke(self, messages):
            raise RuntimeError("ollama not running")

    monkeypatch.setattr("pitch_edge.agents.llm.get_llm", lambda model=None: _RaisingLLM())
    gen = GroundedGenerator(api_key="test-key")
    docs = [Document("pred:gbdt:m1", "Model gbdt: P(home)=55.0%.", {"type": "prediction"})]
    ans = gen.answer("why?", docs)
    assert ans.backend == "claude" and calls  # local failed silently, Claude still ran


def test_fable_path_uses_beta_fallbacks_no_thinking_param_and_verifies(monkeypatch):
    calls = _fake_anthropic(monkeypatch, "Home is favoured at 55.0% [pred:gbdt:m1]; the draw sits at 30%.")
    gen = GroundedGenerator(api_key="test-key")  # default model is claude-fable-5-1
    docs = [Document("pred:gbdt:m1", "Model gbdt: P(home)=55.0%, P(draw)=25.0%.", {"type": "prediction"})]
    ans = gen.answer("why?", docs)
    assert ans.backend == "claude" and ans.citations == ["pred:gbdt:m1"]
    assert calls["model"] == "claude-fable-5-1" and calls["fallbacks"] == "default"
    assert calls["betas"] == ["server-side-fallback-2026-07-01"] and "thinking" not in calls
    assert "Never invent" in calls["system"][0]["text"]
    assert not ans.verified and ans.unverified_numbers == ["30"] and "could not be matched" in ans.text


def test_opus_path_uses_adaptive_thinking(monkeypatch):
    calls = _fake_anthropic(monkeypatch, "Fine [m].")
    gen = GroundedGenerator(api_key="k", model="claude-opus-5")
    gen.answer("q", [Document("m", "Fine.", {"type": "match"})])
    assert calls["thinking"] == {"type": "adaptive"} and "fallbacks" not in calls


def test_refusal_falls_back_to_template(monkeypatch):
    _fake_anthropic(monkeypatch, "", stop_reason="refusal")
    ans = GroundedGenerator(api_key="k").answer("q", [Document("m", "Fine.", {"type": "match"})])
    assert ans.backend == "template"
