"""Warehouse -> feature frame -> backtests persisted -> RAG index -> signal pipeline, offline."""

from __future__ import annotations

import pandas as pd
import pytest

from pitch_edge.backtest.engine import WalkForwardConfig
from pitch_edge.data.ingest import store_matches
from pitch_edge.models import DixonColesMatchModel, GBDTMatchModel
from pitch_edge.pipeline import (
    build_rag_index,
    build_signal_pipeline,
    latest_backtest_tables,
    load_feature_frame,
    persist_features,
    run_backtests,
    synthetic_quotes_from_elo,
)
from pitch_edge.rag.index import VectorIndex

pytestmark = pytest.mark.integration


@pytest.fixture
def loaded(warehouse, synthetic_league_matches):
    store_matches(warehouse, synthetic_league_matches)
    return warehouse


def test_load_feature_frame_round_trips_odds(loaded):
    f = load_feature_frame(loaded, leagues=["SYN"])
    assert len(f) == 528
    assert {"PSH", "PSCH", "elo_diff", "h_gf_r5", "ref_home_bias"} <= set(f.columns)
    assert persist_features(loaded, f) == 528
    assert loaded.count("features") == 528


def test_run_backtests_persists_everything(loaded, tmp_path):
    f = load_feature_frame(loaded, leagues=["SYN"])
    cfg = WalkForwardConfig(min_train_matches=150, retrain_every_days=90, edge_threshold=0.0, calibration_min_rows=100)
    results = run_backtests(
        f,
        [DixonColesMatchModel(), GBDTMatchModel(n_estimators=25)],
        cfg,
        wh=loaded,
        data_dir=tmp_path / "data",
        report_dir=tmp_path / "r",
        label="t",
    )
    assert set(results) == {"dixon_coles", "gbdt"}
    summ, bets = latest_backtest_tables(loaded)
    assert len(summ) == 6 and not bets.empty
    assert loaded.count("model_predictions") == 2 * len(results["gbdt"].predictions)
    assert (tmp_path / "data" / "model_card_gbdt.json").exists() and (tmp_path / "r" / "CASE_STUDY.md").exists()


def test_rag_index_built_from_warehouse(loaded, tmp_path):
    f = load_feature_frame(loaded, leagues=["SYN"])
    persist_features(loaded, f)
    run_backtests(
        f,
        [DixonColesMatchModel()],
        WalkForwardConfig(min_train_matches=300, retrain_every_days=120),
        wh=loaded,
        report_dir=tmp_path / "r",
    )
    idx = build_rag_index(loaded, VectorIndex(persist_dir=tmp_path / "vec", force_tfidf=True), max_matches=100)
    assert idx.count() > 100
    hits = idx.query("Team01 vs Team02 model probability", k=3)
    assert hits and any(d.metadata["type"] in ("match", "prediction") for d, _ in hits)


def test_signal_pipeline_stops_at_gate_then_logs_paper_trades(loaded):
    f = load_feature_frame(loaded, leagues=["SYN"])
    model = GBDTMatchModel(n_estimators=25).fit(f)
    fixtures = f.sort_values("date").tail(6).copy()
    fixtures["date"] = pd.Timestamp("2030-01-01")
    quotes = synthetic_quotes_from_elo(fixtures, margin=1.02)
    quotes[0]["home"] = 6.0  # one clearly mispriced quote so a proposal survives the risk manager
    pipe = build_signal_pipeline(loaded, f, model, fixtures=fixtures, quotes=quotes)
    thread_id, state = pipe.run_to_gate()
    assert "alerts" not in state  # gate blocked everything
    props = pipe.pending_proposals(thread_id)
    assert props, "expected at least one proposal"
    key = f"{props[0]['match_id']}|{props[0]['outcome']}"
    final = pipe.resume_with_decisions(thread_id, {key: "approved"}, approved_by="tester")
    assert len(final["alerts"]) == 1 and final["alerts"][0]["approved_by"] == "tester"
    assert loaded.count("paper_trades") == 1
    assert any("human_approval" in line for line in final["log"])
