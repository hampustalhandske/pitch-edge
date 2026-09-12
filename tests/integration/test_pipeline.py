"""Warehouse -> feature frame -> backtests persisted -> RAG index, offline."""

from __future__ import annotations

import pytest

from pitch_edge.backtest.engine import WalkForwardConfig
from pitch_edge.data.ingest import store_matches
from pitch_edge.models import DixonColesMatchModel, GBDTMatchModel
from pitch_edge.pipeline import (
    build_rag_index,
    latest_backtest_tables,
    load_feature_frame,
    persist_features,
    run_backtests,
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
