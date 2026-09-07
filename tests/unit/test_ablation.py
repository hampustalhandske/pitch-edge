from __future__ import annotations

import pytest

from pitch_edge.ablation import FEATURE_GROUPS, run_ablation
from pitch_edge.backtest.engine import WalkForwardConfig
from pitch_edge.features.build import FeatureBuilder
from pitch_edge.models.gbdt import GBDTMatchModel

pytestmark = pytest.mark.unit


def test_exclude_prefixes_removes_group(synthetic_league_matches):
    f = FeatureBuilder().build(synthetic_league_matches)
    m = GBDTMatchModel(n_estimators=10, exclude_prefixes=FEATURE_GROUPS["referee"], name_suffix="_x").fit(f)
    assert m.name == "gbdt_x"
    assert not any(c.startswith("ref_") for c in m.card()["features"])
    full = GBDTMatchModel(n_estimators=10).fit(f)
    assert any(c.startswith("ref_") for c in full.card()["features"])


def test_run_ablation_table_shape(synthetic_league_matches):
    f = FeatureBuilder().build(synthetic_league_matches)
    cfg = WalkForwardConfig(min_train_matches=300, retrain_every_days=180, edge_threshold=0.0, calibration_min_rows=100)
    table, results = run_ablation(f, groups=["referee", "elo"], config=cfg, n_estimators=15)
    assert list(table["group_removed"]) == ["referee", "elo"]
    assert {"delta_log_loss", "delta_clv_pct", "n_bets_full"} <= set(table.columns)
    assert set(results) == {"gbdt_full", "gbdt_no_referee", "gbdt_no_elo"}
