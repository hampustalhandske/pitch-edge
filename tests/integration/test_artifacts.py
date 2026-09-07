"""Artifacts stage: team strengths, feature importance, data universe."""

from __future__ import annotations

import json

import pytest

from pitch_edge.artifacts import build_all_artifacts, build_data_universe
from pitch_edge.config import get_settings
from pitch_edge.data.ingest import store_matches
from pitch_edge.features.build import FeatureBuilder

pytestmark = pytest.mark.integration


def test_build_all_artifacts_from_synthetic_warehouse(warehouse, synthetic_league_matches):
    store_matches(warehouse, synthetic_league_matches)
    features = FeatureBuilder().build(synthetic_league_matches)
    warehouse.replace("features", features)

    report = build_all_artifacts(warehouse, features)
    assert report["team_strength"] == 12
    assert report["feature_importance"] > 5
    uni = json.loads((get_settings().artifacts_dir / "data_universe.json").read_text())
    assert uni["matches"] == 528 and uni["leagues"] == 1


def test_data_universe_on_empty_warehouse(warehouse):
    out = build_data_universe(warehouse)
    assert "matches" not in out and out["odds"] == 0
