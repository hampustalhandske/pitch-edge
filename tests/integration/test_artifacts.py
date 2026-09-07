"""Artifacts stage: team strengths, feature importance, GNN embeddings, in-play paths, data universe."""

from __future__ import annotations

import json

import pytest

from pitch_edge.artifacts import build_all_artifacts, build_data_universe
from pitch_edge.config import get_settings
from pitch_edge.data.ingest import store_matches
from pitch_edge.features.build import FeatureBuilder

pytestmark = pytest.mark.integration


def test_build_all_artifacts_from_synthetic_warehouse(warehouse, synthetic_league_matches, statsbomb_events_df):
    store_matches(warehouse, synthetic_league_matches)
    # a handful of "matches" of events so the GNN / in-play stages have enough graphs & sequences
    frames = []
    for i in range(8):
        e = statsbomb_events_df.copy()
        e["statsbomb_match_id"] = 900 + i
        e["event_id"] = e["event_id"] + f"_{i}"
        if i % 2:  # alternate final results so the in-play baseline sees more than one class
            shots = e["type"] == "Shot"
            e.loc[shots & (e["team"] == "Home FC"), "shot_outcome"] = "Saved"
            e.loc[shots & (e["team"] == "Away FC") & (e["minute"] == 40), "shot_outcome"] = "Goal"
        frames.append(e)
    import pandas as pd

    warehouse.upsert("statsbomb_events", pd.concat(frames), keys=["event_id"])
    features = FeatureBuilder().build(synthetic_league_matches)
    warehouse.replace("features", features)

    report = build_all_artifacts(warehouse, features)
    assert report["team_strength"] == 12
    assert report["feature_importance"] > 5
    assert report["player_embeddings"] == 6
    assert report["inplay_paths"] > 0
    assert warehouse.count("player_similarity") > 0
    paths = warehouse.read("inplay_paths")
    assert {"home", "draw", "away", "base_home", "minute"} <= set(paths.columns)
    uni = json.loads((get_settings().artifacts_dir / "data_universe.json").read_text())
    assert uni["matches"] == 528 and uni["leagues"] == 1


def test_data_universe_on_empty_warehouse(warehouse):
    out = build_data_universe(warehouse)
    assert "matches" not in out and out["odds"] == 0
