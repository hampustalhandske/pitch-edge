"""Circumstance-stratified evidence: sample-size floor, real-edge detection, FDR correction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pitch_edge.backtest.slices import (
    MIN_SLICE_N,
    fixture_slice_values,
    nearest_checkpoint,
    select_trusted_model,
    slice_evidence,
)

pytestmark = pytest.mark.unit


def _base_frame(n: int, rng: np.random.Generator) -> pd.DataFrame:
    result = rng.integers(0, 3, size=n)
    return pd.DataFrame(
        {
            "match_id": [f"m{i}" for i in range(n)],
            "result": result,
            "mkt_home": 0.4,
            "mkt_draw": 0.3,
            "mkt_away": 0.3,
        }
    )


def _with_true_probs(df: pd.DataFrame, home: float, draw: float, away: float) -> pd.DataFrame:
    return df.assign(p_home=home, p_draw=draw, p_away=away)


def test_slice_below_min_n_dropped():
    rng = np.random.default_rng(0)
    big = _with_true_probs(_base_frame(MIN_SLICE_N + 10, rng), 0.4, 0.3, 0.3)
    small = _with_true_probs(_base_frame(MIN_SLICE_N - 10, rng), 0.4, 0.3, 0.3)
    predictions = pd.concat([big, small], ignore_index=True)
    features = pd.DataFrame(
        {"match_id": predictions["match_id"], "referee": ["RefA"] * len(big) + ["RefB"] * len(small)}
    )

    out = slice_evidence(predictions, features, "gbdt")

    refs = out[out["slice_dim"] == "referee"]["slice_value"].tolist()
    assert "RefA" in refs
    assert "RefB" not in refs


def test_real_edge_is_detected_as_significant():
    rng = np.random.default_rng(1)
    n = 400
    result = rng.integers(0, 3, size=n)
    # The model knows the true (skewed) outcome distribution; the market only sees a flat one —
    # a large, consistent edge that should clear both the n floor and FDR correction.
    df = pd.DataFrame(
        {
            "match_id": [f"m{i}" for i in range(n)],
            "result": result,
            "p_home": np.where(result == 0, 0.8, 0.1),
            "p_draw": np.where(result == 1, 0.8, 0.1),
            "p_away": np.where(result == 2, 0.8, 0.1),
            "mkt_home": 1 / 3,
            "mkt_draw": 1 / 3,
            "mkt_away": 1 / 3,
        }
    )
    features = df[["match_id"]].assign(referee="RefReal")

    out = slice_evidence(df, features, "gbdt")
    row = out[(out["slice_dim"] == "referee") & (out["slice_value"] == "RefReal")].iloc[0]

    assert row["edge_bits"] > 0
    assert bool(row["significant"])
    assert row["q_value"] < 0.01


def test_fdr_correction_keeps_false_positives_rare_under_the_null():
    rng = np.random.default_rng(2)
    n_slices, n_per_slice = 50, MIN_SLICE_N + 10
    frames = []
    for i in range(n_slices):
        result = rng.integers(0, 3, size=n_per_slice)
        # Model and market share the exact same (noisy, uninformative) probabilities per slice —
        # no real edge anywhere, so under a well-calibrated correction, few slices should still
        # be flagged significant purely by chance.
        noise = rng.normal(scale=0.02, size=(n_per_slice, 3))
        probs = np.clip(1 / 3 + noise, 0.05, 0.9)
        probs = probs / probs.sum(axis=1, keepdims=True)
        frames.append(
            pd.DataFrame(
                {
                    "match_id": [f"s{i}_m{j}" for j in range(n_per_slice)],
                    "result": result,
                    "p_home": probs[:, 0],
                    "p_draw": probs[:, 1],
                    "p_away": probs[:, 2],
                    "mkt_home": probs[:, 0],
                    "mkt_draw": probs[:, 1],
                    "mkt_away": probs[:, 2],
                    "league_code": f"L{i}",
                }
            )
        )
    predictions = pd.concat(frames, ignore_index=True)
    features = predictions[["match_id"]]

    out = slice_evidence(predictions, features, "gbdt")
    league_rows = out[out["slice_dim"] == "league_code"]

    assert len(league_rows) == n_slices
    assert league_rows["significant"].sum() <= 5


def test_fixture_slice_values_matches_the_same_fixed_buckets():
    row = pd.Series(
        {
            "league_code": "E0",
            "referee": "M. Oliver",
            "home_rest_days": 7,
            "away_rest_days": 2,
            "sv_missing_pct_diff": 0.2,
            "wx_precipitation": 5.0,
            "wx_wind_speed_10m": 10.0,
        }
    )
    values = fixture_slice_values(row)
    assert values == {
        "league_code": "E0",
        "referee": "M. Oliver",
        "rest_days_gap_bucket": "5+d",
        "squad_value_gap_bucket": "large_gap",
        "weather_bucket": "adverse",
    }


def test_nearest_checkpoint_never_uses_a_checkpoint_on_or_after_as_of():
    evidence = pd.DataFrame(
        {
            "model": ["gbdt", "gbdt", "gbdt"],
            "slice_dim": ["league_code"] * 3,
            "slice_value": ["E0"] * 3,
            "edge_bits": [0.01, 0.02, 0.03],
            "checkpoint_date": ["2023-01-01", "2023-06-01", "2024-01-01"],
        }
    )
    out = nearest_checkpoint(evidence, pd.Timestamp("2023-12-01"))
    assert out["checkpoint_date"].tolist() == ["2023-06-01"]


def test_select_trusted_model_picks_best_edge_even_if_negative():
    evidence = pd.DataFrame(
        {
            "model": ["gbdt", "dixon_coles"],
            "slice_dim": ["league_code", "league_code"],
            "slice_value": ["E0", "E0"],
            "edge_bits": [-0.02, -0.01],
            "checkpoint_date": ["2023-01-01", "2023-01-01"],
        }
    )
    best, matches = select_trusted_model(evidence, {"league_code": "E0"}, ["gbdt", "dixon_coles"], pd.Timestamp("2024-01-01"))
    assert best == "dixon_coles"
    assert len(matches) == 2


def test_select_trusted_model_none_when_no_evidence_at_all():
    evidence = pd.DataFrame(columns=["model", "slice_dim", "slice_value", "edge_bits", "checkpoint_date"])
    best, matches = select_trusted_model(evidence, {"league_code": "E0"}, ["gbdt"], pd.Timestamp("2024-01-01"))
    assert best is None
    assert matches.empty
