"""Point-in-time properties of `StochasticStrengthModel`: a match's predicted probabilities depend only on the fitted
parameters and that match's own row — not on which other matches share the prediction batch, how many calls came
before, or what the walk-forward frame contains after the match."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pitch_edge.backtest.engine import WalkForwardBacktester, WalkForwardConfig
from pitch_edge.features.build import FeatureBuilder
from pitch_edge.models import StochasticStrengthModel

pytestmark = pytest.mark.unit

PROB_COLS = ["raw_home", "raw_draw", "raw_away", "p_home", "p_draw", "p_away"]


@pytest.fixture(scope="module")
def features(synthetic_league_matches) -> pd.DataFrame:
    f = FeatureBuilder().build(synthetic_league_matches)
    f["date"] = pd.to_datetime(f["date"])
    return f.sort_values(["date", "match_id"]).reset_index(drop=True)


@pytest.fixture(scope="module")
def split(features):
    cut = features["date"].quantile(0.75)
    return features[features["date"] < cut], features[features["date"] >= cut]


def test_prediction_of_a_match_does_not_depend_on_the_rest_of_the_batch(split):
    train, test = split
    model = StochasticStrengthModel().fit(train)
    whole = model.predict_proba(test)
    head = StochasticStrengthModel().fit(train).predict_proba(test.head(7))
    assert np.abs(whole.loc[head.index].to_numpy() - head.to_numpy()).max() < 1e-9


def test_repeated_predictions_are_identical(split):
    train, test = split
    model = StochasticStrengthModel().fit(train)
    first = model.predict_proba(test).to_numpy()
    second = model.predict_proba(test).to_numpy()
    assert np.abs(first - second).max() < 1e-9


def test_missing_elo_in_a_later_row_does_not_move_earlier_predictions(split):
    train, test = split
    model = StochasticStrengthModel().fit(train)
    clean = model.predict_proba(test)
    gapped = test.copy()
    gapped.loc[gapped.index[-5:], ["elo_home", "elo_away"]] = np.nan
    filled = model.predict_proba(gapped)
    keep = test.index[:-5]
    assert np.abs(clean.loc[keep].to_numpy() - filled.loc[keep].to_numpy()).max() < 1e-9
    assert np.isfinite(filled.to_numpy()).all()


def _config() -> WalkForwardConfig:
    return WalkForwardConfig(
        min_train_matches=200, retrain_every_days=14, calibrate=True, calibration_min_rows=100, strategies=[]
    )


def test_walk_forward_predictions_are_truncation_invariant(features):
    dates = features["date"].drop_duplicates().sort_values().reset_index(drop=True)
    cut = dates.iloc[int(len(dates) * 0.7)]
    # the cut lands inside a fold whose full-frame test set continues after it
    assert ((features["date"] > cut) & (features["date"] < cut + pd.Timedelta(days=14))).any()

    full = WalkForwardBacktester(_config()).run(features, StochasticStrengthModel()).predictions
    trunc = features[features["date"] <= cut].reset_index(drop=True)
    part = WalkForwardBacktester(_config()).run(trunc, StochasticStrengthModel()).predictions

    a = full.set_index("match_id").loc[part["match_id"]][PROB_COLS].to_numpy(dtype=float)
    b = part.set_index("match_id")[PROB_COLS].to_numpy(dtype=float)
    assert len(b) > 100
    assert np.abs(a - b).max() < 1e-9
