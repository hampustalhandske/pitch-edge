"""Degenerate inputs must degrade to sane probabilities, never NaN, all the way through the engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pitch_edge.backtest.engine import WalkForwardBacktester, WalkForwardConfig
from pitch_edge.models.base import MatchModel, normalise_probs, to_frame
from pitch_edge.models.dixon_coles import DixonColesModel
from pitch_edge.models.poisson import DixonColesMatchModel

pytestmark = pytest.mark.unit


def test_dixon_coles_extreme_scorelines_stay_finite():
    """A tiny, lopsided league (every match a 9-0) used to push lambdas to inf and the tau matrix to NaN."""
    rows = []
    for i, d in enumerate(pd.date_range("2024-01-01", periods=12, freq="7D")):
        rows.append({"date": d, "home_team": "Giant", "away_team": f"Minnow{i % 3}", "home_goals": 9, "away_goals": 0})
        rows.append({"date": d, "home_team": f"Minnow{i % 3}", "away_team": "Giant", "home_goals": 0, "away_goals": 8})
    model = DixonColesModel().fit(pd.DataFrame(rows))
    assert np.isfinite(list(model.attack_.values())).all() and abs(model.rho_) <= 0.3
    probs = model.predict_outcome_probabilities("Giant", "Minnow0")
    assert np.isfinite(list(probs.values())).all() and abs(sum(probs.values()) - 1) < 1e-6
    assert probs["home"] > 0.9


def test_score_matrix_guard_on_degenerate_parameters():
    m = DixonColesModel()
    m.fitted_ = True
    m.attack_ = {"A": 50.0}  # exp(50) overflows the tau correction
    m.defence_ = {"A": 0.0, "B": -50.0}
    m.home_advantage_ = 0.0
    m.rho_ = -0.3
    matrix = m.predict_score_matrix("A", "B")
    assert np.isfinite(matrix).all() and abs(matrix.sum() - 1) < 1e-9


class _NaNModel(MatchModel):
    name = "nan_model"

    def fit(self, train):
        return self

    def predict_proba(self, X):
        arr = np.full((len(X), 3), 1 / 3)
        arr[0] = np.nan
        return (
            to_frame(arr, X.index)
            if len(X) == 0
            else pd.DataFrame(arr, columns=["home", "draw", "away"], index=X.index)
        )


def test_engine_replaces_non_finite_probabilities(synthetic_league_matches):
    from pitch_edge.features.build import FeatureBuilder

    f = FeatureBuilder().build(synthetic_league_matches)
    res = WalkForwardBacktester(
        WalkForwardConfig(min_train_matches=300, retrain_every_days=120, calibration_min_rows=50)
    ).run(f, _NaNModel())
    assert np.isfinite(res.predictions[["p_home", "p_draw", "p_away"]].to_numpy()).all()


def test_per_league_adapter_survives_a_bad_league(synthetic_league_matches):
    f = synthetic_league_matches.copy()
    bad = f.head(30).copy()
    bad["league_code"] = "BAD"
    bad["home_goals"] = 12.0
    bad["away_goals"] = 0.0
    both = pd.concat([f, bad], ignore_index=True).assign(result=0)
    model = DixonColesMatchModel(min_league_rows=10).fit(both)
    p = model.predict_proba(both.tail(5))
    assert np.isfinite(p.to_numpy()).all()


def test_normalise_probs_replaces_non_finite_rows_with_uniform(caplog):
    """`to_frame` (used by every MatchModel's predict_proba: GBDT, Dixon-Coles, Transformer, GRU) must
    never let a NaN/inf row reach a live signal or the dashboard — the backtester has its own guard
    (test_engine_replaces_non_finite_probabilities above), but the live `signals`/`agentic-signals`
    path calls `predict_proba` directly, with no engine in between."""
    arr = np.array([[0.2, 0.5, 0.3], [np.nan, 0.5, 0.3], [1.0, np.inf, 0.0]])
    with caplog.at_level("WARNING"):
        out = normalise_probs(arr)
    assert np.isfinite(out).all()
    assert np.allclose(out.sum(axis=1), 1.0)
    assert np.allclose(out[0], [0.2, 0.5, 0.3])  # a healthy row is untouched (beyond renormalizing)
    assert np.allclose(out[1], [1 / 3, 1 / 3, 1 / 3])  # NaN row -> uniform
    assert np.allclose(out[2], [1 / 3, 1 / 3, 1 / 3])  # inf row -> uniform
    assert "non-finite" in caplog.text


def test_normalise_probs_all_finite_input_unaffected():
    arr = np.array([[1.0, 2.0, 3.0]])
    out = normalise_probs(arr)
    assert np.isfinite(out).all()
    assert np.allclose(out, [[1 / 6, 2 / 6, 3 / 6]])
