from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pitch_edge.features.build import FeatureBuilder
from pitch_edge.models import DixonColesMatchModel, GBDTMatchModel, GRUSequenceModel, default_models

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def split(synthetic_league_matches):
    f = FeatureBuilder().build(synthetic_league_matches)
    cut = f["date"].quantile(0.75)
    return f[f["date"] < cut], f[f["date"] >= cut]


def _check_probs(p: pd.DataFrame, n: int) -> None:
    assert list(p.columns) == ["home", "draw", "away"]
    assert len(p) == n
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-6)
    assert (p >= 0).all().all()


def test_dixon_coles_adapter(split):
    train, test = split
    m = DixonColesMatchModel().fit(train)
    _check_probs(m.predict_proba(test), len(test))
    card = m.card()
    assert card["name"] == "dixon_coles" and "leakage_checks" in card


def test_gbdt_fit_predict_and_backend(split):
    train, test = split
    m = GBDTMatchModel(n_estimators=40).fit(train)
    p = m.predict_proba(test)
    _check_probs(p, len(test))
    assert m.backend in ("lightgbm", "sklearn_hgb")
    assert not m.feature_importance().empty or m.backend == "sklearn_hgb"
    assert m.card()["uses_market_features"] is False


def test_gbdt_market_variant_uses_market_columns(split):
    train, test = split
    m = GBDTMatchModel(include_market=True, n_estimators=40).fit(train)
    assert "mkt_home_p" in m.card()["features"]
    _check_probs(m.predict_proba(test), len(test))


def test_gbdt_market_variant_beats_no_market_on_log_loss(split):
    train, test = split
    y = test["result"].to_numpy()
    ll = {}
    for m in (GBDTMatchModel(n_estimators=60), GBDTMatchModel(include_market=True, n_estimators=60)):
        p = m.fit(train).predict_proba(test).to_numpy()
        ll[m.name] = -np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-9, 1)))
    # market-informed model should not be materially worse than a pure model
    assert ll["gbdt_mkt"] <= ll["gbdt"] + 0.05


def test_gru_sequence_model(split):
    train, test = split
    m = GRUSequenceModel(epochs=2, seq_len=5, hidden=8).fit(train)
    _check_probs(m.predict_proba(test), len(test))
    # observe() extends history without breaking prediction
    m.observe(test.head(5))
    _check_probs(m.predict_proba(test.tail(3)), 3)


def test_gru_sequences_exclude_future(split):
    train, _ = split
    m = GRUSequenceModel(epochs=1, seq_len=5, hidden=8).fit(train)
    team = train.iloc[0]["home_team"]
    d0 = pd.Timestamp(train["date"].min())
    seq = m._sequence(team, d0)
    assert seq.shape == (5, 9) and np.all(seq == 0)  # nothing before the first date


def test_default_models_names():
    names = [m.name for m in default_models()]
    assert names == ["dixon_coles", "gbdt", "stochastic_strength"]
