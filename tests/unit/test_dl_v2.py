"""Lightning sequence models (GRU + Transformer)."""

from __future__ import annotations

import numpy as np
import pytest

from pitch_edge.features.build import FeatureBuilder
from pitch_edge.models import TransformerSequenceModel, available_models
from pitch_edge.models.sequence import GRUSequenceModel

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def split(synthetic_league_matches):
    f = FeatureBuilder().build(synthetic_league_matches)
    cut = f["date"].quantile(0.75)
    return f[f["date"] < cut], f[f["date"] >= cut]


@pytest.mark.parametrize("cls", [GRUSequenceModel, TransformerSequenceModel])
def test_lightning_models_fit_predict_and_report(cls, split):
    train, test = split
    m = cls(epochs=3, seq_len=6, hidden=16, patience=2).fit(train)
    p = m.predict_proba(test)
    assert list(p.columns) == ["home", "draw", "away"] and np.allclose(p.sum(axis=1), 1.0, atol=1e-5)
    card = m.card()
    assert card["last_fit"]["epochs_run"] >= 1 and 0.5 <= card["last_fit"]["temperature"] <= 3.0
    assert np.isfinite(card["last_fit"]["val_log_loss"]) and card["last_fit"]["n_val"] >= 64
    assert "temperature scaling" in card["calibration"]


def test_transformer_is_registered():
    names = [m.name for m in available_models()]
    assert "transformer_sequence" in names


def test_validation_slice_is_the_most_recent(split):
    train, _ = split
    m = GRUSequenceModel(epochs=1, seq_len=4, hidden=8).fit(train)
    n_val = m.last_fit_["n_val"]
    assert n_val == max(int(len(train) * 0.15), 64)
