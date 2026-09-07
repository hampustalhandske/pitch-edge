from __future__ import annotations

import pandas as pd
import pytest

from pitch_edge.models.dixon_coles import DixonColesModel

pytestmark = pytest.mark.unit


def test_fit_requires_columns():
    model = DixonColesModel()
    with pytest.raises(ValueError, match="missing columns"):
        model.fit(pd.DataFrame({"home_team": ["A"], "away_team": ["B"]}))


def test_fit_requires_nonempty():
    model = DixonColesModel()
    empty = pd.DataFrame(columns=["home_team", "away_team", "home_goals", "away_goals", "date"])
    with pytest.raises(ValueError, match="empty"):
        model.fit(empty)


def test_predict_before_fit_raises():
    model = DixonColesModel()
    with pytest.raises(RuntimeError, match="fit"):
        model.predict_outcome_probabilities("A", "B")


def test_fit_and_predict_probabilities_sum_to_one(small_match_history):
    model = DixonColesModel().fit(small_match_history)
    probs = model.predict_outcome_probabilities("Alpha", "Beta")
    assert set(probs) == {"home", "draw", "away"}
    assert probs["home"] >= 0 and probs["draw"] >= 0 and probs["away"] >= 0
    assert pytest.approx(sum(probs.values()), abs=1e-6) == 1.0


def test_score_matrix_sums_to_one(small_match_history):
    model = DixonColesModel().fit(small_match_history)
    matrix = model.predict_score_matrix("Alpha", "Beta")
    assert matrix.shape == (model.max_goals + 1, model.max_goals + 1)
    assert pytest.approx(matrix.sum(), abs=1e-6) == 1.0
    assert (matrix >= 0).all()


def test_home_advantage_recovered_from_synthetic_data(synthetic_league_matches):
    model = DixonColesModel().fit(synthetic_league_matches)
    # Home advantage was baked into the synthetic data generator (+0.25); the
    # fitted coefficient should be positive and in a plausible ballpark.
    assert model.home_advantage_ > 0


def test_stronger_team_favoured():
    """A team that always wins big at home should be predicted as strong favourite
    against a team that always loses big, regardless of home/away in a neutral test."""
    dates = pd.date_range("2023-01-01", periods=20, freq="7D")
    rows = []
    for d in dates:
        rows.append({"date": d, "home_team": "Strong", "away_team": "Weak", "home_goals": 4, "away_goals": 0})
        rows.append({"date": d, "home_team": "Weak", "away_team": "Strong", "home_goals": 0, "away_goals": 4})
    df = pd.DataFrame(rows)
    model = DixonColesModel().fit(df)
    probs = model.predict_outcome_probabilities("Strong", "Weak")
    assert probs["home"] > probs["away"]
    assert probs["home"] > 0.5


def test_unknown_team_falls_back_to_average_strength(small_match_history):
    model = DixonColesModel().fit(small_match_history)
    # Unknown teams default to attack/defence 0.0 rather than raising.
    probs = model.predict_outcome_probabilities("NeverSeenTeam", "Alpha")
    assert pytest.approx(sum(probs.values()), abs=1e-6) == 1.0
