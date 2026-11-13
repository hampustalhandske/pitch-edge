from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pitch_edge.models.dixon_coles import DixonColesModel
from pitch_edge.models.poisson import DixonColesMatchModel

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


# ---------------------------------------------------------------- fallbacks are counted, never silent
NO_FALLBACKS = {
    "league_fit_failed": 0,
    "global_fit": 0,
    "other_league_model": 0,
    "independent_poisson": 0,
    "uniform": 0,
    "start_parameters": 0,
}


def _two_leagues(synthetic_league_matches: pd.DataFrame) -> pd.DataFrame:
    """The synthetic league (`SYN`) plus a copy with renamed teams as a second league (`SY2`)."""
    second = synthetic_league_matches.assign(
        league_code="SY2",
        home_team="B_" + synthetic_league_matches["home_team"],
        away_team="B_" + synthetic_league_matches["away_team"],
    )
    return pd.concat([synthetic_league_matches, second], ignore_index=True)


def _failing_fit(monkeypatch, should_fail):
    """Make `DixonColesModel.fit` raise whenever `should_fail(matches)` is true."""
    real_fit = DixonColesModel.fit

    def fit(self, matches):
        if should_fail(matches):
            raise FloatingPointError("planted fit failure")
        return real_fit(self, matches)

    monkeypatch.setattr(DixonColesModel, "fit", fit)


def test_a_normal_fit_reports_no_fallbacks(small_match_history, synthetic_league_matches):
    model = DixonColesModel().fit(small_match_history)
    model.predict_score_matrix("Alpha", "Beta")
    assert model.fallback_counts() == {"independent_poisson": 0, "uniform": 0, "start_parameters": 0}
    adapter = DixonColesMatchModel().fit(_two_leagues(synthetic_league_matches))
    adapter.predict_proba(_two_leagues(synthetic_league_matches).tail(5))
    assert adapter.card()["fallbacks"] == NO_FALLBACKS


def test_a_degenerate_tau_falls_back_to_independent_poisson_and_is_counted(small_match_history):
    model = DixonColesModel().fit(small_match_history)
    model.rho_ = float("nan")  # tau-corrected matrix has a NaN total
    matrix = model.predict_score_matrix("Alpha", "Beta")
    assert matrix.sum() == pytest.approx(1.0)
    assert model.fallback_counts() == {"independent_poisson": 1, "uniform": 0, "start_parameters": 0}


def test_degenerate_lambdas_fall_back_to_a_uniform_matrix_and_are_counted(small_match_history):
    model = DixonColesModel().fit(small_match_history)
    model.home_advantage_ = float("nan")  # both the tau and the independent-Poisson matrices are NaN
    matrix = model.predict_score_matrix("Alpha", "Beta")
    assert matrix == pytest.approx(np.full_like(matrix, 1.0 / matrix.size))
    assert model.fallback_counts() == {"independent_poisson": 0, "uniform": 1, "start_parameters": 0}


def test_a_failed_league_fit_and_predictions_for_that_league_are_counted(synthetic_league_matches, monkeypatch):
    _failing_fit(monkeypatch, lambda m: m["home_team"].str.startswith("B_").any())
    both = _two_leagues(synthetic_league_matches)
    model = DixonColesMatchModel().fit(both)
    assert model.card()["leagues_fitted"] == ["SYN"]
    model.predict_proba(both[both["league_code"] == "SY2"].head(3))
    assert model.card()["fallbacks"] == {**NO_FALLBACKS, "league_fit_failed": 1, "other_league_model": 3}


def test_the_global_fallback_fit_is_counted(synthetic_league_matches, monkeypatch):
    _failing_fit(monkeypatch, lambda m: m["home_team"].nunique() < 24)  # every single-league fit fails
    model = DixonColesMatchModel().fit(_two_leagues(synthetic_league_matches))
    assert model.card()["leagues_fitted"] == ["__all__"]
    assert model.card()["fallbacks"] == {**NO_FALLBACKS, "league_fit_failed": 2, "global_fit": 1}


def test_matrix_fallbacks_of_the_league_models_are_kept_across_refits(synthetic_league_matches):
    both = _two_leagues(synthetic_league_matches)
    model = DixonColesMatchModel().fit(both)
    model._models["SYN"].rho_ = float("nan")
    model.predict_proba(both[both["league_code"] == "SYN"].head(2))
    assert model.card()["fallbacks"]["independent_poisson"] == 2
    model.fit(both)  # a refit replaces the league models; the lifetime count stays
    assert model.card()["fallbacks"]["independent_poisson"] == 2


def test_a_non_finite_optimiser_result_keeps_the_start_parameters_and_is_counted(small_match_history, monkeypatch):
    import pitch_edge.models.dixon_coles as dc

    real = dc.minimize

    def nan_result(*args, **kwargs):
        out = real(*args, **kwargs)
        out.x = np.full_like(out.x, np.nan)
        return out

    monkeypatch.setattr(dc, "minimize", nan_result)
    model = DixonColesModel().fit(small_match_history)
    assert model.fallback_counts() == {"independent_poisson": 0, "uniform": 0, "start_parameters": 1}
    assert model.rho_ == pytest.approx(-0.01)  # the starting rho
