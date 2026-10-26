"""Deterministic ask-agent data assembly: candidate fixtures, model predictions, market edges."""

from __future__ import annotations

import pandas as pd
import pytest

from pitch_edge.agents.qa_data import market_edges, resolve_fixture, run_models, top_bets_candidates
from pitch_edge.models.base import MatchModel

pytestmark = pytest.mark.unit


class _FixedModel(MatchModel):
    """Predicts the same fixed probability triple for every row, regardless of `fit`."""

    name = "fixed"

    def __init__(self, home: float, draw: float, away: float):
        self._p = (home, draw, away)
        self.fit_calls = 0

    def fit(self, train):
        self.fit_calls += 1
        return self

    def predict_proba(self, X):
        h, d, a = self._p
        return pd.DataFrame({"home": h, "draw": d, "away": a}, index=X.index)


def _features_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "match_id": ["m1", "m2", "m3"],
            "date": ["2024-06-01", "2024-06-05", "2024-07-20"],
            "home_team": ["Arsenal", "Liverpool", "Chelsea"],
            "away_team": ["Chelsea", "Arsenal", "Liverpool"],
            "PSH": [2.0, None, 2.5],
            "PSD": [3.2, None, 3.0],
            "PSA": [4.0, None, 2.8],
        }
    )


def test_top_bets_candidates_requires_real_odds_and_the_window():
    features = _features_frame()
    out = top_bets_candidates(features, pd.Timestamp("2024-06-01"), window_days=14)
    assert out["match_id"].tolist() == ["m1"]  # m2 has no real odds, m3 is outside the window


def test_resolve_fixture_fuzzy_matches_either_side():
    features = _features_frame()
    row = resolve_fixture(features, pd.Timestamp("2024-06-01"), "arsenal", "chelsea")
    assert row is not None and row["match_id"].iloc[0] == "m1"


def test_resolve_fixture_returns_none_for_unknown_teams():
    features = _features_frame()
    assert resolve_fixture(features, pd.Timestamp("2024-06-01"), "Real Madrid", "Barcelona") is None


def test_run_models_fits_strictly_before_as_of():
    features = _features_frame()
    fixtures = features.iloc[[0]]
    model = _FixedModel(0.5, 0.3, 0.2)

    preds = run_models(features, fixtures, pd.Timestamp("2024-06-01"), [model])

    assert model.fit_calls == 1
    assert preds.iloc[0][["p_home", "p_draw", "p_away"]].tolist() == [0.5, 0.3, 0.2]


def test_market_edges_computes_no_vig_edge_per_outcome():
    fixtures = _features_frame().iloc[[0]]
    predictions = pd.DataFrame([{"match_id": "m1", "model": "fixed", "p_home": 0.6, "p_draw": 0.25, "p_away": 0.15}])

    out = market_edges(fixtures, predictions)

    home_row = out[out["outcome"] == "home"].iloc[0]
    assert home_row["model_probability"] == pytest.approx(0.6)
    assert home_row["decimal_odds"] == pytest.approx(2.0)
    assert home_row["edge"] == pytest.approx(home_row["model_probability"] - home_row["market_probability"])


# --------------------------------------------------------------------------- evidence status per candidate


def _research(rung: str = "L2", floor: str = "L4") -> dict:
    return {
        "display_floor": floor,
        "validated_edge_exists": False,
        "families": [{"family": "fam-a", "hypotheses": [{"id": "fam-a-x-v1", "rung": rung}]}],
    }


def test_every_ask_model_maps_to_its_registered_r7_bet_rule_hypothesis():
    from pathlib import Path

    from pitch_edge.agents.qa_data import ASK_MODEL_HYPOTHESES
    from pitch_edge.models import default_models

    registry = Path(__file__).resolve().parents[2] / "research" / "registry"
    names = [m.name for m in default_models()]
    assert set(ASK_MODEL_HYPOTHESES) == set(names)
    for name in names:
        hypothesis_id = ASK_MODEL_HYPOTHESES[name]
        assert hypothesis_id == f"outcome-models-{name.replace('_', '-')}-bet-rule-v1"
        assert (registry / f"{hypothesis_id}.yaml").is_file()


def test_a_mapped_ask_model_carries_its_hypothesis_rung_and_stays_below_the_floor():
    from pitch_edge.agents.qa_data import candidate_evidence

    research = {
        "display_floor": "L4",
        "families": [
            {
                "family": "outcome-models",
                "hypotheses": [{"id": "outcome-models-stochastic-strength-bet-rule-v1", "rung": "L1"}],
            }
        ],
    }
    ev = candidate_evidence("stochastic_strength", research)
    assert ev["hypothesis_id"] == "outcome-models-stochastic-strength-bet-rule-v1"
    assert ev["rung"] == "L1"
    assert ev["validated_edge"] is False
    assert ev["status"] == "L1, below display floor L4"
    assert ev["label"] == "model-vs-market disagreement, not a validated edge"


def test_models_without_a_registered_hypothesis_are_below_the_floor():
    from pitch_edge.agents.qa_data import candidate_evidence

    for model in ("unregistered_model", None):
        ev = candidate_evidence(model, _research())
        assert ev["hypothesis_id"] is None
        assert ev["rung"] == "unregistered"
        assert ev["display_floor"] == "L4"
        assert ev["validated_edge"] is False
        assert ev["status"] == "no registered evidence / below floor"
        assert ev["label"] == "model-vs-market disagreement, not a validated edge"


@pytest.mark.parametrize(("rung", "floor", "validated"), [("L3", "L4", False), ("L4", "L4", True), ("L4", "L5", False)])
def test_a_registered_model_is_a_validated_edge_only_at_or_above_the_floor(rung, floor, validated):
    from pitch_edge.agents.qa_data import candidate_evidence

    ev = candidate_evidence("gbdt", _research(rung, floor), model_hypotheses={"gbdt": "fam-a-x-v1"})
    assert (ev["hypothesis_id"], ev["rung"], ev["validated_edge"]) == ("fam-a-x-v1", rung, validated)
    if not validated:
        assert ev["label"] == "model-vs-market disagreement, not a validated edge"


def test_annotate_candidates_adds_evidence_without_changing_the_candidates():
    from pitch_edge.agents.qa_data import annotate_candidates

    candidates = [{"match_id": "m1", "home_team": "Arsenal"}, {"match_id": "m2", "home_team": "Liverpool"}]
    out = annotate_candidates(candidates, {"m1": "gbdt", "m2": None}, _research())
    assert [c["match_id"] for c in out] == ["m1", "m2"]
    assert out[0]["home_team"] == "Arsenal"
    assert out[0]["evidence"]["model"] == "gbdt"
    assert out[1]["evidence"]["model"] is None
    assert out[1]["evidence"]["validated_edge"] is False
    assert "evidence" not in candidates[0]  # inputs untouched
