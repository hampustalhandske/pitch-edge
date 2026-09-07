from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pitch_edge.features.build import FeatureBuilder, feature_columns

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def features(synthetic_league_matches):
    return FeatureBuilder().build(synthetic_league_matches)


def test_shapes_and_targets(features, synthetic_league_matches):
    assert len(features) == len(synthetic_league_matches)
    assert set(features["result"].unique()) <= {0, 1, 2}
    assert (features["total_goals"] == features["home_goals"] + features["away_goals"]).all()


def test_elo_is_pre_match_and_starts_at_initial(features):
    first = features.sort_values("date").iloc[0]
    assert first["elo_home"] == 1500.0 and first["elo_away"] == 1500.0
    # later matches have diverged ratings
    assert features["elo_diff"].abs().max() > 20


def test_rolling_form_has_no_lookahead(features, synthetic_league_matches):
    """A team's r5 goals-for before its second match must equal goals in its first match only."""
    m = synthetic_league_matches.sort_values("date")
    team = m.iloc[0]["home_team"]
    team_matches = m[(m["home_team"] == team) | (m["away_team"] == team)].sort_values("date")
    first, second = team_matches.iloc[0], team_matches.iloc[1]
    gf_first = first["home_goals"] if first["home_team"] == team else first["away_goals"]
    row = features[features["match_id"] == second["match_id"]].iloc[0]
    col = "h_gf_r5" if second["home_team"] == team else "a_gf_r5"
    assert row[col] == pytest.approx(gf_first)
    first_row = features[features["match_id"] == first["match_id"]].iloc[0]
    assert np.isnan(first_row["h_gf_r5" if first["home_team"] == team else "a_gf_r5"])


def test_market_features_use_early_odds_only(features):
    assert features["mkt_overround"].dropna().between(0.0, 0.15).all()
    base = feature_columns(features, include_market=False)
    assert not any(c.startswith("mkt_") for c in base)
    assert "mkt_home_p" in feature_columns(features, include_market=True)
    assert not any(c.startswith("PSC") for c in feature_columns(features, include_market=True))


def test_referee_and_rest_features_present(features):
    assert features["ref_home_bias"].notna().sum() > 0
    assert features["away_rest_days"].notna().sum() > 0
    assert features["away_fatigue_index"].notna().all()


def test_external_elo_overrides_internal(synthetic_league_matches):
    m = synthetic_league_matches.copy()
    m["home_elo"] = 1700.0
    m["away_elo"] = 1600.0
    f = FeatureBuilder().build(m)
    assert (f["elo_diff"] == 100.0).all()


def test_weather_and_venue_join(synthetic_league_matches):
    teams = sorted(set(synthetic_league_matches["home_team"]))
    venues = pd.DataFrame(
        {"team": teams, "lat": np.linspace(50, 56, len(teams)), "lon": np.linspace(-3, 1, len(teams))}
    )
    weather = pd.DataFrame({"match_id": synthetic_league_matches["match_id"].head(10), "wx_precipitation": 2.0})
    f = FeatureBuilder().build(synthetic_league_matches, weather=weather, venues=venues)
    assert f["away_travel_km"].notna().all() and f["away_travel_km"].max() > 100
    assert f["wx_precipitation"].notna().sum() == 10
