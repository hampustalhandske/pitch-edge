from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import responses

from pitch_edge.data.alt.audio import extract_momentum_features
from pitch_edge.data.alt.news import NewsScanner, sentiment_velocity
from pitch_edge.data.alt.opensky import OpenSkySource
from pitch_edge.data.alt.referee import referee_features
from pitch_edge.data.alt.travel import fatigue_index, haversine_km, rest_and_congestion, travel_distance
from pitch_edge.data.alt.venues import VenueGeocoder
from pitch_edge.data.alt.weather import OpenMeteoWeather

pytestmark = pytest.mark.unit


# ------------------------------------------------------------------- travel
def test_haversine_london_manchester():
    assert 255 < haversine_km(51.5549, -0.1084, 53.4631, -2.2913) < 275


def test_rest_and_congestion_uses_only_prior_matches():
    m = pd.DataFrame(
        {
            "match_id": ["a", "b", "c"],
            "date": pd.to_datetime(["2024-01-01", "2024-01-04", "2024-01-12"]),
            "home_team": ["X", "Y", "X"],
            "away_team": ["Y", "X", "Z"],
        }
    )
    r = rest_and_congestion(m).set_index("match_id")
    assert np.isnan(r.loc["a", "home_rest_days"])
    assert r.loc["b", "away_rest_days"] == 3  # X played 3 days earlier
    assert r.loc["c", "home_matches_last_14d"] == 2
    assert r.loc["c", "home_rest_days"] == 8


def test_travel_distance_and_fatigue():
    m = pd.DataFrame({"match_id": ["a"], "home_team": ["X"], "away_team": ["Y"]})
    coords = pd.DataFrame({"team": ["X", "Y"], "lat": [51.5, 53.5], "lon": [-0.1, -2.3]})
    d = travel_distance(m, coords)
    assert 200 < d.loc[0, "away_travel_km"] < 300
    f = pd.DataFrame({"away_rest_days": [2.0, 10.0], "away_matches_last_14d": [4, 1], "away_travel_km": [800.0, 50.0]})
    fi = fatigue_index(f)
    assert fi.iloc[0] > fi.iloc[1]


# ------------------------------------------------------------------ referee
def test_referee_features_are_prior_only(synthetic_league_matches):
    r = referee_features(synthetic_league_matches, min_prior_matches=3)
    first_by_ref = synthetic_league_matches.sort_values("date").groupby("referee").head(1)
    assert r.set_index("match_id").loc[first_by_ref["match_id"], "ref_cards_per_game"].isna().all()
    assert r["ref_home_bias"].dropna().between(-0.5, 0.5).all()


# -------------------------------------------------------------------- news
def test_news_scoring_links_teams_and_sentiment(tmp_path):
    scanner = NewsScanner(cache_dir=tmp_path)
    items = pd.DataFrame(
        {
            "item_id": ["1", "2"],
            "feed": ["t", "t"],
            "published_at": pd.to_datetime(["2024-01-01 10:00", "2024-01-02 10:00"]),
            "title": ["Arsenal star ruled out with hamstring injury", "Man City cruise to brilliant win"],
            "summary": ["", ""],
            "link": ["", ""],
        }
    )
    scored = scanner.score(items, ["Arsenal", "Man City", "Chelsea"])
    assert scored.loc[0, "team"] == "Arsenal" and scored.loc[0, "is_injury_news"]
    assert scored.loc[1, "sentiment"] > scored.loc[0, "sentiment"]
    vel = sentiment_velocity(scored, window_hours=48)
    assert set(vel["team"]) == {"Arsenal", "Man City"}


@responses.activate
def test_feed_fetch_parses_rss(tmp_path):
    rss = b"""<?xml version="1.0"?><rss version="2.0"><channel><title>t</title>
    <item><title>Liverpool team news</title><link>http://x/1</link><guid>g1</guid>
    <description>Salah <b>benched</b></description><pubDate>Mon, 01 Jan 2024 10:00:00 GMT</pubDate></item>
    </channel></rss>"""
    responses.add(responses.GET, "https://feed.test/rss", body=rss, status=200)
    df = NewsScanner(cache_dir=tmp_path).fetch_feed("t", "https://feed.test/rss")
    assert len(df) == 1 and df.loc[0, "summary"] == "Salah  benched"


@responses.activate
def test_bluesky_403_degrades_to_empty(tmp_path):
    responses.add(responses.GET, "https://public.api.bsky.app/xrpc/app.bsky.feed.searchPosts", status=403)
    assert NewsScanner(cache_dir=tmp_path).fetch_bluesky("x").empty


# Polymarket live-snapshot connector was removed (see `data/alt/pmxt_archive.py`); its
# coverage is tested in `tests/unit/test_pmxt_archive.py`.


@responses.activate
def test_opensky_parses_states(tmp_path):
    payload = {"time": 1700000000, "states": [["abc123", "SAS123  ", "Sweden"] + [None] * 14]}
    responses.add(responses.GET, "https://opensky-network.org/api/states/all", json=payload, status=200)
    df = OpenSkySource(cache_dir=tmp_path).fetch_states_in_bbox(51, -1, 52, 0)
    assert df.loc[0, "callsign"] == "SAS123"
    assert OpenSkySource.implemented_as_feature is False


# ------------------------------------------------------------------ weather
@responses.activate
def test_open_meteo_weather_for_matches(tmp_path):
    hours = pd.date_range("2024-01-06", periods=48, freq="h")
    payload = {
        "hourly": {
            "time": [t.strftime("%Y-%m-%dT%H:%M") for t in hours],
            "temperature_2m": list(range(48)),
            "precipitation": [0.1] * 48,
            "wind_speed_10m": [5.0] * 48,
            "relative_humidity_2m": [80] * 48,
        }
    }
    responses.add(responses.GET, "https://archive-api.open-meteo.com/v1/archive", json=payload, status=200)
    matches = pd.DataFrame(
        {"match_id": ["m"], "date": [pd.Timestamp("2024-01-06")], "home_team": ["X"], "kickoff_time": ["15:00"]}
    )
    coords = pd.DataFrame({"team": ["X"], "lat": [51.5], "lon": [-0.1]})
    wx = OpenMeteoWeather(cache_dir=tmp_path).weather_for_matches(matches, coords)
    assert wx.loc[0, "wx_temperature_2m"] == 15


# ------------------------------------------------------------------- venues
@responses.activate
def test_venue_geocoder_wikidata_then_bundled(tmp_path):
    payload = {
        "results": {
            "bindings": [
                {
                    "clubLabel": {"value": "Brentford F.C."},
                    "venueLabel": {"value": "Gtech"},
                    "coord": {"value": "Point(-0.2886 51.4907)"},
                }
            ]
        }
    }
    responses.add(responses.GET, "https://query.wikidata.org/sparql", json=payload, status=200)
    g = VenueGeocoder(cache_dir=tmp_path)
    out = g.build_team_coordinates(["Brentford", "Arsenal", "Unknown Town"], ["England"]).set_index("team")
    assert out.loc["Brentford", "geo_source"] == "wikidata"
    assert out.loc["Arsenal", "geo_source"] == "bundled"
    assert "Unknown Town" not in out.index


# -------------------------------------------------------------------- audio
def test_audio_momentum_detects_roar_burst():
    sr = 22050
    t = np.linspace(0, 6, 6 * sr, endpoint=False)
    y = 0.02 * np.random.default_rng(0).normal(size=t.size).astype(np.float32)
    burst = (t > 3.0) & (t < 3.4)
    y[burst] += (0.8 * np.sin(2 * np.pi * 300 * t[burst])).astype(np.float32)
    feats = extract_momentum_features(y, sr=sr)
    assert feats.roar_spikes.sum() >= 1
    peak_t = feats.times_s[np.argmax(feats.momentum_index())]
    assert 2.5 < peak_t < 4.0
    assert feats.band_shape.shape[0] == 24
