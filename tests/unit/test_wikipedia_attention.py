"""Wikipedia attention connector: title resolution, pageview parsing (mocked HTTP), leakage-free anomaly feature."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import responses

from pitch_edge.data.alt.wikipedia_attention import (
    PAGEVIEWS_URL,
    WikipediaAttentionSource,
    attention_anomaly,
    attention_features,
)

pytestmark = pytest.mark.unit


def test_title_override_and_search_candidates(tmp_path):
    src = WikipediaAttentionSource(cache_dir=tmp_path)
    assert src.resolve_title("Nott'm Forest") == "Nottingham Forest F.C."
    cands = src.search_candidates("Man United")
    assert cands[0].startswith("manchester united") and cands[-1] == "Man United football club"


@responses.activate
def test_search_skips_season_articles(tmp_path):
    responses.add(
        responses.GET,
        "https://en.wikipedia.org/w/api.php",
        json={"query": {"search": [{"title": "2010–11 Real Madrid CF season"}, {"title": "Real Madrid CF"}]}},
    )
    src = WikipediaAttentionSource(cache_dir=tmp_path)
    assert src.resolve_title("Real Madrid") == "Real Madrid CF"


@responses.activate
def test_fetch_daily_chunks_by_year_and_parses(tmp_path):
    def items(year, n):
        return {"items": [{"timestamp": f"{year}010{d}00", "views": 100 + d} for d in range(1, n + 1)]}

    for year in (2024, 2025):
        url = PAGEVIEWS_URL.format(
            project="en.wikipedia",
            article="Arsenal_F.C.",
            granularity="daily",
            start=f"{year}0101" + "00" if year == 2025 else "2024122500",
            end=f"{year}010500" if year == 2025 else "2024123100",
        )
        responses.add(responses.GET, url, json=items(year, 3))
    src = WikipediaAttentionSource(cache_dir=tmp_path)
    df = src.fetch_daily("Arsenal F.C.", "2024-12-25", "2025-01-05")
    assert len(df) == 6 and set(df["article"]) == {"Arsenal F.C."} and df["views"].min() == 101


def test_attention_anomaly_flags_spike_and_is_leak_free():
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    views = np.full(60, 1000.0)
    views[50] = 20000.0  # spike on day 50
    pv = pd.DataFrame({"team": "Alpha", "article": "Alpha FC", "date": dates, "views": views})
    an = attention_anomaly(pv).set_index("date")
    assert an.loc[dates[51], "pv_z"] > 5  # the day AFTER the spike sees it (lead_days = 1)
    assert abs(an.loc[dates[50], "pv_z"]) < 1  # the spike day itself does not use its own views
    m = pd.DataFrame(
        {
            "match_id": ["m1", "m2"],
            "date": [dates[51], dates[50]],
            "home_team": ["Alpha", "Alpha"],
            "away_team": ["Beta", "Beta"],
        }
    )
    f = attention_features(m, pv).set_index("match_id")
    assert f.loc["m1", "pv_home_z"] > 5 and abs(f.loc["m2", "pv_home_z"]) < 1
    assert np.isnan(f.loc["m1", "pv_away_z"])  # no pageviews for Beta
