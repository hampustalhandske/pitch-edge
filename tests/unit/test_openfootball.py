from __future__ import annotations

import pytest
import responses

from pitch_edge.data.sources.openfootball import OpenFootballSource, _extract_score

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "m, expected",
    [
        ({"score": {"ft": [2, 1]}}, (2, 1)),
        ({"score": [0, 0]}, (0, 0)),
        ({"score1": 3, "score2": 1}, (3, 1)),
        ({"score": {"ht": [1, 0]}}, None),
        ({}, None),
    ],
)
def test_extract_score_variants(m, expected):
    assert _extract_score(m) == expected


@responses.activate
def test_rounds_layout_and_fixture_split(tmp_path):
    payload = {
        "name": "x",
        "rounds": [
            {
                "name": "1",
                "matches": [
                    {
                        "date": "2023-08-12",
                        "team1": {"name": "Arsenal FC"},
                        "team2": {"name": "Chelsea FC"},
                        "score1": 2,
                        "score2": 1,
                    }
                ],
            },
            {"name": "2", "matches": [{"date": "2099-08-19", "team1": "Chelsea FC", "team2": "Arsenal FC"}]},
        ],
    }
    responses.add(
        responses.GET,
        "https://raw.githubusercontent.com/openfootball/football.json/master/2023-24/en.1.json",
        json=payload,
    )
    src = OpenFootballSource(cache_dir=tmp_path)
    played = src.fetch_matches("en.1", [2023])
    assert len(played) == 1 and played.iloc[0]["home_team"] == "Arsenal FC"
    upcoming = src.fetch_upcoming_fixtures("en.1", 2023)
    assert len(upcoming) == 1 and upcoming.iloc[0]["home_goals"] != upcoming.iloc[0]["home_goals"]  # NaN
