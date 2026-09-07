from __future__ import annotations

import pandas as pd
import pandera.errors
import pytest

from pitch_edge.data.contracts import validate_matches, validate_odds_long
from pitch_edge.data.sources.base import REQUIRED_MATCH_COLUMNS, MatchDataSource

pytestmark = pytest.mark.unit


def _valid() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "match_id": ["1"],
            "date": [pd.Timestamp("2023-01-01")],
            "league": ["L"],
            "season": ["2023/24"],
            "home_team": ["A"],
            "away_team": ["B"],
            "home_goals": [1],
            "away_goals": [0],
            "source": ["t"],
        }
    )


def test_valid_frame_passes_and_coerces_goals_to_float():
    out = validate_matches(_valid())
    assert out["home_goals"].dtype.kind == "f"


def test_negative_goals_rejected():
    bad = _valid()
    bad.loc[0, "home_goals"] = -1
    with pytest.raises(pandera.errors.SchemaErrors):
        validate_matches(bad)


def test_duplicate_match_id_rejected():
    bad = pd.concat([_valid(), _valid()])
    with pytest.raises(pandera.errors.SchemaErrors):
        validate_matches(bad)


def test_odds_long_price_must_exceed_one():
    bad = pd.DataFrame(
        {
            "match_id": ["1"],
            "bookmaker": ["PS"],
            "market": ["1x2"],
            "side": ["home"],
            "price": [0.9],
            "is_closing": [False],
            "snapshot_ts": [pd.Timestamp("2023-01-01")],
        }
    )
    with pytest.raises(pandera.errors.SchemaErrors):
        validate_odds_long(bad)


class _Dummy(MatchDataSource):
    name = "dummy"

    def fetch_matches(self, **kwargs):
        return pd.DataFrame()


def test_source_validate_missing_columns_and_null_goals():
    with pytest.raises(ValueError, match="missing required columns"):
        _Dummy().validate(_valid().drop(columns=["home_goals"]))
    bad = _valid()
    bad.loc[0, "away_goals"] = None
    with pytest.raises(ValueError, match="null goal"):
        _Dummy().validate(bad)
    assert _Dummy().validate(pd.DataFrame(columns=list(REQUIRED_MATCH_COLUMNS))).empty
