from __future__ import annotations

import pandas as pd
import pytest

from pitch_edge.data.sources.base import REQUIRED_MATCH_COLUMNS, MatchDataSource

pytestmark = pytest.mark.unit


class _DummySource(MatchDataSource):
    name = "dummy"

    def fetch_matches(self, **kwargs) -> pd.DataFrame:
        return pd.DataFrame()


def _valid_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "match_id": ["1"],
            "date": [pd.Timestamp("2023-01-01")],
            "league": ["Test"],
            "season": ["2023/24"],
            "home_team": ["A"],
            "away_team": ["B"],
            "home_goals": [1],
            "away_goals": [0],
            "source": ["dummy"],
        }
    )


def test_validate_passes_well_formed_frame():
    source = _DummySource()
    df = source.validate(_valid_df())
    assert len(df) == 1


def test_validate_raises_on_missing_columns():
    source = _DummySource()
    bad = _valid_df().drop(columns=["home_goals"])
    with pytest.raises(ValueError, match="missing required columns"):
        source.validate(bad)


def test_validate_raises_on_null_goals():
    source = _DummySource()
    bad = _valid_df()
    bad.loc[0, "home_goals"] = None
    with pytest.raises(ValueError, match="null goal values"):
        source.validate(bad)


def test_validate_allows_empty_dataframe_with_correct_schema():
    source = _DummySource()
    empty = pd.DataFrame(columns=list(REQUIRED_MATCH_COLUMNS))
    result = source.validate(empty)
    assert result.empty
