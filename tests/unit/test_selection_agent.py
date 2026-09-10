"""Selection agent: deterministic data-quality screening, never an interestingness filter, never
an LLM call — see agents/selection.py for why the judgment doesn't need one."""

from __future__ import annotations

import pandas as pd
import pytest

from pitch_edge.agents.selection import select_fixtures
from pitch_edge.models.base import MatchModel

pytestmark = pytest.mark.unit


class _StubModel(MatchModel):
    def __init__(self, name):
        self.name = name

    def fit(self, df):
        return self

    def predict_proba(self, df):
        raise NotImplementedError


def _fixtures():
    return [
        {
            "match_id": "m1",
            "home_team": "A",
            "away_team": "B",
            "league_code": "E0",
            "lineup_source": "confirmed",
            "date": "2025-01-01",
        },
        {
            "match_id": "m2",
            "home_team": "C",
            "away_team": "D",
            "league_code": "SWE",
            "lineup_source": "provisional",
            "date": "2025-01-01",
        },
    ]


def _write_by_league(tmp_path):
    pd.DataFrame(
        [{"model": "gbdt", "league_code": "E0", "n": 400, "log_loss": 0.6, "market_log_loss": 0.63, "edge_bits": 0.03}]
    ).to_csv(tmp_path / "by_league.csv", index=False)


def test_select_fixtures_keeps_only_routable_leagues(tmp_path):
    _write_by_league(tmp_path)
    state = {"fixtures": _fixtures(), "log": []}
    out = select_fixtures(state, tmp_path, wh=None, available=[_StubModel("gbdt")])
    # m1's league (E0) has backtest evidence for gbdt -> ready; m2 (SWE) has none -> screened out.
    assert [f["match_id"] for f in out["fixtures"]] == ["m1"]
    assert len(out["selected_fixtures"]) == 2
    assert out["selected_fixtures"][0]["ready"] is True
    assert out["selected_fixtures"][1]["ready"] is False
    assert "no backtest evidence" in out["selected_fixtures"][1]["rationale"]
    assert any("select_fixtures" in line for line in out["log"])


def test_select_fixtures_notes_concerns_without_blocking_readiness(tmp_path):
    _write_by_league(tmp_path)
    fixtures = [_fixtures()[0]]  # m1, E0, confirmed lineup
    fixtures[0]["lineup_source"] = "provisional"
    out = select_fixtures({"fixtures": fixtures, "log": []}, tmp_path, wh=None, available=[_StubModel("gbdt")])
    assert out["fixtures"][0]["match_id"] == "m1"
    assert "lineup still provisional" in out["selected_fixtures"][0]["concerns"]
    assert out["selected_fixtures"][0]["ready"] is True  # a concern alone never blocks readiness


def test_select_fixtures_no_fixtures(tmp_path):
    out = select_fixtures({"fixtures": [], "log": []}, tmp_path)
    assert out["fixtures"] == [] and out["selected_fixtures"] == []
