"""Model router: reads real by_league.csv backtest evidence, never a hardcoded default."""

from __future__ import annotations

import pandas as pd
import pytest

from pitch_edge.agents.router import select_model_for_league
from pitch_edge.backtest.report import model_league_performance
from pitch_edge.models.base import MatchModel

pytestmark = pytest.mark.unit


class _StubModel(MatchModel):
    def __init__(self, name: str):
        self.name = name

    def fit(self, df):  # noqa: D102
        return self

    def predict_proba(self, df):  # noqa: D102
        raise NotImplementedError


def _write_by_league(tmp_path, rows):
    pd.DataFrame(rows).to_csv(tmp_path / "by_league.csv", index=False)
    return tmp_path


def test_model_league_performance_missing_file(tmp_path):
    assert model_league_performance(tmp_path, "E0") is None


def test_model_league_performance_reads_rows(tmp_path):
    _write_by_league(
        tmp_path,
        [
            {
                "model": "gbdt",
                "league_code": "E0",
                "n": 500,
                "log_loss": 0.6,
                "market_log_loss": 0.62,
                "edge_bits": 0.02,
            },
            {
                "model": "dixon_coles",
                "league_code": "E0",
                "n": 500,
                "log_loss": 0.65,
                "market_log_loss": 0.62,
                "edge_bits": -0.03,
            },
        ],
    )
    perf = model_league_performance(tmp_path, "E0")
    assert perf is not None
    assert perf["gbdt"]["edge_bits"] == pytest.approx(0.02)
    assert perf["dixon_coles"]["edge_bits"] == pytest.approx(-0.03)


def test_select_model_for_league_picks_best_positive_edge(tmp_path):
    _write_by_league(
        tmp_path,
        [
            {
                "model": "gbdt",
                "league_code": "SWE",
                "n": 200,
                "log_loss": 0.6,
                "market_log_loss": 0.61,
                "edge_bits": 0.01,
            },
            {
                "model": "gbdt_mkt",
                "league_code": "SWE",
                "n": 200,
                "log_loss": 0.58,
                "market_log_loss": 0.61,
                "edge_bits": 0.04,
            },
            {
                "model": "dixon_coles",
                "league_code": "SWE",
                "n": 200,
                "log_loss": 0.7,
                "market_log_loss": 0.61,
                "edge_bits": -0.1,
            },
        ],
    )
    available = [_StubModel("gbdt"), _StubModel("gbdt_mkt"), _StubModel("dixon_coles")]
    picked = select_model_for_league("SWE", tmp_path, available)
    assert picked is not None and picked.name == "gbdt_mkt"


def test_select_model_for_league_picks_least_bad_model_when_nothing_beats_market(tmp_path):
    """Real market-beating evidence is rare; the router still returns the best-available model
    (least-bad, not None) so the pipeline routes real fixtures through real evidence rather than
    dropping every league where nothing has ever beaten the market."""
    _write_by_league(
        tmp_path,
        [
            {
                "model": "gbdt",
                "league_code": "RUS",
                "n": 200,
                "log_loss": 0.7,
                "market_log_loss": 0.6,
                "edge_bits": -0.15,
            },
            {
                "model": "dixon_coles",
                "league_code": "RUS",
                "n": 200,
                "log_loss": 0.75,
                "market_log_loss": 0.6,
                "edge_bits": -0.25,
            },
        ],
    )
    picked = select_model_for_league("RUS", tmp_path, [_StubModel("gbdt"), _StubModel("dixon_coles")])
    assert picked is not None and picked.name == "gbdt"  # -0.15 beats -0.25, even though both are negative


def test_select_model_for_league_none_without_evidence(tmp_path):
    assert select_model_for_league("XX", tmp_path, [_StubModel("gbdt")]) is None


def test_select_model_for_league_none_when_available_models_have_no_evidence(tmp_path):
    """Evidence exists for the league, but for a model that isn't in `available` — must still be
    None (the router can't route to a model it can't actually run), not a KeyError or wrong pick."""
    _write_by_league(
        tmp_path,
        [
            {
                "model": "some_other_model",
                "league_code": "E0",
                "n": 200,
                "log_loss": 0.6,
                "market_log_loss": 0.6,
                "edge_bits": 0.02,
            }
        ],
    )
    assert select_model_for_league("E0", tmp_path, [_StubModel("gbdt")]) is None
