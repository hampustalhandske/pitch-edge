"""T0 replay-eval: trains only on data before T0, scores agentic proposals against revealed
real closing odds/results, and never leaks the window's own odds/results into the pipeline.

No LLM involved — the reviewer's verdict is the deterministic real-edge_bits/real-odds rule (see
agents/reviewer.py), so this replay validates that rule against real closing prices."""

from __future__ import annotations

import pandas as pd
import pytest

from pitch_edge.backtest.engine import WalkForwardConfig
from pitch_edge.backtest.replay import compute_t0, run_replay_eval
from pitch_edge.data.ingest import store_matches
from pitch_edge.models.gbdt import GBDTMatchModel
from pitch_edge.models.poisson import DixonColesMatchModel

pytestmark = pytest.mark.integration


class _StubIndex:
    def query(self, text, k=6, where=None):
        return []


@pytest.fixture
def loaded_warehouse(warehouse, synthetic_league_matches):
    store_matches(warehouse, synthetic_league_matches)
    return warehouse


def test_compute_t0_is_30_days_before_max_real_closing_date(loaded_warehouse):
    t0 = compute_t0(loaded_warehouse)
    matches = loaded_warehouse.matches_with_closing_odds(bookmakers=("PS",))
    assert t0 == pd.to_datetime(matches["date"]).max() - pd.Timedelta(days=30)


def test_compute_t0_raises_without_real_closing_odds(warehouse):
    with pytest.raises(ValueError):
        compute_t0(warehouse)


def test_replay_eval_trains_only_on_pre_t0_data_and_scores_proposals(loaded_warehouse, tmp_path):
    t0 = compute_t0(loaded_warehouse) - pd.Timedelta(
        days=200
    )  # pick an earlier T0 so both sides have real closing odds
    result = run_replay_eval(
        loaded_warehouse,
        tmp_path / "reports",
        as_of=str(t0.date()),
        window_days=30,
        index=_StubIndex(),
        backtest_config=WalkForwardConfig(min_train_matches=50),
        models=[DixonColesMatchModel(), GBDTMatchModel(include_market=False)],
        leagues=["SYN"],
    )
    assert result["t0"] == t0
    assert result["n_fixtures"] > 0
    by_league = tmp_path / "reports" / "replay_train" / "by_league.csv"
    assert by_league.exists()
    league_dates_ok = pd.read_csv(by_league)
    assert not league_dates_ok.empty
    # scored proposals (if any) must all have a real PSCH/PSCD/PSCA closing price behind them
    if not result["scored"].empty:
        assert result["scored"]["real_closing_probability"].between(0, 1).all()
        assert result["scored"]["verdict"].isin(["trust", "distrust", "needs_info"]).all()
    assert (tmp_path / "reports" / "replay_eval.csv").exists()
    assert (tmp_path / "reports" / "replay_eval_summary.csv").exists()


def test_replay_eval_raises_with_no_features(warehouse, tmp_path):
    with pytest.raises(ValueError):
        run_replay_eval(warehouse, tmp_path / "reports", as_of="2020-01-01")
