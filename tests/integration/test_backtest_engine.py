"""Features -> models -> walk-forward engine -> report, on the synthetic league."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pitch_edge.backtest.engine import StakingStrategy, WalkForwardBacktester, WalkForwardConfig, compare_models
from pitch_edge.backtest.report import calibration_table, results_table, write_report
from pitch_edge.features.build import FeatureBuilder
from pitch_edge.models import DixonColesMatchModel, GBDTMatchModel

pytestmark = pytest.mark.integration

CFG = WalkForwardConfig(min_train_matches=150, retrain_every_days=60, edge_threshold=0.0, calibration_min_rows=100)


@pytest.fixture(scope="module")
def features(synthetic_league_matches):
    return FeatureBuilder().build(synthetic_league_matches)


@pytest.fixture(scope="module")
def dc_result(features):
    return WalkForwardBacktester(CFG).run(features, DixonColesMatchModel())


def test_predictions_cover_only_out_of_sample_rows(dc_result, features):
    preds = dc_result.predictions
    first_test_date = features.sort_values("date").iloc[CFG.min_train_matches]["date"]
    assert preds["date"].min() >= first_test_date
    assert preds["match_id"].is_unique
    assert np.allclose(preds[["p_home", "p_draw", "p_away"]].sum(axis=1), 1.0, atol=1e-6)


def test_bets_use_early_odds_and_clv_uses_closing(dc_result, features):
    bets = dc_result.bets
    assert not bets.empty
    f = features.set_index("match_id")
    sample = bets.iloc[0]
    side = {"home": "H", "draw": "D", "away": "A"}[sample["outcome"]]
    assert sample["bet_odds"] == f.loc[sample["match_id"], f"PS{side}"]
    assert sample["closing_odds"] == f.loc[sample["match_id"], f"PSC{side}"]


def test_every_strategy_gets_a_summary_row(dc_result):
    s = dc_result.summaries
    assert set(s["strategy"]) == {"kelly_quarter", "kelly_half", "flat_1pct"}
    assert {"mean_clv_pct", "clv_t_stat", "max_drawdown", "final_bankroll", "sharpe"} <= set(s.columns)
    flat = dc_result.bets[dc_result.bets["strategy"] == "flat_1pct"]
    # flat stake = 1% of *current* bankroll, so stakes are all within a sane band
    assert flat["stake"].between(1, 40).all()


def test_calibration_block_includes_market_benchmark(dc_result):
    c = dc_result.calibration
    assert 0 < c["multiclass_log_loss"] < 2 and "market_multiclass_log_loss" in c
    assert c["n_predictions"] == len(dc_result.predictions)


def test_model_with_true_edge_shows_positive_clv(features):
    """Early odds carry sd=0.02 noise vs the true probabilities; a decent model betting
    only where it disagrees with the early line should beat the closing line on average."""
    res = WalkForwardBacktester(
        WalkForwardConfig(min_train_matches=150, retrain_every_days=60, edge_threshold=0.04)
    ).run(features, DixonColesMatchModel(lookback_days=None))
    q = res.summaries.set_index("strategy").loc["kelly_quarter"]
    assert q["n_bets"] > 30
    assert q["mean_clv_pct"] > 0


def test_bankroll_never_negative_and_stake_caps(dc_result):
    bets = dc_result.bets[dc_result.bets["strategy"] == "kelly_quarter"]
    assert (bets["bankroll_after"] > 0).all()
    # stake never exceeds cap * bankroll before the bet
    before = bets["bankroll_after"] - bets["profit"]
    assert (bets["stake"] <= 0.03 * before + 1e-6).all()


def test_insufficient_data_and_missing_odds_raise(features):
    with pytest.raises(ValueError, match="Not enough"):
        WalkForwardBacktester(WalkForwardConfig(min_train_matches=10_000)).run(features, DixonColesMatchModel())
    stripped = features.drop(columns=["PSH", "PSD", "PSA"])
    bt = WalkForwardBacktester(CFG)
    res = bt.run(stripped, DixonColesMatchModel())
    assert bt.bet_price_source == "B365" and not res.bets.empty  # honest fallback to the next real price
    none_at_all = stripped.drop(columns=["B365H", "B365D", "B365A"])
    res2 = WalkForwardBacktester(CFG).run(none_at_all, DixonColesMatchModel())
    assert res2.bets.empty  # no quoted price anywhere -> nothing bettable, engine still reports


def test_compare_models_and_report(features, tmp_path):
    results = compare_models(features, [DixonColesMatchModel(), GBDTMatchModel(n_estimators=30)], CFG)
    table = results_table(results)
    assert set(table["model"]) == {"dixon_coles", "gbdt"}
    assert set(calibration_table(results)["model"]) == {"dixon_coles", "gbdt"}
    path = write_report(results, tmp_path / "data", tmp_path / "rep", title="t")
    text = path.read_text()
    assert "Model vs market" in text and (tmp_path / "data" / "by_league.csv").exists()
    assert path.name == "CASE_STUDY.md" and path.parent == tmp_path / "rep"


def test_custom_strategy_and_no_calibration(features):
    cfg = WalkForwardConfig(
        min_train_matches=150,
        retrain_every_days=90,
        calibrate=False,
        strategies=[StakingStrategy("flat_2pct", "flat", flat_stake_pct=0.02)],
    )
    res = WalkForwardBacktester(cfg).run(features, DixonColesMatchModel())
    assert list(res.summaries["strategy"]) == ["flat_2pct"]
    pd.testing.assert_series_equal(res.predictions["p_home"], res.predictions["raw_home"], check_names=False)
