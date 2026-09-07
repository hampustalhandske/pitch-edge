"""Feature-group ablations: does an alternative-data feature earn its place?

Runs the same walk-forward backtest for a GBDT with and without a feature
group (referee bias, weather, travel/fatigue, Elo, rolling form) and reports
the delta in log-loss and closing-line value. This is the experiment behind
CASE_STUDY.md; negative or insignificant deltas are reported as such.
"""

from __future__ import annotations

import pandas as pd

from pitch_edge.backtest.engine import BacktestResult, WalkForwardConfig, compare_models
from pitch_edge.models.base import MatchModel
from pitch_edge.models.gbdt import GBDTMatchModel

FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "referee": ("ref_",),
    "weather": ("wx_",),
    "travel_fatigue": (
        "away_travel_km",
        "away_fatigue_index",
        "home_rest",
        "away_rest",
        "home_matches_last",
        "away_matches_last",
    ),
    "elo": ("elo_",),
    "rolling_form": ("h_", "a_", "form_diff", "attack_diff", "defence_diff"),
    # Phase 5 signals — pre-registered in CASE_STUDY.md (Result 7)
    "wiki_attention": ("pv_",),
    "rotation_load": ("rot_",),
    # Phase 6 — pre-registered in CASE_STUDY.md Result 8
    "squad_value": ("sv_",),
}


def run_ablation(
    features: pd.DataFrame,
    groups: list[str] | None = None,
    config: WalkForwardConfig | None = None,
    n_estimators: int = 300,
) -> tuple[pd.DataFrame, dict[str, BacktestResult]]:
    groups = groups or list(FEATURE_GROUPS)
    # the ablation's "full" model reads EVERY group, including the ones the production default excludes
    models: list[MatchModel] = [GBDTMatchModel(n_estimators=n_estimators, exclude_prefixes=(), name_suffix="_full")]
    for g in groups:
        models.append(
            GBDTMatchModel(n_estimators=n_estimators, exclude_prefixes=FEATURE_GROUPS[g], name_suffix=f"_no_{g}")
        )
    results = compare_models(features, models, config)
    base = results["gbdt_full"]
    base_q = base.summaries.set_index("strategy").loc["kelly_quarter"]
    rows = []
    for g in groups:
        r = results[f"gbdt_no_{g}"]
        q = r.summaries.set_index("strategy").loc["kelly_quarter"]
        rows.append(
            {
                "group_removed": g,
                "log_loss_full": base.calibration["multiclass_log_loss"],
                "log_loss_without": r.calibration["multiclass_log_loss"],
                "delta_log_loss": r.calibration["multiclass_log_loss"] - base.calibration["multiclass_log_loss"],
                "clv_full_pct": base_q["mean_clv_pct"],
                "clv_without_pct": q["mean_clv_pct"],
                "delta_clv_pct": base_q["mean_clv_pct"] - q["mean_clv_pct"],
                "clv_t_full": base_q["clv_t_stat"],
                "clv_t_without": q["clv_t_stat"],
                "n_bets_full": int(base_q["n_bets"]),
                "n_bets_without": int(q["n_bets"]),
            }
        )
    return pd.DataFrame(rows), results
