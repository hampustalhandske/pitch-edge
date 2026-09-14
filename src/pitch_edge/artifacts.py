"""Derived artifacts the dashboard shows beyond the backtest: per-league Dixon-Coles team
strengths, GBDT feature importances, and a data-universe summary. All computed from
warehouse tables; all persisted back so the app never trains anything at render time."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pandas as pd

from pitch_edge.config import get_settings
from pitch_edge.data.storage import Warehouse
from pitch_edge.models.gbdt import GBDTMatchModel
from pitch_edge.models.poisson import DixonColesMatchModel


def _write_artifact(name: str, text: str) -> None:
    path = get_settings().artifacts_dir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def build_team_strengths(wh: Warehouse, features: pd.DataFrame) -> int:
    if features.empty:
        return 0
    dc = DixonColesMatchModel().fit(features)
    rows = []
    for code, model in dc._models.items():
        for team in model.teams_:
            rows.append(
                {
                    "league_code": code,
                    "team": team,
                    "attack": model.attack_[team],
                    "defence": model.defence_[team],
                    "home_advantage": model.home_advantage_,
                    "rho": model.rho_,
                    "as_of": pd.Timestamp(features["date"].max()),
                }
            )
    return wh.replace("team_strength", pd.DataFrame(rows))


def build_feature_importance(wh: Warehouse, features: pd.DataFrame) -> int:
    if features.empty:
        return 0
    gb = GBDTMatchModel(n_estimators=200).fit(features)
    imp = gb.feature_importance()
    if imp.empty:
        # sklearn HGB has no impurity importances; use permutation importance on a holdout slice
        from sklearn.inspection import permutation_importance

        hold = features.sort_values("date").tail(min(3000, len(features) // 5))
        X = gb._matrix(hold)
        r = permutation_importance(
            gb._clf, X, hold["result"].to_numpy(), n_repeats=3, random_state=0, scoring="neg_log_loss"
        )
        imp = pd.Series(r.importances_mean, index=gb._cols).sort_values(ascending=False)
    df = imp.reset_index()
    df.columns = ["feature", "importance"]
    df["backend"] = gb.backend
    return wh.replace("feature_importance", df)


def build_data_universe(wh: Warehouse) -> dict:
    q = wh.query
    out: dict = {"generated_at": datetime.now(UTC).isoformat(timespec="seconds")}
    if wh.table_exists("matches"):
        out["matches"] = int(wh.count("matches"))
        out["leagues"] = int(q("SELECT count(DISTINCT league_code) FROM matches").iloc[0, 0])
        out["countries"] = (
            int(q("SELECT count(DISTINCT country) FROM matches WHERE country IS NOT NULL").iloc[0, 0])
            if "country" in wh.columns("matches")
            else 0
        )
        out["teams"] = int(q("SELECT count(DISTINCT home_team) FROM matches").iloc[0, 0])
        rng = q("SELECT min(date)::DATE a, max(date)::DATE b FROM matches").iloc[0]
        out["date_range"] = [str(rng["a"]), str(rng["b"])]
        cols = set(wh.columns("matches"))
        country_expr = "coalesce(country, league_code)" if "country" in cols else "league_code"
        stats_expr = "count(*) FILTER (WHERE home_shots IS NOT NULL)" if "home_shots" in cols else "0"
        elo_expr = "count(*) FILTER (WHERE home_elo IS NOT NULL)" if "home_elo" in cols else "0"
        out["by_league"] = q(
            f"SELECT league_code, {country_expr} AS country, count(*) n, min(date)::DATE first_date, max(date)::DATE last_date, "
            f"{stats_expr} AS with_stats, {elo_expr} AS with_elo FROM matches GROUP BY 1,2 ORDER BY n DESC"
        ).to_dict(orient="records")
        out["by_season"] = q("SELECT season, count(*) n FROM matches GROUP BY 1 ORDER BY 1").to_dict(orient="records")
    for t in (
        "odds",
        "team_ratings",
        "venues",
        "weather",
        "news_items",
        "dim_soccer_markets",
        "pmxt_orderbook",
        "statsbomb_matches",
        "statsbomb_events",
        "openfootball_matches",
        "features",
        "model_predictions",
        "backtest_bets",
    ):
        out[t] = int(wh.count(t))
    if wh.table_exists("odds"):
        out["odds_by_book"] = q(
            "SELECT bookmaker, is_closing, market, count(*) n FROM odds GROUP BY 1,2,3 ORDER BY n DESC"
        ).to_dict(orient="records")
    out["sources"] = wh.health().to_dict(orient="records") if wh.table_exists("pipeline_runs") else []
    _write_artifact("data_universe.json", json.dumps(out, indent=2, default=str))
    return out


def build_all_artifacts(wh: Warehouse, features: pd.DataFrame | None = None) -> dict[str, int]:
    get_settings().ensure_dirs()
    report: dict[str, int] = {}
    if features is None and wh.table_exists("features"):
        features = wh.read("features")
    if features is not None and not features.empty:
        report["team_strength"] = build_team_strengths(wh, features)
        report["feature_importance"] = build_feature_importance(wh, features)
    report["universe_matches"] = int(build_data_universe(wh).get("matches", 0))
    return report
