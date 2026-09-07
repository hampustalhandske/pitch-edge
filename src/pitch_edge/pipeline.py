"""End-to-end orchestration: warehouse -> features -> models -> backtests -> RAG -> signals.

One code path from raw data to every number on the dashboard, so nothing
unvalidated can reach the UI. The CLI, scheduler, dashboard and LangGraph
nodes all call into here.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from pitch_edge.agents.graph import GraphDependencies, SignalPipeline
from pitch_edge.agents.risk import RiskLimits, RiskManager, RiskState
from pitch_edge.backtest.engine import BacktestResult, WalkForwardConfig, compare_models
from pitch_edge.backtest.report import calibration_table, results_table, write_report
from pitch_edge.config import Settings, get_settings
from pitch_edge.data.sources.openfootball import OpenFootballSource
from pitch_edge.data.storage import Warehouse
from pitch_edge.data.teams import TeamNameResolver
from pitch_edge.features.build import BASE_FEATURES, FeatureBuilder
from pitch_edge.features.context import load_context
from pitch_edge.models import default_models
from pitch_edge.models.base import MatchModel
from pitch_edge.rag.documents import (
    match_documents,
    news_documents,
    prediction_documents,
    statsbomb_documents,
)
from pitch_edge.rag.index import VectorIndex

logger = logging.getLogger(__name__)

DEFAULT_BACKTEST_LEAGUES = ["E0", "E1", "D1", "SP1", "I1", "F1", "N1", "P1", "B1", "SC0", "T1"]
DEVELOPING_MARKET_CODES = [
    "ARG",
    "AUT",
    "BRA",
    "CHN",
    "DNK",
    "FIN",
    "IRL",
    "JPN",
    "MEX",
    "NOR",
    "POL",
    "ROU",
    "RUS",
    "SWE",
    "SWZ",
    "USA",
]


# ---------------------------------------------------------------------- features
def load_feature_frame(
    wh: Warehouse, leagues: list[str] | None = None, min_date: str | None = None, with_context: bool = True
) -> pd.DataFrame:
    """Feature store for `leagues`. `with_context` adds the Phase-5 Transfermarkt rotation/referee context
    and Wikipedia attention anomalies when those tables exist (they are simply absent otherwise)."""
    leagues = leagues or DEFAULT_BACKTEST_LEAGUES
    matches = wh.matches_with_closing_odds(bookmakers=("PS", "B365", "Avg", "Mkt", "Max", "BFE"))
    if matches.empty:
        return matches
    matches = matches[matches["league_code"].isin(leagues)]
    if min_date:
        matches = matches[pd.to_datetime(matches["date"]) >= pd.Timestamp(min_date)]
    weather = wh.read("weather")
    venues = wh.read("venues")
    context = load_context(wh, matches) if with_context else None
    features = FeatureBuilder().build(
        matches,
        weather=weather if not weather.empty else None,
        venues=venues if not venues.empty else None,
        context=context,
    )
    return features


def persist_features(wh: Warehouse, features: pd.DataFrame) -> int:
    keep = [c for c in features.columns if not c.startswith("_")]
    return wh.replace("features", features[keep])


# ---------------------------------------------------------------------- backtest
def run_backtests(
    features: pd.DataFrame,
    models: list[MatchModel] | None = None,
    config: WalkForwardConfig | None = None,
    wh: Warehouse | None = None,
    report_dir: str | Path | None = None,
    label: str = "main",
    run_id: str | None = None,
) -> dict[str, BacktestResult]:
    settings = get_settings()
    models = models or default_models()
    results = compare_models(features, models, config)
    report_dir = Path(report_dir) if report_dir else settings.reports_dir / label
    write_report(results, report_dir, title=f"Walk-forward backtest — {label}")
    if wh is not None:
        run_id = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        for name, r in results.items():
            bets = r.bets.assign(backtest_id=f"{label}:{name}:{run_id}")
            if not bets.empty:
                wh.upsert("backtest_bets", bets.assign(outcome=bets["outcome"] + "|" + bets["strategy"]))
            summ = r.summaries.assign(
                backtest_id=f"{label}:{name}:{run_id}:" + r.summaries["strategy"],
                label=label,
                run_id=run_id,
                **dict(r.calibration.items()),
            )
            wh.upsert("backtest_summaries", summ)
            preds = r.predictions.assign(model_name=name, run_id=run_id)
            wh.upsert("model_predictions", preds)
        for name, m in zip(results, models, strict=True):
            (report_dir / f"model_card_{name}.json").write_text(json.dumps(m.card(), indent=2, default=str))
    return results


def latest_backtest_tables(wh: Warehouse) -> tuple[pd.DataFrame, pd.DataFrame]:
    summ = wh.read("backtest_summaries")
    if summ.empty:
        return summ, pd.DataFrame()
    latest_run = summ.sort_values("run_id")["run_id"].iloc[-1]
    summ = summ[summ["run_id"] == latest_run]
    bets = wh.read("backtest_bets")
    if not bets.empty:
        bets = bets[bets["backtest_id"].str.endswith(latest_run)]
    return summ, bets


# --------------------------------------------------------------------------- RAG
def build_rag_index(wh: Warehouse, index: VectorIndex | None = None, max_matches: int = 4000) -> VectorIndex:
    index = index or VectorIndex()
    matches = wh.matches_with_closing_odds(bookmakers=("PS",))
    if not matches.empty:
        index.add(match_documents(matches, limit=max_matches))
    preds = wh.read("model_predictions")
    if not preds.empty:
        latest = preds.sort_values("run_id")["run_id"].iloc[-1]
        preds = preds[preds["run_id"] == latest]
        feats = wh.read("features")
        for model_name, grp in preds.groupby("model_name"):
            index.add(prediction_documents(grp.sort_values("date").tail(1500), str(model_name), feats))
    news = wh.read("news_items")
    if not news.empty:
        index.add(news_documents(news.sort_values("published_at").tail(2000)))
    if wh.table_exists("statsbomb_events") and wh.table_exists("statsbomb_matches"):
        ev = wh.read("statsbomb_events")
        sbm = wh.read("statsbomb_matches")
        if not ev.empty:
            index.add(statsbomb_documents(ev, sbm))
    return index


# ------------------------------------------------------------------ live signals
def upcoming_fixture_frame(wh: Warehouse, features: pd.DataFrame, leagues: list[str] | None = None) -> pd.DataFrame:
    """Upcoming fixtures (openfootball) joined to the latest feature snapshot per team.

    Pre-match odds for *future* fixtures need a live odds provider; without one
    we attach synthetic quotes derived from Elo (labelled as such) so the graph
    can be exercised end-to-end. Real closing-line evaluation happens only in the backtest.
    """
    of_map = {
        "E0": ("en.1", "England"),
        "D1": ("de.1", "Germany"),
        "SP1": ("es.1", "Spain"),
        "I1": ("it.1", "Italy"),
        "F1": ("fr.1", "France"),
    }
    leagues = leagues or ["E0", "D1", "SP1", "I1", "F1"]
    src = OpenFootballSource()
    year = datetime.now(UTC).year - (1 if datetime.now(UTC).month < 7 else 0)
    frames = []
    for code in leagues:
        if code not in of_map:
            continue
        try:
            fx = src.fetch_upcoming_fixtures(of_map[code][0], year)
        except Exception as exc:  # noqa: BLE001
            logger.warning("fixtures %s: %s", code, exc)
            continue
        if not fx.empty:
            fx["league_code"] = code
            frames.append(fx)
    if not frames:
        return pd.DataFrame()
    fx = pd.concat(frames, ignore_index=True)
    teams = sorted(set(features["home_team"]) | set(features["away_team"]))
    resolver = TeamNameResolver(teams)
    fx["home_team"] = fx["home_team"].map(lambda n: resolver.resolve(str(n)) or n)
    fx["away_team"] = fx["away_team"].map(lambda n: resolver.resolve(str(n)) or n)
    fx = fx[fx["home_team"].isin(teams) & fx["away_team"].isin(teams)]
    fx = fx[pd.to_datetime(fx["date"]) >= pd.Timestamp.now().normalize()].sort_values("date").head(60)
    if fx.empty:
        return fx
    # latest pre-match snapshot per team = features of their most recent match, carried forward
    latest_home = features.sort_values("date").groupby("home_team").tail(1).set_index("home_team")
    latest_away = features.sort_values("date").groupby("away_team").tail(1).set_index("away_team")
    feat_cols = [c for c in features.columns if c.startswith(("h_", "a_", "elo_", "form_", "attack_", "defence_"))]
    for c in feat_cols:
        src_tbl = latest_home if c.startswith(("h_", "elo_home")) else latest_away
        key = "home_team" if c.startswith(("h_", "elo_home")) else "away_team"
        if c in src_tbl.columns:
            fx[c] = fx[key].map(src_tbl[c])
    fx["elo_home"] = fx["home_team"].map(latest_home["elo_home"]) if "elo_home" in latest_home else 1500.0
    fx["elo_away"] = fx["away_team"].map(latest_away["elo_away"]) if "elo_away" in latest_away else 1500.0
    fx["elo_diff"] = fx["elo_home"].fillna(1500) - fx["elo_away"].fillna(1500)
    fx["elo_exp_home"] = 1 / (1 + 10 ** (-(fx["elo_diff"] + 60) / 400))
    for c in (
        "home_rest_days",
        "away_rest_days",
        "home_matches_last_14d",
        "away_matches_last_14d",
        "away_travel_km",
        "away_fatigue_index",
        "ref_cards_per_game",
        "ref_home_bias",
        "wx_temperature_2m",
        "wx_precipitation",
        "wx_wind_speed_10m",
        "mkt_home_p",
        "mkt_draw_p",
        "mkt_away_p",
        "mkt_overround",
        "league_id",
        *[c for c in BASE_FEATURES if c.startswith(("rot_", "pv_"))],
    ):
        if c not in fx:
            fx[c] = float("nan")
    return fx.reset_index(drop=True)


def synthetic_quotes_from_elo(fixtures: pd.DataFrame, margin: float = 1.05) -> list[dict]:
    """SYNTHETIC pre-match quotes (labelled) derived from Elo expectation — stands in for a live odds API."""
    quotes = []
    for _, r in fixtures.iterrows():
        e = float(r.get("elo_exp_home", 0.5) or 0.5)
        p_draw = 0.26
        p_home = max(0.05, e * (1 - p_draw))
        p_away = max(0.05, 1 - p_draw - p_home)
        quotes.append(
            {
                "match_id": r["match_id"],
                "bookmaker": "synthetic_elo_book",
                "home": round(margin / p_home, 2),
                "draw": round(margin / p_draw, 2),
                "away": round(margin / p_away, 2),
            }
        )
    return quotes


def build_signal_pipeline(
    wh: Warehouse,
    features: pd.DataFrame,
    model: MatchModel,
    limits: RiskLimits | None = None,
    fixtures: pd.DataFrame | None = None,
    quotes: list[dict] | None = None,
) -> SignalPipeline:
    fixtures_df = fixtures if fixtures is not None else upcoming_fixture_frame(wh, features)

    def scout() -> list[dict]:
        return (
            fixtures_df.assign(date=fixtures_df["date"].astype(str)).to_dict(orient="records")
            if not fixtures_df.empty
            else []
        )

    def featurize(rows: list[dict]) -> list[dict]:
        return rows

    def infer(rows: list[dict]) -> list[dict]:
        if not rows:
            return []
        X = pd.DataFrame(rows)
        X["date"] = pd.to_datetime(X["date"])
        probs = model.predict_proba(X)
        return [
            {"match_id": mid, "model": model.name, "p_home": float(h), "p_draw": float(d), "p_away": float(a)}
            for mid, h, d, a in zip(X["match_id"], probs["home"], probs["draw"], probs["away"], strict=True)
        ]

    def fetch_odds(rows: list[dict]) -> list[dict]:
        if quotes is not None:
            return quotes
        return synthetic_quotes_from_elo(pd.DataFrame(rows)) if rows else []

    def sink(alerts: list[dict]) -> None:
        if alerts:
            wh.upsert("paper_trades", pd.DataFrame(alerts))

    deps = GraphDependencies(
        scout, featurize, infer, fetch_odds, RiskManager(limits or RiskLimits(), RiskState()), sink, model.name
    )
    return SignalPipeline(deps)


# --------------------------------------------------------------------- full run
def full_refresh(
    settings: Settings | None = None,
    leagues: list[str] | None = None,
    seasons: list[int] | None = None,
    include_statsbomb: bool = True,
    statsbomb_max_matches: int | None = 40,
    backtest_min_date: str = "2015-07-01",
    fast: bool = False,
) -> dict:
    """Ingest everything, build features, run all backtests, rebuild the RAG index. Returns a summary dict."""
    from pitch_edge.data.ingest import ingest_everything

    settings = settings or get_settings()
    settings.ensure_dirs()
    summary: dict = {}
    with Warehouse(settings.db_path) as wh:
        summary["ingest"] = ingest_everything(
            wh,
            leagues=leagues,
            seasons=seasons,
            include_statsbomb=include_statsbomb,
            include_weather=not fast,
            statsbomb_max_matches=statsbomb_max_matches,
        )
        features = load_feature_frame(wh, min_date=backtest_min_date)
        summary["feature_rows"] = int(len(features))
        if not features.empty:
            persist_features(wh, features)
            models = default_models(include_market=True)
            if fast:
                models = [m for m in models if m.name != "gru_sequence"]
            results = run_backtests(features, models, WalkForwardConfig(), wh=wh, label="main")
            summary["backtest"] = results_table(results).to_dict(orient="records")
            summary["calibration"] = calibration_table(results).to_dict(orient="records")
        summary["rag_docs"] = build_rag_index(wh).count()
        from pitch_edge.artifacts import build_all_artifacts

        summary["artifacts"] = build_all_artifacts(wh, features if not features.empty else None)
    return summary
