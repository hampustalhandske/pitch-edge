"""Fixture prediction for the dossier — computed by the CLI *before* rendering, persisted first.

The dossier never trains; this helper does, once, on the persisted feature store, and writes the
result to `fixture_predictions` so section 1 quotes a warehouse row like every other number. The
row is labelled raw (no fold calibration) and carries the market reference used, if any.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from pitch_edge.data.storage import Warehouse
from pitch_edge.intel.fixture import Fixture
from pitch_edge.models import available_models
from pitch_edge.models.base import MatchModel

logger = logging.getLogger(__name__)


def carry_forward_row(features: pd.DataFrame, fx: Fixture) -> pd.DataFrame:
    """One feature row for an unplayed fixture: each side's latest pre-match snapshot carried forward
    (the same convention as `pipeline.upcoming_fixture_frame`)."""
    feats = features.sort_values("date")
    h_rows = feats[(feats["home_team"] == fx.home_team) | (feats["away_team"] == fx.home_team)]
    a_rows = feats[(feats["home_team"] == fx.away_team) | (feats["away_team"] == fx.away_team)]
    if h_rows.empty or a_rows.empty:
        raise ValueError("no feature history for one of the teams")
    hr, ar = h_rows.iloc[-1], a_rows.iloc[-1]
    row: dict = {
        "match_id": fx.match_id or fx.fixture_key,
        "date": pd.Timestamp(fx.date) if fx.date is not None else pd.Timestamp.utcnow(),
        "home_team": fx.home_team,
        "away_team": fx.away_team,
        "league_code": fx.league_code,
    }
    hp = "h_" if hr["home_team"] == fx.home_team else "a_"
    ap = "h_" if ar["home_team"] == fx.away_team else "a_"
    for c in feats.columns:
        if c.startswith("h_"):
            row[c] = hr.get(hp + c[2:], np.nan)
        elif c.startswith("a_"):
            row[c] = ar.get(ap + c[2:], np.nan)
    row["elo_home"] = hr["elo_home"] if hp == "h_" else hr["elo_away"]
    row["elo_away"] = ar["elo_home"] if ap == "h_" else ar["elo_away"]
    row["elo_diff"] = float(row["elo_home"]) - float(row["elo_away"])
    row["elo_exp_home"] = 1 / (1 + 10 ** (-(row["elo_diff"] + 60) / 400))
    for w in (5, 10):
        row[f"form_diff_r{w}"] = row.get(f"h_pts_r{w}", np.nan) - row.get(f"a_pts_r{w}", np.nan)
        row[f"attack_diff_r{w}"] = row.get(f"h_gf_r{w}", np.nan) - row.get(f"a_gf_r{w}", np.nan)
        row[f"defence_diff_r{w}"] = row.get(f"a_ga_r{w}", np.nan) - row.get(f"h_ga_r{w}", np.nan)
    if "league_code" in feats and fx.league_code in set(feats["league_code"]):
        row["league_id"] = (
            int(feats.loc[feats["league_code"] == fx.league_code, "league_id"].iloc[-1]) if "league_id" in feats else 0
        )
    return pd.DataFrame([row])


def market_reference(wh: Warehouse, fx: Fixture) -> tuple[dict, str | None]:
    """Latest prediction-market no-vig triple for the fixture, if any venue quotes it."""
    if not wh.table_exists("market_snapshots"):
        return {}, None
    from pitch_edge.odds.leadlag import align_snapshots, snapshots_for_fixture

    long = align_snapshots(wh.read("market_snapshots"), freq="1min")
    if long.empty:
        return {}, None
    g = snapshots_for_fixture(long, fx.home_team, fx.away_team).sort_values("ts")
    if g.empty:
        return {}, None
    venue = str(g["venue"].iloc[-1])
    last = g[g["venue"] == venue].groupby("outcome").tail(1).set_index("outcome")["prob"]
    if not {"home", "draw", "away"}.issubset(last.index):
        return {}, None
    return {
        f"mkt_{o}": float(last[o]) for o in ("home", "draw", "away")
    }, f"{venue} snapshot {pd.Timestamp(g['ts'].iloc[-1]).strftime('%Y-%m-%d %H:%M')}"


def predict_fixture(
    wh: Warehouse,
    fx: Fixture,
    model_name: str = "gbdt",
    model: MatchModel | None = None,
    features: pd.DataFrame | None = None,
) -> dict:
    features = wh.read("features") if features is None else features
    if features.empty:
        raise ValueError("feature store is empty — run `pitch-edge features` first")
    features["date"] = pd.to_datetime(features["date"])
    if model is None:
        model = next(m for m in available_models() if m.name == model_name)
        model.fit(features)
    X = carry_forward_row(features, fx)
    p = model.predict_proba(X).iloc[0]
    mkt, mkt_src = market_reference(wh, fx)
    version = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    row = {
        "fixture_key": fx.fixture_key,
        "model_name": model.name,
        "version": version,
        "home_team": fx.home_team,
        "away_team": fx.away_team,
        "match_date": pd.Timestamp(fx.date) if fx.date is not None else pd.NaT,
        "p_home": float(p["home"]),
        "p_draw": float(p["draw"]),
        "p_away": float(p["away"]),
        "mkt_home": mkt.get("mkt_home", np.nan),
        "mkt_draw": mkt.get("mkt_draw", np.nan),
        "mkt_away": mkt.get("mkt_away", np.nan),
        "market_source": mkt_src,
        "features_to": pd.Timestamp(features["date"].max()),
        "n_train": int(len(features)),
        "calibrated": False,
    }
    wh.upsert("fixture_predictions", pd.DataFrame([row]))
    return row
