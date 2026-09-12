"""Circumstance-stratified backtest evidence, checkpointed across time.

Extends the (model, league) evidence in `report.py::league_table` to finer circumstance
slices — how big the rest gap is, who the referee was, how lopsided the squads' value is,
how far the away side travelled, the weather, and which outcome the model favoured — with
a minimum sample-size floor and a Benjamini-Hochberg false-discovery-rate correction within
each (model, slice_dim) family before a slice counts as `significant`.

Checkpointed at the backtest's own retrain cadence: a checkpoint dated `T` is built only from
predictions strictly before `T`, so a slice's evidence never reflects matches that happened on
or after the date it is dated to. This mirrors the walk-forward discipline the rest of the
project already applies to model fitting — applied here to evidence about which model to trust.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from pitch_edge.backtest.engine import BacktestResult
from pitch_edge.models.base import OUTCOMES

MIN_SLICE_N = 50
FDR_ALPHA = 0.05

SLICE_DIMENSIONS = (
    "league_code",
    "predicted_favorite",
    "rest_days_gap_bucket",
    "referee",
    "squad_value_gap_bucket",
    "travel_fatigue_bucket",
    "weather_bucket",
)

REST_DAYS_GAP_BINS = ([-0.01, 2, 4, np.inf], ["<2d", "2-4d", "5+d"])
SQUAD_VALUE_GAP_BINS = ([-0.01, 0.05, 0.15, np.inf], ["balanced", "moderate_gap", "large_gap"])
WEATHER_ADVERSE_PRECIPITATION = 0.0
WEATHER_ADVERSE_WIND_SPEED = 30.0

# `travel_fatigue_bucket` and `predicted_favorite` are excluded from live single-fixture lookups
# (`fixture_slice_values`): the former is quantile-binned (`pd.qcut`) against the training
# distribution, which a lone future fixture has no distribution to be binned against without
# persisting the training quantile edges; the latter needs a model's own prediction, which is
# circular for the model-selection step this feeds. Both stay valid for offline evidence
# generation over historical data, where a whole distribution is available.
LIVE_SLICE_DIMENSIONS = ("league_code", "referee", "rest_days_gap_bucket", "squad_value_gap_bucket", "weather_bucket")

_CIRCUMSTANCE_COLUMNS = (
    "match_id",
    "home_rest_days",
    "away_rest_days",
    "away_travel_km",
    "away_fatigue_index",
    "referee",
    "wx_precipitation",
    "wx_wind_speed_10m",
    "sv_missing_pct_diff",
)


def add_slice_columns(predictions: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Join circumstance columns from `features` (the same frame the backtest ran on — already
    point-in-time-correct by construction) onto `predictions` by `match_id`, and derive every
    bucket in `SLICE_DIMENSIONS` that its source columns support."""
    cols = ["match_id"] + [
        c for c in _CIRCUMSTANCE_COLUMNS if c != "match_id" and c in features.columns and c not in predictions.columns
    ]
    df = predictions.merge(features[cols].drop_duplicates("match_id"), on="match_id", how="left")

    fav_idx = df[[f"p_{o}" for o in OUTCOMES]].to_numpy().argmax(axis=1)
    df["predicted_favorite"] = np.array(OUTCOMES)[fav_idx]

    if {"home_rest_days", "away_rest_days"} <= set(df.columns):
        gap = (df["home_rest_days"] - df["away_rest_days"]).abs()
        edges, labels = REST_DAYS_GAP_BINS
        df["rest_days_gap_bucket"] = pd.cut(gap, bins=edges, labels=labels)

    if "away_fatigue_index" in df.columns:
        df["travel_fatigue_bucket"] = pd.qcut(
            df["away_fatigue_index"], q=3, labels=["low", "medium", "high"], duplicates="drop"
        )

    if "sv_missing_pct_diff" in df.columns:
        gap = df["sv_missing_pct_diff"].abs()
        edges, labels = SQUAD_VALUE_GAP_BINS
        df["squad_value_gap_bucket"] = pd.cut(gap, bins=edges, labels=labels)

    if {"wx_precipitation", "wx_wind_speed_10m"} <= set(df.columns):
        known = df[["wx_precipitation", "wx_wind_speed_10m"]].notna().all(axis=1)
        adverse = (df["wx_precipitation"] > WEATHER_ADVERSE_PRECIPITATION) | (
            df["wx_wind_speed_10m"] > WEATHER_ADVERSE_WIND_SPEED
        )
        df["weather_bucket"] = pd.Series(np.where(adverse, "adverse", "normal"), index=df.index).where(known)

    return df


def fixture_slice_values(row: pd.Series) -> dict[str, str]:
    """The `LIVE_SLICE_DIMENSIONS` bucket values for one fixture row (a `features` row, not a
    backtest prediction), using the same fixed thresholds `add_slice_columns` uses — for matching
    a not-yet-played fixture against `slice_evidence`'s offline evidence."""
    values: dict[str, str] = {}
    if "league_code" in row and pd.notna(row["league_code"]):
        values["league_code"] = str(row["league_code"])
    if "referee" in row and pd.notna(row["referee"]):
        values["referee"] = str(row["referee"])
    if pd.notna(row.get("home_rest_days")) and pd.notna(row.get("away_rest_days")):
        gap = abs(row["home_rest_days"] - row["away_rest_days"])
        edges, labels = REST_DAYS_GAP_BINS
        values["rest_days_gap_bucket"] = str(pd.cut([gap], bins=edges, labels=labels)[0])
    if pd.notna(row.get("sv_missing_pct_diff")):
        gap = abs(row["sv_missing_pct_diff"])
        edges, labels = SQUAD_VALUE_GAP_BINS
        values["squad_value_gap_bucket"] = str(pd.cut([gap], bins=edges, labels=labels)[0])
    if pd.notna(row.get("wx_precipitation")) and pd.notna(row.get("wx_wind_speed_10m")):
        adverse = row["wx_precipitation"] > WEATHER_ADVERSE_PRECIPITATION or (
            row["wx_wind_speed_10m"] > WEATHER_ADVERSE_WIND_SPEED
        )
        values["weather_bucket"] = "adverse" if adverse else "normal"
    return values


def _log_loss_edge(sub: pd.DataFrame) -> dict | None:
    """None if `sub` has fewer than `MIN_SLICE_N` rows with a real (non-null) market price."""
    y = sub["result"].to_numpy().astype(int)
    P = sub[[f"p_{o}" for o in OUTCOMES]].to_numpy()
    M = sub[[f"mkt_{o}" for o in OUTCOMES]].to_numpy()
    ok = ~np.isnan(M).any(axis=1)
    n = int(ok.sum())
    if n < MIN_SLICE_N:
        return None
    rows = np.arange(n)
    per_row_ll = -np.log(np.clip(P[ok][rows, y[ok]], 1e-12, 1))
    per_row_mll = -np.log(np.clip(M[ok][rows, y[ok]], 1e-12, 1))
    diff_bits = (per_row_mll - per_row_ll) / np.log(2)
    edge_bits = float(diff_bits.mean())
    se = float(diff_bits.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    t_stat = edge_bits / se if se > 0 else 0.0
    p_value = float(2 * (1 - norm.cdf(abs(t_stat)))) if se > 0 else 1.0
    return {
        "n": n,
        "log_loss": float(per_row_ll.mean()),
        "market_log_loss": float(per_row_mll.mean()),
        "edge_bits": edge_bits,
        "ci_low": edge_bits - 1.96 * se,
        "ci_high": edge_bits + 1.96 * se,
        "t_stat": float(t_stat),
        "p_value": p_value,
    }


def _bh_qvalues(p_values: pd.Series) -> pd.Series:
    """Benjamini-Hochberg adjusted p-values (q-values), same index as `p_values`."""
    n = len(p_values)
    if n == 0:
        return pd.Series([], dtype=float)
    order = p_values.sort_values().index
    ranks = np.arange(1, n + 1)
    raw_q = p_values.loc[order].to_numpy() * n / ranks
    q = np.clip(np.minimum.accumulate(raw_q[::-1])[::-1], 0, 1)
    return pd.Series(q, index=order).reindex(p_values.index)


def slice_evidence(predictions: pd.DataFrame, features: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """One row per (slice_dim, slice_value): `n`, `log_loss`, `market_log_loss`, `edge_bits`,
    a 95% CI, and a Benjamini-Hochberg `q_value`/`significant` computed within each slice_dim
    family. A slice with fewer than `MIN_SLICE_N` matches is dropped entirely, not shown with a
    caveat — there is nothing here for a human or an LLM to over-read."""
    df = add_slice_columns(predictions, features)
    rows = []
    for dim in SLICE_DIMENSIONS:
        if dim not in df.columns:
            continue
        for value, g in df.groupby(dim, observed=True):
            if pd.isna(value):
                continue
            stats = _log_loss_edge(g)
            if stats is None:
                continue
            rows.append({"model": model_name, "slice_dim": dim, "slice_value": str(value), **stats})
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(
            columns=[
                "model",
                "slice_dim",
                "slice_value",
                "n",
                "log_loss",
                "market_log_loss",
                "edge_bits",
                "ci_low",
                "ci_high",
                "t_stat",
                "p_value",
                "q_value",
                "significant",
            ]
        )
    out["q_value"] = np.nan
    for _dim, g in out.groupby("slice_dim"):
        out.loc[g.index, "q_value"] = _bh_qvalues(g["p_value"])
    out["significant"] = out["q_value"] <= FDR_ALPHA
    return out


def slice_evidence_checkpoints(
    result: BacktestResult, features: pd.DataFrame, checkpoint_every_days: int | None = None
) -> pd.DataFrame:
    """Snapshot `slice_evidence` at each of the backtest's own retrain-cadence checkpoints, each
    computed from predictions strictly before that checkpoint only, with a `checkpoint_date`
    column added. A checkpoint dated `T` never reflects matches on or after `T`."""
    preds = result.predictions.copy()
    preds["date"] = pd.to_datetime(preds["date"])
    step = checkpoint_every_days or result.config.retrain_every_days
    start, end = preds["date"].min(), preds["date"].max()
    if pd.isna(start) or pd.isna(end):
        return pd.DataFrame()
    checkpoints = pd.date_range(start, end, freq=f"{step}D")[1:]
    frames = []
    for cp in checkpoints:
        before = preds[preds["date"] < cp]
        if before.empty:
            continue
        ev = slice_evidence(before, features, result.model_name)
        if ev.empty:
            continue
        ev = ev.assign(checkpoint_date=cp.date().isoformat())
        frames.append(ev)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def nearest_checkpoint(slice_evidence: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Rows from the checkpoint nearest before `as_of` — never a later one, so a question dated in
    the past is never answered with evidence from matches that hadn't happened yet as of that date."""
    if slice_evidence.empty:
        return slice_evidence
    dates = pd.to_datetime(slice_evidence["checkpoint_date"])
    eligible = slice_evidence[dates < as_of]
    if eligible.empty:
        return eligible.iloc[0:0]
    nearest = pd.to_datetime(eligible["checkpoint_date"]).max()
    return eligible[pd.to_datetime(eligible["checkpoint_date"]) == nearest]


def select_trusted_model(
    slice_evidence: pd.DataFrame, fixture_slices: dict[str, str], model_names: list[str], as_of: pd.Timestamp
) -> tuple[str | None, pd.DataFrame]:
    """The model with the best `edge_bits` among evidence rows matching this fixture's own
    circumstances (`fixture_slices`, from `fixture_slice_values`) at the checkpoint nearest before
    `as_of` — even when that edge is negative, the least-bad model rather than none at all. Returns
    `(None, empty)` only when there is no matching evidence for any of `model_names` whatsoever.
    Mirrors the deleted `agents/router.py::select_model_for_league`'s design, generalized from
    league-only to every fixed circumstance slice."""
    cp = nearest_checkpoint(slice_evidence, as_of)
    if cp.empty or not fixture_slices:
        return None, cp.iloc[0:0]
    is_match = cp.apply(lambda row: fixture_slices.get(row["slice_dim"]) == row["slice_value"], axis=1)
    matches = cp[is_match & cp["model"].isin(model_names)]
    if matches.empty:
        return None, matches
    best = matches.loc[matches["edge_bits"].idxmax()]
    return str(best["model"]), matches
