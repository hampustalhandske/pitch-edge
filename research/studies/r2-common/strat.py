"""Scoring of R2 paper strategies: CLV / ROI with match-day bootstrap CIs and stability breakdowns.

A bet frame has one row per paper bet with columns: match_date, division, odds (taken, vig-inclusive),
p_close (de-vigged benchmark probability of the side backed) and y (1 if the side won, else 0; NaN when
not settled, e.g. Asian handicap). CLV per bet is log(odds * p_close).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import r2lib as L

MIN_LEAGUE = 100


def score(bets: pd.DataFrame, lo: str, hi: str) -> dict:
    """CLV (primary) and ROI (secondary) with CIs, moments, thirds, per-division and leave-one-division-out."""
    if len(bets) == 0:
        return {"clv": {"n": 0}, "n": 0}
    clv = np.log(bets["odds"].to_numpy() * bets["p_close"].to_numpy())
    out: dict = {"n": len(bets), "clv": L.boot_mean(clv, bets["match_date"]), "moments": L.sharpe_moments(clv)}
    out["share_clv_pos"] = float((clv > 0).mean())
    out["mean_odds"] = float(bets["odds"].mean())
    if bets["y"].notna().all():
        ret = bets["odds"].to_numpy() * bets["y"].to_numpy() - 1
        out["roi"] = L.boot_mean(ret, bets["match_date"])
        out["hit_rate"] = float(bets["y"].mean())
        out["expected_hit_rate_close"] = float(bets["p_close"].mean())
    th = L.thirds(bets["match_date"], lo, hi)
    out["thirds"] = {}
    for k in (1, 2, 3):
        m = (th == k).to_numpy()
        if m.sum():
            out["thirds"][k] = L.boot_mean(clv[m], bets["match_date"][m])
    counts = bets["division"].value_counts()
    grp = np.where(bets["division"].map(counts) >= MIN_LEAGUE, bets["division"], "OTHER")
    per = pd.DataFrame({"div": grp, "clv": clv}).groupby("div")["clv"].agg(["size", "mean"])
    out["per_division"] = per.round(5).to_dict(orient="index")
    out["n_div_positive"] = int((per["mean"] > 0).sum())
    out["n_div"] = len(per)
    lodo = {}
    for div in bets["division"].unique():
        keep = (bets["division"] != div).to_numpy()
        lodo[div] = float(clv[keep].mean()) if keep.any() else np.nan
    out["lodo_min"] = float(np.nanmin(list(lodo.values())))
    out["lodo_max"] = float(np.nanmax(list(lodo.values())))
    return out


def ledger_strategy(led: L.Ledger, cfg: dict, res: dict, wname: str, label: str) -> None:
    """Record a strategy's primary CLV result (one-sided, positive) as one trial row."""
    if res.get("n", 0) == 0:
        led.add(cfg, {"n": 0}, "clv_log_mean", wname, label, "positive")
        return
    led.add(cfg, res["clv"], "clv_log_mean", wname, label, "positive", res["moments"])


def clv_matrix(frame_dates: pd.Series, bet_sets: list[pd.Series]) -> np.ndarray:
    """(T, N) matrix for PBO: per eligible match (time-ordered), the CLV of config j's bet or 0 if no bet.

    Each element of `bet_sets` is a Series of CLV indexed by the eligible-frame index.
    """
    order = frame_dates.sort_values(kind="stable").index
    return np.column_stack([s.reindex(order).fillna(0.0).to_numpy() for s in bet_sets])
