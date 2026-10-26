"""r5-xv-leadlag-v1: does one venue's hourly move predict the other's next-hour move, and which venue's price
carries information about the Betfair Exchange close that the other lacks?

Hourly tests, per (event, side, H) with 2 <= hours to kickoff and H + 1h <= kickoff:
  dK(H) = Kalshi mid(H) - mid(H - 1h);  dP(H) = PM last trade(H) - last trade(H - 1h), where every PM price
  must be a fresh trade in its own hour (H - 1h, H, H + 1h), so a no-trade hour never shows as a zero move.
  kalshi_leads_pm_hourly: slope of dP(H + 1h) on dK(H)    (H1 > 0)
  pm_leads_kalshi_hourly: slope of dK(H + 1h) on dP(H)    (H1 > 0)
Close tests, per (match, side) at the last hour H* <= kickoff - 1h, football-data-keyed matches with a valid
Betfair close, PM priced at the order-book mid (book coverage only, because PM last-trade bounce biases this
regression toward "Kalshi adds information"):
  y = q_close - PM_mid(H*), x = Kalshi_mid(H*) - PM_mid(H*), b = OLS slope of y on x.
  kalshi_adds_info_vs_bf_close: b > 0;  pm_adds_info_vs_bf_close: 1 - b > 0  (the two are complementary).
Slopes are scored with run_trial as means of per-row influence values (mean = slope); exact block-bootstrap
slope CIs are stored beside them.

Usage: arch -arm64 uv run python research/studies/r5-xv-leadlag-v1/run.py {exploratory|evaluation}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "r5-common"))
sys.path.insert(0, str(HERE.parent / "r1-common"))

from common import power_devig  # noqa: E402
from harness import exact_slope_ci, record, slope_values  # noqa: E402
from partb import CODE_FILES, R5, SIDES, panel, snapshot_tag, universe  # noqa: E402

REG_ID = "r5-xv-leadlag-v1"
MAX_BF_BOOKSUM = 1.05


def hourly_frame(pan: pd.DataFrame) -> pd.DataFrame:
    p = pan[["event_ticker", "side", "H", "ko", "match_day", "series", "hours_to_ko", "mid", "p_fresh"]].copy()
    key = ["event_ticker", "side"]
    for lag, suf in ((1, "_prev"), (-1, "_next")):
        sh = p[key + ["H", "mid", "p_fresh"]].copy()
        sh["H"] = sh.H + pd.Timedelta(hours=lag)
        p = p.merge(sh, on=key + ["H"], how="left", suffixes=("", suf))
    p["dK"] = p.mid - p.mid_prev
    p["dK_next"] = p.mid_next - p.mid
    p["dP"] = p.p_fresh - p.p_fresh_prev
    p["dP_next"] = p.p_fresh_next - p.p_fresh
    ok = (p.hours_to_ko >= 2) & p[["dK", "dK_next", "dP", "dP_next"]].notna().all(axis=1)
    f = p[ok].reset_index(drop=True)
    f["decision_id"] = f.event_ticker + "|" + f.side + "|" + f.H.dt.strftime("%Y-%m-%dT%H")
    f["block"] = f.match_day
    return f


def close_frame(pan: pd.DataFrame, u: pd.DataFrame) -> pd.DataFrame:
    fd = pd.read_parquet(R5 / "fd.parquet").set_index("fd_match_key")
    cols = [f"betfair_ex_{s}_close" for s in SIDES]
    fu = u[u.fd_match_key.notna()].join(fd[cols], on="fd_match_key")
    odds = fu[cols].to_numpy(float)
    q = power_devig(odds)
    bs = (1.0 / odds).sum(axis=1)
    q[~(bs <= MAX_BF_BOOKSUM)] = np.nan
    qd = pd.DataFrame(q, columns=list(SIDES), index=fu.event_ticker).stack().rename("q_close").reset_index()
    qd.columns = ["event_ticker", "side", "q_close"]
    p = pan[(pan.hours_to_ko >= 1) & pan.mid.notna() & pan.b_mid.notna()]
    p = p.sort_values("H").groupby(["event_ticker", "side"]).tail(1)
    f = p.merge(qd, on=["event_ticker", "side"], how="inner")
    f = f[np.isfinite(f.q_close)].reset_index(drop=True)
    f["x"] = f.mid - f.b_mid
    f["y"] = f.q_close - f.b_mid
    f["decision_id"] = f.event_ticker + "|" + f.side
    f["block"] = f.match_day
    return f


def run_test(name: str, x: np.ndarray, y: np.ndarray, fr: pd.DataFrame, window: str, flip: bool = False) -> dict:
    days = fr.block.to_numpy()
    ex = exact_slope_ci(x, y, days)
    if flip:
        ex = {**ex, "slope": 1 - ex["slope"], "ci_lo": 1 - ex["ci_hi"], "ci_hi": 1 - ex["ci_lo"]}

    def sl(m: np.ndarray) -> float:
        xc = x[m] - x[m].mean()
        b = float(np.sum(xc * (y[m] - y[m].mean())) / np.sum(xc * xc))
        return 1 - b if flip else b

    thirds = np.array_split(np.sort(np.unique(days)), 3)
    lg = fr.series.to_numpy()
    vc = pd.Series(lg).value_counts()
    extra = {
        "clean": True,
        "within_ceiling": bool(-0.5 <= ex["slope"] <= 1.5),
        "exact_block_bootstrap_slope": ex,
        "subperiod_estimates": [sl(np.isin(days, t)) for t in thirds],
        "leave_one_league_out": {s: sl(lg != s) for s in vc[vc >= max(30, len(lg) // 50)].index},
        "corr": float(np.corrcoef(x, y)[0, 1]),
    }
    if window == "evaluation":
        v = slope_values(x, y)
        if flip:
            v = 1 - v
        frame = pd.DataFrame({"decision_id": fr.decision_id, "block": days, "value": v})
        record(REG_ID, {"test": name}, frame, "ols_slope", snapshot_tag(), extra, CODE_FILES + [Path(__file__)])
    return {"n": len(x), "days": len(np.unique(days)), **extra}


def main(window: str) -> None:
    u, drops = universe()
    u = u[u.window == window]
    pan = panel(u)
    hf = hourly_frame(pan)
    detail: dict = {"window": window, "events": len(u), "hourly_rows": len(hf), "tests": {}}
    detail["tests"]["kalshi_leads_pm_hourly"] = run_test(
        "kalshi_leads_pm_hourly", hf.dK.to_numpy(), hf.dP_next.to_numpy(), hf, window
    )
    detail["tests"]["pm_leads_kalshi_hourly"] = run_test(
        "pm_leads_kalshi_hourly", hf.dP.to_numpy(), hf.dK_next.to_numpy(), hf, window
    )
    # Descriptive (not trials): contemporaneous and own-lag slopes.
    detail["contemporaneous_dP_on_dK"] = exact_slope_ci(hf.dK.to_numpy(), hf.dP.to_numpy(), hf.block.to_numpy())
    detail["own_lag_dP_next_on_dP"] = exact_slope_ci(hf.dP.to_numpy(), hf.dP_next.to_numpy(), hf.block.to_numpy())
    detail["own_lag_dK_next_on_dK"] = exact_slope_ci(hf.dK.to_numpy(), hf.dK_next.to_numpy(), hf.block.to_numpy())
    cf = close_frame(pan, u)
    detail["close_rows"] = len(cf)
    if len(cf) > 10:
        detail["tests"]["kalshi_adds_info_vs_bf_close"] = run_test(
            "kalshi_adds_info_vs_bf_close", cf.x.to_numpy(), cf.y.to_numpy(), cf, window
        )
        detail["tests"]["pm_adds_info_vs_bf_close"] = run_test(
            "pm_adds_info_vs_bf_close", cf.x.to_numpy(), cf.y.to_numpy(), cf, window, flip=True
        )
    out = R5 / REG_ID
    out.mkdir(parents=True, exist_ok=True)
    (out / f"detail_{window}.json").write_text(json.dumps(detail, indent=2, default=str))
    print(json.dumps(detail, indent=1, default=str))


if __name__ == "__main__":
    main(sys.argv[1])
