"""Q2 (r1-pm-bias-v1): favourite-longshot and draw bias of Polymarket relative to the de-vigged sharp close.

Per outcome: gap = Polymarket power-normalised probability at kickoff - h minus the de-vigged Betfair
Exchange closing probability. Trials: mean gap over longshots (benchmark prob < 0.20), favourites (> 0.60)
and draws. A full bucket table is written as detail.

Usage: arch -arm64 uv run python research/studies/r1-pm-bias-v1/run.py {exploratory|evaluation}
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "r1-common"))

from common import SIDES, pm_probs  # noqa: E402
from runner import by_league, finish, start, summarise, thirds, trial_row  # noqa: E402

REG_ID = "r1-pm-bias-v1"


def long_frame(df: pd.DataFrame, pn: np.ndarray, ok: np.ndarray, raw: np.ndarray | None = None) -> pd.DataFrame:
    q = df[[f"q_bf_close_{s}" for s in SIDES]].to_numpy()
    ok = ok & np.isfinite(q).all(axis=1)
    recs = []
    for j, s in enumerate(SIDES):
        recs.append(
            pd.DataFrame(
                {
                    "match_day": df.match_day[ok].to_numpy(),
                    "division": df.division[ok].to_numpy(),
                    "side": s,
                    "q": q[ok, j],
                    "pm": pn[ok, j],
                    "won": (df.result_side[ok] == s).astype(float).to_numpy(),
                }
            )
        )
    out = pd.concat(recs, ignore_index=True)
    out["gap"] = out.pm - out.q
    return out


def main(window: str) -> None:
    reg, sha, df, book, drops = start(REG_ID, window)
    f = reg["filters"]
    h = float(reg["decision_hours_before_kickoff"])
    pn, ok = pm_probs(df, book, h, f["max_stale_hours"], f["raw_sum_min"], f["raw_sum_max"], "power")
    lf = long_frame(df, pn, ok)
    rows, detail = [], {"drops": drops, "window": window, "n_matches": int(lf.match_day.size / 3), "trials": {}}
    for t in reg["grid"]["segment"]:
        name, direction = t["name"], t["direction"]
        if name == "longshot":
            m = lf.q < t["max_prob"]
        elif name == "favourite":
            m = lf.q > t["min_prob"]
        elif name == "draw":
            m = lf.side == "draw"
        else:
            raise ValueError(name)
        sub = lf[m]
        cfg = {"segment": t, "pm_hours_before_kickoff": h, **f}
        s = summarise(sub.gap.to_numpy(), sub.match_day.to_numpy(), direction)
        rows.append(trial_row(reg, sha, window, cfg, "mean_prob_gap_pm_minus_bf_close", s, direction))
        detail["trials"][name] = {
            "gap": s,
            "mean_q": float(sub.q.mean()) if len(sub) else None,
            "mean_pm": float(sub.pm.mean()) if len(sub) else None,
            "win_rate": float(sub.won.mean()) if len(sub) else None,
            "thirds": thirds(sub.gap.to_numpy(), sub.match_day.to_numpy(), direction),
            "by_league": by_league(sub.gap.to_numpy(), sub.match_day.to_numpy(), sub.division.to_numpy(), direction),
        }
    edges = reg["bucket_edges"]
    lf["bucket"] = pd.cut(lf.q, edges, include_lowest=True).astype(str)
    table = []
    for (side_grp, b), g in lf.assign(side_grp=np.where(lf.side == "draw", "draw", "home_away")).groupby(
        ["side_grp", "bucket"]
    ):
        s = summarise(g.gap.to_numpy(), g.match_day.to_numpy(), "two-sided")
        table.append(
            {"side": side_grp, "bucket": b, "mean_q": g.q.mean(), "mean_pm": g.pm.mean(), "win_rate": g.won.mean(), **s}
        )
    detail["bucket_table"] = table
    print(pd.DataFrame(table)[["side", "bucket", "n", "mean_q", "mean_pm", "win_rate", "value", "ci_lo", "ci_hi"]])
    # secondary: raw (unnormalised) Polymarket prices
    raw_snap_pn, raw_ok = pm_probs(df, book, h, f["max_stale_hours"], f["raw_sum_min"], f["raw_sum_max"], "raw")
    lr = long_frame(df, raw_snap_pn, raw_ok)
    detail["raw_price_secondary"] = {
        "longshot": summarise(lr.gap[lr.q < 0.2].to_numpy(), lr.match_day[lr.q < 0.2].to_numpy(), "positive"),
        "favourite": summarise(lr.gap[lr.q > 0.6].to_numpy(), lr.match_day[lr.q > 0.6].to_numpy(), "negative"),
        "draw": summarise(
            lr.gap[lr.side == "draw"].to_numpy(), lr.match_day[lr.side == "draw"].to_numpy(), "two-sided"
        ),
    }
    finish(HERE, REG_ID, window, rows, detail)


if __name__ == "__main__":
    main(sys.argv[1])
