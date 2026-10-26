"""Q4 (r1-pm-convergence-v1): does the Polymarket-minus-sharp-open gap at T predict Polymarket's move to kickoff?

Per (match, outcome), at ts = kickoff - T with the football-data open known by ts:
  gap  = p_PM(ts) - q_open           (last trade known by ts minus the de-vigged sharp open)
  move = p_PM(last before kickoff) - p_PM(ts)
Primary: pooled OLS slope of move on gap, match-day block bootstrap; convergence means slope < 0 (-1 = full).
Secondary (descriptive): two-regressor fit move ~ p_PM(ts) + q_open; and the sharp side's own move
(q_bf_close - q_open) on the same gap, to see which venue moves toward which.

Usage: arch -arm64 uv run python research/studies/r1-pm-convergence-v1/run.py {exploratory|evaluation}
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "r1-common"))

from common import SIDES, block_bootstrap_slope, one_sided_p  # noqa: E402
from runner import finish, start, trial_row  # noqa: E402

REG_ID = "r1-pm-convergence-v1"


def frame(df: pd.DataFrame, book, hours: float, max_stale: float, max_stale_last: float) -> pd.DataFrame:
    recs = []
    for r in df.itertuples(index=False):
        ts = r.ko - pd.Timedelta(hours=hours)
        if not (r.open_known_at <= ts):
            continue
        for s in SIDES:
            qo = getattr(r, f"q_open_{s}")
            if not np.isfinite(qo):
                continue
            cid = getattr(r, f"cond_{s}")
            p, _, st, kn, _ = book.at(cid, ts)
            pl, _, stl, knl, _ = book.at(cid, r.ko)
            if not (np.isfinite(p) and np.isfinite(pl)) or st > max_stale or stl > max_stale_last:
                continue
            assert kn <= ts.value
            recs.append(
                {
                    "match_day": r.match_day,
                    "division": r.division,
                    "side": s,
                    "p": p,
                    "q_open": qo,
                    "q_close": getattr(r, f"q_bf_close_{s}"),
                    "p_last": pl,
                    "same_trade": knl == kn,
                }
            )
    out = pd.DataFrame(recs)
    out["gap"] = out.p - out.q_open
    out["move"] = out.p_last - out.p
    return out


def slope_summary(x, y, days) -> dict:
    if len(x) < 3:
        return {"n": len(x), "n_days": 0, "value": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "p": np.nan}
    b, boot = block_bootstrap_slope(x, y, days)
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return {
        "n": len(x),
        "n_days": int(pd.Series(days).nunique()),
        "value": b,
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "p": float(one_sided_p(boot[np.isfinite(boot)], b, "negative")),
    }


def main(window: str) -> None:
    reg, sha, df, book, drops = start(REG_ID, window)
    f = reg["filters"]
    rows, detail = [], {"drops": drops, "window": window, "trials": {}}
    for hours in reg["grid"]["hours_before_kickoff"]:
        fr = frame(df, book, float(hours), f["max_stale_hours"], f["max_stale_last_hours"])
        cfg = {"hours_before_kickoff": hours, **f}
        s = slope_summary(fr.gap.to_numpy(), fr.move.to_numpy(), fr.match_day.to_numpy())
        rows.append(trial_row(reg, sha, window, cfg, "ols_slope_move_on_gap", s, "negative"))
        X = np.column_stack([np.ones(len(fr)), fr.p, fr.q_open])
        coef = np.linalg.lstsq(X, fr.move.to_numpy(), rcond=None)[0] if len(fr) > 3 else [np.nan] * 3
        sharp = fr[np.isfinite(fr.q_close)]
        days = np.sort(fr.match_day.unique())
        tri = np.array_split(days, 3)
        detail["trials"][f"T-{hours}h"] = {
            "slope": s,
            "share_no_new_trade_before_ko": float(fr.same_trade.mean()) if len(fr) else None,
            "mean_abs_gap": float(fr.gap.abs().mean()) if len(fr) else None,
            "two_regressor_coef_const_pT_qopen": [float(c) for c in coef],
            "sharp_move_on_gap_slope": slope_summary(
                sharp.gap.to_numpy(), (sharp.q_close - sharp.q_open).to_numpy(), sharp.match_day.to_numpy()
            ),
            "excluding_no_new_trade": slope_summary(
                fr.gap[~fr.same_trade].to_numpy(),
                fr.move[~fr.same_trade].to_numpy(),
                fr.match_day[~fr.same_trade].to_numpy(),
            ),
            "thirds": [
                slope_summary(
                    fr.gap[fr.match_day.isin(t)].to_numpy(),
                    fr.move[fr.match_day.isin(t)].to_numpy(),
                    fr.match_day[fr.match_day.isin(t)].to_numpy(),
                )
                | {"from": str(t[0]), "to": str(t[-1])}
                for t in tri
            ],
            "by_side": {
                o: slope_summary(
                    fr.gap[fr.side == o].to_numpy(),
                    fr.move[fr.side == o].to_numpy(),
                    fr.match_day[fr.side == o].to_numpy(),
                )
                for o in SIDES
            },
        }
        big = fr.division.value_counts()
        detail["trials"][f"T-{hours}h"]["by_league"] = {
            lg: slope_summary(
                fr.gap[fr.division == lg].to_numpy(),
                fr.move[fr.division == lg].to_numpy(),
                fr.match_day[fr.division == lg].to_numpy(),
            )
            for lg in big[big >= 100].index
        }
    finish(HERE, REG_ID, window, rows, detail)


if __name__ == "__main__":
    main(sys.argv[1])
