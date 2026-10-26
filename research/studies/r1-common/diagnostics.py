"""Post-hoc diagnostics for the R1 family (labelled post-hoc in the reports; not registered trials).

1. Bounce check for r1-pm-convergence-v1: Roll's estimator of last-trade noise, sigma_e^2 = -cov(dp_t, dp_t+1)
   on consecutive hourly bar-to-bar price changes per Polymarket market (evaluation window, pre-kickoff), and
   the slope a pure bounce would produce in move-on-gap regressions, -sigma_e^2 / var(gap).
2. De-vig sensitivity for r1-pm-bias-v1: longshot / favourite gaps with the Betfair close de-vigged
   multiplicatively instead of by the power method.
3. Open staleness: how far the de-vigged sharp open sits from the Betfair close, by window (evaluation).
Writes data/research/r1/diagnostics.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from common import DATA, SIDES, PriceBook, block_bootstrap, load_matches, pm_probs

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "r1-pm-convergence-v1"))


def roll_noise(df: pd.DataFrame, bars: pd.DataFrame) -> float:
    """Pooled Roll estimator of last-trade noise variance on hourly-consecutive Yes-equivalent prices."""
    lab = bars.outcome_label.str.lower()
    b = bars[lab.isin(["yes", "no"])].copy()
    yes = b.outcome_label.str.lower().eq("yes")
    b["p"] = np.where(yes, b.close, 1 - b.close)
    b = b.sort_values(["condition_id", "hour_start", "outcome_label"], ascending=[True, True, False])
    b = b.drop_duplicates(["condition_id", "hour_start"])
    ko = pd.concat([df[["ko", f"cond_{s}"]].rename(columns={f"cond_{s}": "condition_id"}) for s in SIDES])
    b = b.merge(ko, on="condition_id")
    b = b[b.hour_start + pd.Timedelta(hours=1) <= b.ko]
    covs, n = [], 0
    for _, g in b.groupby("condition_id"):
        h = g.hour_start.to_numpy()
        p = g.p.to_numpy()
        if len(p) < 4:
            continue
        consec = np.diff(h) == np.timedelta64(1, "h")
        dp = np.diff(p)
        pair = consec[:-1] & consec[1:]
        if pair.sum() == 0:
            continue
        covs.append(np.sum(dp[:-1][pair] * dp[1:][pair]))
        n += pair.sum()
    return float(-np.sum(covs) / n), int(n)


def main() -> None:
    out: dict = {}
    df_all, _ = load_matches()
    df = df_all[df_all.window == "evaluation"].reset_index(drop=True)
    bars = pd.read_parquet(DATA / "bars.parquet")
    book = PriceBook(bars)
    s2, npairs = roll_noise(df, bars)
    out["roll_noise_var"] = s2
    out["roll_noise_sd"] = float(np.sqrt(max(s2, 0)))
    out["roll_pairs"] = npairs
    from run import frame  # r1-pm-convergence-v1

    for h in (24, 6, 2):
        fr = frame(df, book, float(h), 6.0, 3.0)
        vg = float(fr.gap.var())
        out[f"T-{h}h"] = {"var_gap": vg, "bounce_only_slope": -s2 / vg, "n": len(fr)}
    # de-vig sensitivity for bias
    pn, ok = pm_probs(df, book, 1.0, 24, 0.9, 1.1, "power")
    odds = df[[f"betfair_ex_{s}_close" for s in SIDES]].to_numpy()
    imp = 1 / odds
    qm = imp / imp.sum(axis=1, keepdims=True)
    qp = df[[f"q_bf_close_{s}" for s in SIDES]].to_numpy()
    valid = ok & np.isfinite(qp).all(axis=1)
    days = np.repeat(df.match_day.to_numpy()[valid], 3)
    gap_m = (pn[valid] - qm[valid]).ravel()
    q_m = qm[valid].ravel()
    res = {}
    for name, m in {"longshot": q_m < 0.2, "favourite": q_m > 0.6}.items():
        bb = block_bootstrap(gap_m[m], days[m])
        res[name] = {
            "n": int(m.sum()),
            "mean_gap": float(gap_m[m].mean()),
            "ci": np.percentile(bb, [2.5, 97.5]).tolist(),
        }
    out["bias_multiplicative_bf_devig"] = res
    # open staleness: |q_open - q_bf_close| summed over outcomes
    qo = df[[f"q_open_{s}" for s in SIDES]].to_numpy()
    d = np.abs(qo - qp).sum(axis=1)
    out["open_vs_bf_close_sum_abs_median"] = float(np.nanmedian(d))
    print(json.dumps(out, indent=2))
    (DATA / "diagnostics.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
