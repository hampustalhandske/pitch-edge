"""Family-level grading for `bookmaker-line-structure` (STANDARDS.md L3 a/b).

Reads every R2 study's interim trials ledger, applies Benjamini-Hochberg (q = 0.10) across every trial in the
family that has a p-value, and computes the deflated Sharpe ratio for every return-like (CLV) trial using the
family's trial count. Two variances for the null Sharpe ratios are reported: `dsr_cross` uses the cross-trial
variance of per-bet Sharpe ratios (Bailey & Lopez de Prado's recipe; inflated here because many trials are
genuinely vig-negative rather than null), `dsr_null` uses the estimation variance 1/(n-1) of a zero-SR trial
with the same number of bets. Trials with n < 10 are kept in the BH denominator with p = 1 (their bootstrap is
degenerate). Writes
data/research/r2/family_grades.csv and prints the rows that reach q < 0.10.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r2lib as L  # noqa: E402

STUDIES = ["r2-q1-close-calibration", "r2-q2-open-consensus-gap", "r2-q3-stale-soft-open", "r2-q4-totals-open-gap"]


def main() -> None:
    frames = []
    for s in STUDIES:
        p = L.REPO / "research" / "studies" / s / "trials.csv"
        if p.exists():
            t = pd.read_csv(p)
            t["study"] = s
            frames.append(t)
    t = pd.concat(frames, ignore_index=True)
    p = t["p_value"].to_numpy(dtype=float).copy()
    p[(t["n"] < 10).to_numpy() & np.isfinite(p)] = 1.0
    t["p_used"] = p
    t["family_q"], t["family_bh_reject"] = L.bh(p, q=0.10)
    ret = t["metric_name"].eq("clv_log_mean") & t["sr"].notna()
    n_trials = int(t.loc[ret, "config_hash"].nunique())
    var_sr = float(t.loc[ret, "sr"].var(ddof=1))
    t["dsr_cross"] = np.nan
    t["dsr_null"] = np.nan
    for i in t.index[ret]:
        r = t.loc[i]
        t.loc[i, "dsr_cross"] = L.dsr(r["sr"], int(r["n"]), r["skew"], r["kurt"], n_trials, var_sr)
        t.loc[i, "dsr_null"] = L.dsr(r["sr"], int(r["n"]), r["skew"], r["kurt"], n_trials, 1 / max(int(r["n"]) - 1, 1))
    t.to_csv(L.DATA / "family_grades.csv", index=False)
    print(
        f"family trials with p-values: {t['p_value'].notna().sum()}  return-like trials: {n_trials}  var(SR): {var_sr:.5f}"
    )
    keep = [
        "study",
        "window",
        "label",
        "metric_name",
        "n",
        "metric",
        "ci_lo",
        "ci_hi",
        "p_used",
        "family_q",
        "dsr_cross",
        "dsr_null",
        "config_json",
    ]
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 90)
    print(t.loc[t["family_bh_reject"], keep].to_string(index=False))


if __name__ == "__main__":
    main()
