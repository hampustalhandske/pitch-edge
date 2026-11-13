"""Q1 (r1-pm-accuracy-v1): is Polymarket's 1X2 price less accurate than the de-vigged sharp benchmark?

For each registered comparison (PM at kickoff - h vs a benchmark), the per-match value is the paired
log-loss difference in bits, LL(PM) - LL(benchmark), on the realised football-data result. Positive means
Polymarket was worse. Uncertainty: match-day block bootstrap.

Usage: arch -arm64 uv run python research/studies/r1-pm-accuracy-v1/run.py {exploratory|evaluation}
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

REG_ID = "r1-pm-accuracy-v1"


def bench_probs(df: pd.DataFrame, bench: str) -> np.ndarray:
    tag = {"betfair_close": "q_bf_close", "pinnacle_close": "q_pin_close", "sharp_open": "q_open"}[bench]
    return df[[f"{tag}_{s}" for s in SIDES]].to_numpy()


def paired(df, pn, q, ok):
    idx = df.result_side.map({s: i for i, s in enumerate(SIDES)}).to_numpy()
    ok = ok & np.isfinite(q).all(axis=1) & pd.notna(idx)
    r = idx[ok].astype(int)
    rows = np.arange(ok.sum())
    pm_p = np.clip(pn[ok][rows, r], 1e-9, 1)
    b_p = np.clip(q[ok][rows, r], 1e-9, 1)
    ll = -np.log2(pm_p) + np.log2(b_p)
    onehot = np.eye(3)[r]
    brier = ((pn[ok] - onehot) ** 2).sum(axis=1) - ((q[ok] - onehot) ** 2).sum(axis=1)
    return ok, ll, brier


def main(window: str) -> None:
    reg, sha, df, book, drops = start(REG_ID, window)
    f = reg["filters"]
    rows, detail = [], {"drops": drops, "window": window, "trials": {}}
    for comp in reg["grid"]["comparison"]:
        h, bench = float(comp["pm_hours_before_kickoff"]), comp["benchmark"]
        cfg = {"pm_hours_before_kickoff": h, "benchmark": bench, **f}
        pn, ok = pm_probs(df, book, h, f["max_stale_hours"], f["raw_sum_min"], f["raw_sum_max"], "power")
        q = bench_probs(df, bench)
        okk, ll, brier = paired(df, pn, q, ok)
        days = df.match_day.to_numpy()[okk]
        s = summarise(ll, days, "positive")
        rows.append(trial_row(reg, sha, window, cfg, "logloss_diff_bits_pm_minus_bench", s, "positive"))
        d = {
            "n_candidates": len(df),
            "n_pm_ok": int(ok.sum()),
            "n_scored": int(okk.sum()),
            "logloss_bits": s,
            "brier_diff": summarise(brier, days, "positive"),
            "thirds": thirds(ll, days, "positive"),
            "by_league": by_league(ll, days, df.division.to_numpy()[okk], "positive"),
            "sensitivity": {},
        }
        # sensitivities (secondary, not trials)
        for name, (stale, method) in {
            "no_staleness_filter": (None, "power"),
            "stale_le_6h": (6.0, "power"),
            "multiplicative_normalisation": (f["max_stale_hours"], "mult"),
        }.items():
            pn2, ok2 = pm_probs(df, book, h, stale, f["raw_sum_min"], f["raw_sum_max"], method)
            o2, ll2, _ = paired(df, pn2, q, ok2)
            d["sensitivity"][name] = summarise(ll2, df.match_day.to_numpy()[o2], "positive")
        if bench == "betfair_close":
            qp = bench_probs(df, "pinnacle_close")
            o3, ll3, _ = paired(df, pn, qp, ok)
            d["sensitivity"]["pinnacle_close_benchmark"] = summarise(ll3, df.match_day.to_numpy()[o3], "positive")
            # Pinnacle vs Betfair close on the same matches: how far apart are the two "sharp" closes?
            both = o3 & okk
            _, llpb, _ = paired(df, qp, q, both)
            d["sensitivity"]["pinnacle_close_minus_betfair_close_bits"] = summarise(
                llpb, df.match_day.to_numpy()[both], "two-sided"
            )
        if bench == "sharp_open":
            lead = ((df.ko - df.open_known_at).dt.total_seconds() / 3600).to_numpy()[okk]
            d["open_known_lead_hours_quantiles"] = np.percentile(lead, [5, 25, 50, 75, 95]).tolist()
            d["open_source_counts"] = df.open_src[okk].value_counts().to_dict()
        detail["trials"][f"T-{h:g}h_vs_{bench}"] = d
    finish(HERE, REG_ID, window, rows, detail)


if __name__ == "__main__":
    main(sys.argv[1])
