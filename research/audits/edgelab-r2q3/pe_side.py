"""Pitch-edge's own numbers for the decisions edgelab was given: CLV and ROI by window.

Reads data/research/r2/edgelab/r2q3-<variant>-pe_bets.parquet (written by build_dataset.py: the study rule on the
eligible matches of the edgelab dataset) and scores CLV = log(odds * p_close) and unit-stake ROI with match-day
block bootstrap CIs, for the whole edgelab window, the search part edgelab scores (first 80% of the window) and
the study's evaluation window.

Usage: python pe_side.py <max|bet365>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "studies" / "r2-common"))
import r2lib as L  # noqa: E402

RANGE = ("2012-07-27", "2025-06-01")


def score(b: pd.DataFrame) -> dict:
    clv = np.log(b["odds"].to_numpy() * b["p_close"].to_numpy())
    roi = b["odds"].to_numpy() * b["y"].to_numpy() - 1
    c, r = L.boot_mean(clv, b["match_date"]), L.boot_mean(roi, b["match_date"])
    return {
        "n": len(b),
        "clv": [c["mean"], c["ci_lo"], c["ci_hi"]],
        "roi": [r["mean"], r["ci_lo"], r["ci_hi"]],
        "hit": float(b["y"].mean()),
        "expected_hit_close": float(b["p_close"].mean()),
        "mean_odds": float(b["odds"].mean()),
        "mean_edge_open": float(b["edge_open"].mean()),
    }


def main(variant: str) -> None:
    b = pd.read_parquet(L.DATA / "edgelab" / f"r2q3-{variant}-pe_bets.parquet")
    b["match_date"] = pd.to_datetime(b["match_date"])
    t0, t1 = pd.Timestamp(RANGE[0]), pd.Timestamp(RANGE[1])
    split = t0 + (t1 - t0) * 0.8
    out = {
        "split_80pct": str(split.date()),
        "whole_window": score(b),
        "search_80pct": score(b[b["match_date"] <= split]),
        "holdout_20pct_not_scored_by_edgelab": score(b[b["match_date"] > split]),
        "study_eval_2019_07_to_2025_06": score(b[b["match_date"] >= "2019-07-01"]),
        "exploratory_2012_07_to_2019_06": score(b[b["match_date"] < "2019-07-01"]),
        "kickoff_recorded_only": score(b[b["ko_recorded"]]),
    }
    path = L.DATA / "edgelab" / f"r2q3-{variant}-pe_side.json"
    path.write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        if isinstance(v, dict):
            print(
                f"{k:42s} n={v['n']:6d} CLV {v['clv'][0] * 100:+.2f}% [{v['clv'][1] * 100:+.2f}, {v['clv'][2] * 100:+.2f}]"
                f"  ROI {v['roi'][0] * 100:+.2f}% [{v['roi'][1] * 100:+.2f}, {v['roi'][2] * 100:+.2f}]  hit {v['hit']:.3f} vs {v['expected_hit_close']:.3f}"
            )


if __name__ == "__main__":
    main(sys.argv[1])
