"""r5-pm-micro-accuracy-v1: Polymarket order-book-mid 1X2 forecast vs the de-vigged Betfair close, log-loss.

Per match at kickoff - h: value = LL(PM mid) - LL(Betfair close) in bits on the football-data result
(positive = PM worse). Secondaries on the same grid point: R1's last-trade version on the matches where both
prices exist, and the paired mid-minus-last-trade log-loss.

Usage: arch -arm64 uv run python research/studies/r5-pm-micro-accuracy-v1/run.py [--counts]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "r5-common"))

from harness import record  # noqa: E402
from parta import (  # noqa: E402
    CODE_FILES,
    R5,
    SIDES,
    by_league,
    lasttrade_probs,
    leave_one_league_out,
    load,
    mid_probs,
    snapshot_tag,
    subperiods,
)

REG_ID = "r5-pm-micro-accuracy-v1"


def ll_bits(p: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return -np.log2(np.clip(p[np.arange(len(p)), idx], 1e-12, 1.0))


def main(counts_only: bool) -> None:
    df, pbook, books, info = load()
    q = df[[f"q_bf_close_{s}" for s in SIDES]].to_numpy()
    res_idx = df.result_side.map({s: j for j, s in enumerate(SIDES)})
    base = np.isfinite(q).all(axis=1) & res_idx.notna().to_numpy()
    snap = None if counts_only else snapshot_tag()
    detail: dict = {"info": info, "trials": {}}
    for h in (1, 24):
        pn, ok, _ = mid_probs(df, books, float(h), 24.0, "mid")
        pl, okl = lasttrade_probs(df, pbook, float(h))
        m = ok & base
        if counts_only:
            print(
                f"T-{h}h: matches mid ok {int(m.sum())}, with last trade {int((m & okl).sum())}, days {df.match_day[m].nunique()}"
            )
            continue
        sub = df[m].reset_index(drop=True)
        ri = res_idx[m].astype(int).to_numpy()
        v = ll_bits(pn[m], ri) - ll_bits(q[m], ri)
        days = sub.match_day.to_numpy()

        def mean_on(mm: np.ndarray, v=v) -> float:
            return float(np.mean(v[mm]))

        extra = {
            "clean": True,
            "within_ceiling": bool(abs(v.mean()) <= 0.10),
            "subperiod_estimates": subperiods(v, days, mean_on),
            "leave_one_league_out": leave_one_league_out(sub.division.to_numpy(), mean_on),
            "by_league": by_league(sub.division.to_numpy(), mean_on),
        }
        cfg = {"hours_before_kickoff": h}
        frame = pd.DataFrame({"decision_id": sub.fd_match_key, "block": days, "value": v})
        r = record(
            REG_ID, cfg, frame, "delta_logloss_bits_mid_minus_bf_close", snap, extra, CODE_FILES + [Path(__file__)]
        )
        both = okl[m]
        if both.sum() > 1:
            vl = ll_bits(pl[m][both], ri[both]) - ll_bits(q[m][both], ri[both])
            vp = ll_bits(pn[m][both], ri[both]) - ll_bits(pl[m][both], ri[both])
            fr = pd.DataFrame({"decision_id": sub.fd_match_key[both], "block": days[both], "value": vl})
            record(
                REG_ID,
                cfg,
                fr,
                "delta_logloss_bits_lasttrade_minus_bf_close_same_matches",
                snap,
                {"clean": True},
                CODE_FILES + [Path(__file__)],
            )
            fr = pd.DataFrame({"decision_id": sub.fd_match_key[both], "block": days[both], "value": vp})
            record(
                REG_ID,
                cfg,
                fr,
                "paired_mid_minus_lasttrade_logloss_bits",
                snap,
                {"clean": True},
                CODE_FILES + [Path(__file__)],
            )
        detail["trials"][f"T-{h}h"] = {
            "value": None if r is None else r.metric_value,
            "n": int(m.sum()),
            "n_both": int(both.sum()),
            **extra,
        }
    if not counts_only:
        out = R5 / REG_ID
        out.mkdir(parents=True, exist_ok=True)
        (out / "detail.json").write_text(json.dumps(detail, indent=2, default=str))


if __name__ == "__main__":
    main("--counts" in sys.argv)
