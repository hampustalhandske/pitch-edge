"""r5-pm-micro-flb-v1: R1's favourite-longshot gap re-measured at the Polymarket order-book mid.

Per outcome at kickoff - 1h: gap = p_mid - q_close (power-normalised PM mid minus power de-vigged Betfair
close); FLB-signed value s * gap with s = +1 for longshots (q_close < 0.20), -1 for favourites (> 0.60).
One counted trial per segment (primary), plus ledger-recorded secondaries on the same grid point: R1's
last-trade gap on the rows where both prices exist, the paired last-trade-minus-mid difference, and the
depth-weighted microprice gap. The draw gap is descriptive (stored in `extra`).

Usage: arch -arm64 uv run python research/studies/r5-pm-micro-flb-v1/run.py [--counts]
  --counts  build the frames and print row counts only (no gap, no close-bearing metric is computed)
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

REG_ID = "r5-pm-micro-flb-v1"
H = 1.0
SEGMENTS = {"longshot": (lambda q: q < 0.20, 1.0), "favourite": (lambda q: q > 0.60, -1.0)}


def long_frame(
    df: pd.DataFrame,
    pn: np.ndarray,
    ok: np.ndarray,
    plast: np.ndarray,
    oklast: np.ndarray,
    pmicro: np.ndarray,
    okmicro: np.ndarray,
) -> pd.DataFrame:
    q = df[[f"q_bf_close_{s}" for s in SIDES]].to_numpy()
    okq = np.isfinite(q).all(axis=1)
    recs = []
    for j, s in enumerate(SIDES):
        m = ok & okq
        recs.append(
            pd.DataFrame(
                {
                    "decision_id": df.fd_match_key[m].to_numpy() + "|" + s,
                    "block": df.match_day[m].to_numpy(),
                    "division": df.division[m].to_numpy(),
                    "side": s,
                    "q": q[m, j],
                    "p_mid": pn[m, j],
                    "p_last": np.where(oklast[m], plast[m, j], np.nan),
                    "p_micro": np.where(okmicro[m], pmicro[m, j], np.nan),
                }
            )
        )
    return pd.concat(recs, ignore_index=True)


def main(counts_only: bool) -> None:
    df, pbook, books, info = load()
    pn, ok, _ = mid_probs(df, books, H, 24.0, "mid")
    pm_, okm, _ = mid_probs(df, books, H, 24.0, "micro")
    pl, okl = lasttrade_probs(df, pbook, H)
    lf = long_frame(df, pn, ok, pl, okl, pm_, okm)
    info.update(
        {
            "matches_mid_ok": int(ok.sum()),
            "matches_mid_and_bf_ok": int(lf.decision_id.str.rsplit("|", n=1).str[0].nunique()),
            "matches_lasttrade_ok": int(okl.sum()),
            "matches_micro_ok": int(okm.sum()),
        }
    )
    seg_rows = {}
    for name, (sel, _sgn) in SEGMENTS.items():
        seg_rows[name] = lf[sel(lf.q)]  # every side, as r1-pm-bias-v1
    if counts_only:
        print(json.dumps(info, indent=1))
        for name, sub in seg_rows.items():
            print(
                name,
                "rows",
                len(sub),
                "days",
                sub.block.nunique(),
                "with last",
                int(sub.p_last.notna().sum()),
                "with micro",
                int(sub.p_micro.notna().sum()),
            )
        return
    snap = snapshot_tag()
    detail: dict = {"info": info, "trials": {}}
    for name, (_sel, sgn) in SEGMENTS.items():
        sub = seg_rows[name].reset_index(drop=True)
        v = sgn * (sub.p_mid - sub.q).to_numpy()
        days = sub.block.to_numpy()

        def mean_on(m: np.ndarray, v=v) -> float:
            return float(np.mean(v[m]))

        extra = {
            "clean": True,
            "within_ceiling": bool(abs(v.mean()) <= 0.02),
            "subperiod_estimates": subperiods(v, days, mean_on),
            "leave_one_league_out": leave_one_league_out(sub.division.to_numpy(), mean_on),
            "by_league": by_league(sub.division.to_numpy(), mean_on),
            "mean_q": float(sub.q.mean()),
            "mean_p_mid": float(sub.p_mid.mean()),
            "info": info,
        }
        cfg = {"segment": name}
        frame = pd.DataFrame({"decision_id": sub.decision_id, "block": days, "value": v})
        res = record(REG_ID, cfg, frame, "mean_flb_signed_gap_mid", snap, extra, CODE_FILES + [Path(__file__)])
        out = {"mid": None if res is None else res.metric_value}
        both = sub[sub.p_last.notna()].reset_index(drop=True)
        if len(both) > 1:
            vl = sgn * (both.p_last - both.q).to_numpy()
            vp = sgn * (both.p_last - both.p_mid).to_numpy()
            fr = pd.DataFrame({"decision_id": both.decision_id, "block": both.block, "value": vl})
            record(
                REG_ID,
                cfg,
                fr,
                "mean_flb_signed_gap_lasttrade_same_rows",
                snap,
                {"clean": True},
                CODE_FILES + [Path(__file__)],
            )
            fr = pd.DataFrame({"decision_id": both.decision_id, "block": both.block, "value": vp})
            record(
                REG_ID,
                cfg,
                fr,
                "paired_lasttrade_minus_mid_flb_signed",
                snap,
                {"clean": True},
                CODE_FILES + [Path(__file__)],
            )
        mic = sub[sub.p_micro.notna()].reset_index(drop=True)
        if len(mic) > 1:
            vm = sgn * (mic.p_micro - mic.q).to_numpy()
            fr = pd.DataFrame({"decision_id": mic.decision_id, "block": mic.block, "value": vm})
            record(
                REG_ID, cfg, fr, "mean_flb_signed_gap_microprice", snap, {"clean": True}, CODE_FILES + [Path(__file__)]
            )
        detail["trials"][name] = {
            **out,
            "n": len(sub),
            "n_with_last": len(both),
            "n_with_micro": len(mic),
            **{k: extra[k] for k in ("subperiod_estimates", "leave_one_league_out", "by_league")},
        }
    # Descriptive: draw and bucket table at the mid (not trials).
    from pitch_edge.research.stats import block_bootstrap

    dr = lf[lf.side == "draw"]
    b = block_bootstrap((dr.p_mid - dr.q).to_numpy(), dr.block.to_numpy(), n_boot=5000, seed=20261020)
    detail["draw_gap_mid"] = {"est": b.estimate, "lo": b.lo, "hi": b.hi, "n": b.n}
    ha = lf[lf.side != "draw"].copy()
    ha["bucket"] = pd.cut(ha.q, [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0], include_lowest=True).astype(str)
    detail["bucket_table_home_away"] = (
        ha.assign(g_mid=ha.p_mid - ha.q, g_last=ha.p_last - ha.q)
        .groupby("bucket")
        .agg(n=("q", "size"), g_mid=("g_mid", "mean"), g_last=("g_last", "mean"), n_last=("p_last", "count"))
        .reset_index()
        .to_dict(orient="records")
    )
    out = R5 / REG_ID
    out.mkdir(parents=True, exist_ok=True)
    (out / "detail.json").write_text(json.dumps(detail, indent=2, default=str))
    print(json.dumps({k: detail[k] for k in ("draw_gap_mid", "bucket_table_home_away")}, indent=1, default=str))


if __name__ == "__main__":
    main("--counts" in sys.argv)
