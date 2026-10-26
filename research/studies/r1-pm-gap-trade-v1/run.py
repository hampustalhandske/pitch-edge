"""Q3 (r1-pm-gap-trade-v1): paper strategy buying Polymarket outcomes priced away from the de-vigged sharp open.

At decision time ts = kickoff - T (only when the football-data open is known by ts), for each outcome with a
fresh last trade p:
  yes side: cost c = p + half_spread + fee, buy when q_open - c > theta; settles 1 if the outcome happens.
  no side:  cost c = (1 - p) + half_spread + fee, buy when (1 - q_open) - c > theta; settles 1 otherwise.
half_spread = max(bar (high - low)/2, 0.005); fee = 0.05 * p * (1 - p) per share.
Primary: CLV vs the de-vigged Betfair Exchange close, log(q_close_side / c). Secondary: CLV vs the last
Polymarket trade before kickoff, ROI on resolution (per $1 staked). One trial per (T, theta, side).

Usage: arch -arm64 uv run python research/studies/r1-pm-gap-trade-v1/run.py {exploratory|evaluation}
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sst

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "r1-common"))

from common import FEE_RATE, MIN_HALF_SPREAD, SIDES, dsr_threshold, pbo_cscv, psr  # noqa: E402
from runner import by_league, finish, start, summarise, thirds, trial_row  # noqa: E402

REG_ID = "r1-pm-gap-trade-v1"


def candidates(df: pd.DataFrame, book, hours: float, max_stale: float) -> pd.DataFrame:
    """One row per (match, outcome) priceable at kickoff - hours, with the open known by then."""
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
            p, hr, stale, kn, _ = book.at(cid, ts)
            if not np.isfinite(p) or stale > max_stale:
                continue
            assert kn <= ts.value and r.open_known_at <= ts, "input not known at decision time"
            p_last, _, stale_last, _, _ = book.at(cid, r.ko)
            recs.append(
                {
                    "fd_match_key": r.fd_match_key,
                    "match_day": r.match_day,
                    "division": r.division,
                    "side": s,
                    "ts": ts,
                    "p": p,
                    "hs": max(hr, MIN_HALF_SPREAD),
                    "q_open": qo,
                    "q_close": getattr(r, f"q_bf_close_{s}"),
                    "p_last": p_last,
                    "stale_last": stale_last,
                    "won": float(r.result_side == s) if isinstance(r.result_side, str) else np.nan,
                }
            )
    return pd.DataFrame(recs)


def decisions(c: pd.DataFrame, theta: float, side: str) -> pd.DataFrame:
    fee = FEE_RATE * c.p * (1 - c.p)
    if side == "yes":
        cost = c.p + c.hs + fee
        edge = c.q_open - cost
        fair_close, fair_last, payoff = c.q_close, c.p_last, c.won
    else:
        cost = (1 - c.p) + c.hs + fee
        edge = (1 - c.q_open) - cost
        fair_close, fair_last, payoff = 1 - c.q_close, 1 - c.p_last, 1 - c.won
    d = c.assign(cost=cost, edge=edge, fair_close=fair_close, fair_last=fair_last, payoff=payoff)
    d = d[(d.edge > theta) & (d.cost > 0) & (d.cost < 1)]
    d = d.assign(
        clv_close=np.log(d.fair_close / d.cost),
        clv_pm_last=np.log(d.fair_last / d.cost),
        roi=d.payoff / d.cost - 1,
    )
    return d.sort_values("ts")


def main(window: str) -> None:
    reg, sha, df, book, drops = start(REG_ID, window)
    f = reg["filters"]
    g = reg["grid"]
    rows, detail = [], {"drops": drops, "window": window, "trials": {}}
    all_days = np.sort(df.match_day.unique())
    blocks = np.array_split(all_days, 16)
    perf, srs, series = [], [], {}
    for hours in g["hours_before_kickoff"]:
        cand = candidates(df, book, float(hours), f["max_stale_hours"])
        for theta in g["theta"]:
            for side_raw in g["side"]:
                # YAML 1.1 reads the registered `[yes, no]` as booleans; map them back to the side names.
                side = {True: "yes", False: "no"}.get(side_raw, side_raw)
                assert side in ("yes", "no")
                cfg = {"hours_before_kickoff": hours, "theta": theta, "side": side, **f}
                d = decisions(cand, float(theta), side)
                prim = d[np.isfinite(d.clv_close)]
                s = summarise(prim.clv_close.to_numpy(), prim.match_day.to_numpy(), "positive")
                rows.append(trial_row(reg, sha, window, cfg, "clv_log_vs_bf_close", s, "positive"))
                key = f"T-{hours}h_theta{theta}_{side}"
                pl = d[np.isfinite(d.clv_pm_last) & (d.stale_last <= f["max_stale_last_hours"])]
                rs = d[np.isfinite(d.roi)]
                x = prim.clv_close.to_numpy()
                sr = float(x.mean() / x.std(ddof=1)) if len(x) > 2 and x.std() > 0 else np.nan
                srs.append(sr)
                series[key] = x
                detail["trials"][key] = {
                    "n_candidates": len(cand),
                    "n_decisions": len(d),
                    "n_with_bf_close": len(prim),
                    "clv_vs_bf_close": s,
                    "clv_vs_pm_last": summarise(pl.clv_pm_last.to_numpy(), pl.match_day.to_numpy(), "positive"),
                    "roi_after_fees": summarise(rs.roi.to_numpy(), rs.match_day.to_numpy(), "positive"),
                    "mean_edge_at_entry": float(d.edge.mean()) if len(d) else None,
                    "mean_cost": float(d.cost.mean()) if len(d) else None,
                    "share_cost_from_spread_and_fee": float(((d.hs + FEE_RATE * d.p * (1 - d.p)) / d.cost).mean())
                    if len(d)
                    else None,
                    "by_outcome": {
                        o: summarise(
                            prim.clv_close[prim.side == o].to_numpy(),
                            prim.match_day[prim.side == o].to_numpy(),
                            "positive",
                        )
                        for o in SIDES
                    },
                    "per_share_sr": sr,
                    "skew": float(sst.skew(x)) if len(x) > 2 else None,
                    "kurtosis": float(sst.kurtosis(x, fisher=False)) if len(x) > 2 else None,
                    "thirds": thirds(prim.clv_close.to_numpy(), prim.match_day.to_numpy(), "positive"),
                    "by_league": by_league(
                        prim.clv_close.to_numpy(), prim.match_day.to_numpy(), prim.division.to_numpy(), "positive"
                    ),
                }
                perf.append([prim.clv_close[prim.match_day.isin(b)].mean() for b in blocks])
    # search-survival statistics across this grid (ledger trial count = grid size here)
    sr0 = dsr_threshold(np.array(srs))
    for key, x in series.items():
        t = detail["trials"][key]
        if len(x) > 2 and np.isfinite(t["per_share_sr"]):
            t["dsr"] = psr(t["per_share_sr"], len(x), t["skew"] or 0.0, t["kurtosis"] or 3.0, sr0)
    perf_m = np.array(perf).T  # (16 blocks, n configs)
    detail["sr0_expected_max_null"] = sr0
    detail["pbo_cscv"] = pbo_cscv(perf_m, 16)
    print("PBO", detail["pbo_cscv"], "SR0", sr0)
    finish(HERE, REG_ID, window, rows, detail)


if __name__ == "__main__":
    main(sys.argv[1])
