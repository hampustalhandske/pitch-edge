"""Data-quality report for the R1 polymarket-efficiency family (no outcome or closing price is scored here).

Prints and writes data/research/r1/dq.json:
- drop log from `load_matches` (groups without exactly one market per outcome, unverified kickoffs)
- kickoff verification: football-data kickoff (Europe/London -> UTC) vs Polymarket close_at
- coverage of Pinnacle / Betfair Exchange open and close by window and division
- Polymarket bar coverage before kickoff and staleness of the last trade at each decision offset
- volume distribution (cumulative USDC traded on the three markets before kickoff)
- open-odds timing check (exploratory window only): distance between Polymarket's normalised price and the
  de-vigged sharp open as a function of time relative to the assumed collection time `open_known_at`.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from common import DATA, SIDES, PriceBook, load_matches, pm_snapshot, power_normalise

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)


def main() -> None:
    out: dict = {}
    df, drops = load_matches()
    out["drops"] = drops
    print("drops", drops)
    bars = pd.read_parquet(DATA / "bars.parquet")
    print("bars", len(bars), bars.source.value_counts().to_dict(), bars.outcome_label.value_counts().to_dict())
    book = PriceBook(bars)

    # coverage
    cov = df.assign(
        bf_open=df[[f"q_bf_open_{s}" for s in SIDES]].notna().all(axis=1),
        bf_close=df[[f"q_bf_close_{s}" for s in SIDES]].notna().all(axis=1),
        pin_open=df[[f"q_pin_open_{s}" for s in SIDES]].notna().all(axis=1),
        pin_close=df[[f"q_pin_close_{s}" for s in SIDES]].notna().all(axis=1),
        bf_open_ok=df.bs_bf_open <= 1.05,
        bf_close_ok=df.bs_bf_close <= 1.05,
    )
    t = cov.groupby("window")[["bf_open", "bf_open_ok", "bf_close", "bf_close_ok", "pin_open", "pin_close"]].sum()
    t["n"] = cov.groupby("window").size()
    print(t)
    out["coverage_by_window"] = t.to_dict()
    td = cov.groupby("division")[["bf_open_ok", "bf_close_ok", "pin_open", "pin_close"]].sum()
    td["n"] = cov.groupby("division").size()
    print(td)
    out["coverage_by_division"] = td.to_dict()

    # bars before kickoff / staleness
    stale = {}
    for h in (48, 24, 6, 2, 1, 0):
        snap = pm_snapshot(df, book, h)
        st = snap[[f"stale_{s}" for s in SIDES]]
        allp = snap[[f"p_{s}" for s in SIDES]].notna().all(axis=1)
        maxst = st.max(axis=1)
        sums = snap[[f"p_{s}" for s in SIDES]].sum(axis=1)
        rec = {
            "all_three_priced": float(allp.mean()),
            "max_stale_le_6h": float((maxst <= 6).mean()),
            "max_stale_le_24h": float((maxst <= 24).mean()),
            "median_max_stale_h": float(maxst.median()),
            "raw_sum_p05_p50_p95": [float(x) for x in sums[allp].quantile([0.05, 0.5, 0.95])],
            "raw_sum_in_0.9_1.1": float(sums[allp].between(0.9, 1.1).mean()),
        }
        for w in ("exploratory", "evaluation"):
            m = (df.window == w).to_numpy()
            rec[f"{w}_all_priced_stale_le_24h"] = int((allp & (maxst <= 24))[m].sum())
            rec[f"{w}_all_priced_stale_le_6h"] = int((allp & (maxst <= 6))[m].sum())
        stale[f"T-{h}h"] = rec
        print(f"T-{h}h", rec)
    out["staleness"] = stale

    # volume (known at kickoff; descriptive only)
    snap0 = pm_snapshot(df, book, 0)
    vol = snap0[[f"vol_{s}" for s in SIDES]].sum(axis=1)
    out["pre_ko_volume_usdc_quantiles"] = {
        str(q): float(v) for q, v in vol.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).items()
    }
    vt = df[["window", "pm_source", "division"]].assign(vol=vol.values)
    print("volume quantiles", out["pre_ko_volume_usdc_quantiles"])
    print(vt.groupby("window").vol.describe())
    print(vt.groupby("division").vol.median().sort_values())
    out["volume_by_window_median"] = vt.groupby("window").vol.median().to_dict()

    # open timing check, exploratory window only
    ex = df[(df.window == "exploratory") & df[[f"q_open_{s}" for s in SIDES]].notna().all(axis=1)].reset_index(
        drop=True
    )
    qopen = ex[[f"q_open_{s}" for s in SIDES]].to_numpy()
    res = []
    for off in range(-48, 25, 3):
        tmp = ex.assign(ts=ex.open_known_at + pd.Timedelta(hours=off))
        tmp = tmp[tmp.ts < tmp.ko]
        if len(tmp) < 50:
            continue
        sn = pm_snapshot(tmp, book, None, col="ts")
        p = sn[[f"p_{s}" for s in SIDES]].to_numpy()
        fresh = (sn[[f"stale_{s}" for s in SIDES]].max(axis=1) <= 6).to_numpy() & np.isfinite(p).all(axis=1)
        pn = power_normalise(p)
        d = np.abs(pn - qopen[tmp.index]).sum(axis=1)
        ok = fresh & np.isfinite(d)
        res.append({"offset_h": off, "n": int(ok.sum()), "mean_abs_dist": float(d[ok].mean()) if ok.any() else None})
    print(pd.DataFrame(res))
    out["open_timing_check_exploratory"] = res

    # PM resolution vs football-data result (exploratory only)
    exr = df[df.window == "exploratory"]
    agree = []
    for r in exr.itertuples():
        lab = {s: getattr(r, f"pm_win_{s}") for s in SIDES}
        yes = [s for s in SIDES if str(lab[s]).lower() == "yes"]
        agree.append(len(yes) == 1 and yes[0] == r.result_side)
    out["pm_resolution_agrees_with_fd_exploratory"] = float(np.mean(agree))
    print("PM resolution agrees with fd result (exploratory):", np.mean(agree))
    (DATA / "dq.json").write_text(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
