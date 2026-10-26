"""H3 checks: re-derive the per-book slopes and the VC-IW contrast, and test whether BW's 'anti-informative'
partial slope (-0.26) is an artefact of controlling for g_avg (the aggregator average, which contains BW and,
possibly, bwin-feed clones).

C1  Re-derive marginal / partial slopes per book and the registered contrast (VC - IW, common rows).
C2  Effective weights: OLS of g_avg on the five named books' deviations (evaluation rows with all five present).
    A large BW weight means the aggregator consensus over-represents bwin's line.
C3  Partial slope of each book controlling for a leave-one-out named consensus (mean dev of the other four named books)
    instead of g_avg; and controlling for both.
C4  Joint regression: move ~ all five named devs (+/- g_avg) on rows where all five are present.
C5  BW partial by season, and the g_avg coefficient that accompanies it (implied weight = -slope_BW / coef_g_avg).

Usage: OMP_NUM_THREADS=1 arch -arm64 uv run python research/audits/r6/h3check.py
"""

from __future__ import annotations

import json

import alib as A
import numpy as np
from timing import coef_ci, long


def main() -> None:
    d = A.universe(A.raw())
    r = A.core_frame(d)
    f = long(d, r)
    ev = f[f.ss.between(2019, 2024)].copy()
    res: dict = {"C1": {}}
    for b in A.CORE + A.LATE:
        g = ev[np.isfinite(ev["x_" + b]) & (ev["x_" + b].abs() <= 0.5) & np.isfinite(ev.y)]
        if b in A.LATE:
            g = g[g.ss == 2024]
        one = np.ones(len(g))
        res["C1"][b] = {
            "marginal": coef_ci(np.column_stack([one, g["x_" + b]]), g.y.to_numpy(), g.date.to_numpy()),
            "partial": coef_ci(np.column_stack([one, g["x_" + b], g.g_avg]), g.y.to_numpy(), g.date.to_numpy()),
        }
    both = ev[
        np.isfinite(ev.x_VC)
        & np.isfinite(ev.x_IW)
        & (ev.x_VC.abs() <= 0.5)
        & (ev.x_IW.abs() <= 0.5)
        & np.isfinite(ev.y)
    ]
    one = np.ones(len(both))
    for kind in ("marginal", "partial"):
        ex = [both.g_avg.to_numpy()] if kind == "partial" else []
        cv = np.linalg.lstsq(np.column_stack([one, both.x_VC, *ex]), both.y, rcond=None)[0][1]
        ci_ = np.linalg.lstsq(np.column_stack([one, both.x_IW, *ex]), both.y, rcond=None)[0][1]
        res["C1"][f"contrast_{kind}"] = {"value": float(cv - ci_), "n": len(both)}
    print(
        json.dumps(
            {
                k: (v if k.startswith("contrast") else {kk: [round(x, 4) for x in vv[:3]] for kk, vv in v.items()})
                for k, v in res["C1"].items()
            },
            indent=0,
        )
    )

    # rows with all five named books present and in range
    m = np.all([np.isfinite(ev["x_" + b]) & (ev["x_" + b].abs() <= 0.5) for b in A.CORE], axis=0) & np.isfinite(ev.y)
    a5 = ev[m].copy()
    X = np.column_stack([np.ones(len(a5))] + [a5["x_" + b] for b in A.CORE])
    w = np.linalg.lstsq(X, a5.g_avg, rcond=None)[0]
    r2 = 1 - np.var(a5.g_avg - X @ w) / np.var(a5.g_avg)
    res["C2_gavg_on_named"] = {
        "n": len(a5),
        "weights": dict(zip(A.CORE, map(float, w[1:]), strict=False)),
        "r2": float(r2),
        "corr_with_gavg": {b: float(np.corrcoef(a5["x_" + b], a5.g_avg)[0, 1]) for b in A.CORE},
    }
    by_season = {}
    for s, g in a5.groupby("ss"):
        Xs = np.column_stack([np.ones(len(g))] + [g["x_" + b] for b in A.CORE])
        by_season[int(s)] = dict(
            zip(A.CORE, np.round(np.linalg.lstsq(Xs, g.g_avg, rcond=None)[0][1:], 3).tolist(), strict=False)
        )
    res["C2_gavg_on_named"]["by_season"] = by_season
    print("C2", json.dumps(res["C2_gavg_on_named"], indent=0))

    # C3 leave-one-out named consensus
    res["C3"] = {}
    for b in A.CORE:
        others = [o for o in A.CORE if o != b]
        cons = a5[["x_" + o for o in others]].mean(1).to_numpy()
        one = np.ones(len(a5))
        x = a5["x_" + b].to_numpy()
        y = a5.y.to_numpy()
        dd = a5.date.to_numpy()
        res["C3"][b] = {
            "marginal": coef_ci(np.column_stack([one, x]), y, dd),
            "partial_gavg": coef_ci(np.column_stack([one, x, a5.g_avg]), y, dd),
            "partial_loo_named": coef_ci(np.column_stack([one, x, cons]), y, dd),
            "partial_both": coef_ci(np.column_stack([one, x, cons, a5.g_avg]), y, dd),
        }
        print("C3", b, {k: [round(z, 3) for z in v[:3]] for k, v in res["C3"][b].items()})
    one = np.ones(len(a5))
    Xj = np.column_stack([one] + [a5["x_" + b] for b in A.CORE])
    res["C4_joint_named"] = {b: coef_ci(Xj, a5.y.to_numpy(), a5.date.to_numpy(), j + 1) for j, b in enumerate(A.CORE)}
    Xg = np.column_stack([Xj, a5.g_avg])
    res["C4_joint_named_plus_gavg"] = {
        nm: coef_ci(Xg, a5.y.to_numpy(), a5.date.to_numpy(), j + 1) for j, nm in enumerate(list(A.CORE) + ["g_avg"])
    }
    print("C4", {k: round(v[0], 3) for k, v in res["C4_joint_named"].items()})
    print("C4+g", {k: [round(z, 3) for z in v[:3]] for k, v in res["C4_joint_named_plus_gavg"].items()})
    # C5 BW partial by season with g_avg coefficient
    c5 = {}
    for s, g in ev[np.isfinite(ev.x_BW) & (ev.x_BW.abs() <= 0.5) & np.isfinite(ev.y)].groupby("ss"):
        Xs = np.column_stack([np.ones(len(g)), g.x_BW, g.g_avg])
        bt = np.linalg.lstsq(Xs, g.y, rcond=None)[0]
        c5[int(s)] = {
            "n": len(g),
            "slope_BW": float(bt[1]),
            "coef_gavg": float(bt[2]),
            "implied_weight": float(-bt[1] / bt[2]),
            "marginal_BW": float(np.polyfit(g.x_BW, g.y, 1)[0]),
        }
    res["C5_BW_by_season"] = c5
    print("C5", json.dumps(c5, indent=0))
    (A.REPO / "research/audits/r6/out/h3check.json").write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
