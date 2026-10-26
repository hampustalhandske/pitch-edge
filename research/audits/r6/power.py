"""Power of the aggregator-bracket test (timing.py T1): what P(book_open > Max_open) would look like if the book's
'open' column had been captured late. Counterfactual = the book's own close (an upper bound on lateness) and a
half-way quote (geometric midpoint of the book's open and close). Also the bracket profile for BF, BFE, 1XB in 2024/25.

Usage: OMP_NUM_THREADS=1 arch -arm64 uv run python research/audits/r6/power.py
"""

from __future__ import annotations

import json

import alib as A
import numpy as np
import pandas as pd
from timing import long

d = A.universe(A.raw())
r = A.core_frame(d)
f = long(d, r)
res = {}
bins = [-9, -0.10, -0.05, -0.02, -0.005, 0.005, 0.02, 0.05, 0.10, 9]
for b, seasons in (
    ("VC", (2019, 2023)),
    ("BF", (2024, 2024)),
    ("BFE", (2024, 2024)),
    ("1XB", (2024, 2024)),
    ("B365", (2019, 2024)),
):
    g = f[f.ss.between(*seasons)].copy()
    g["oc"] = (
        A.odds3(d, b + "C").T.reshape(-1)
        if False
        else np.concatenate([A.odds3(d, b + "C")[:, k] for k in range(3)])[f.ss.between(*seasons).to_numpy()]
    )
    m = np.isfinite(g["odds_" + b]) & np.isfinite(g.mx) & np.isfinite(g.psc) & np.isfinite(g.oc)
    g = g[m]
    g["bin"] = pd.cut(np.log(g.psc / g.ps), bins).astype(str)
    g["obs"] = g["odds_" + b] > g.mx + 1e-9
    g["if_close"] = g.oc > g.mx + 1e-9
    g["if_half"] = np.sqrt(g.oc * g["odds_" + b]) > g.mx + 1e-9
    res[b] = (
        g.groupby("bin")
        .agg(
            n=("obs", "size"),
            observed=("obs", "mean"),
            if_captured_at_close=("if_close", "mean"),
            if_captured_halfway=("if_half", "mean"),
        )
        .round(4)
        .to_dict(orient="index")
    )
    print(b, pd.DataFrame(res[b]).T.to_string())
(A.REPO / "research/audits/r6/out/power.json").write_text(json.dumps(res, indent=1))

# finer lateness fractions for VC and BF, top lengthening bins
fr = {}
for b, seasons in (("VC", (2019, 2023)), ("BF", (2024, 2024))):
    g = f[f.ss.between(*seasons)].copy()
    g["oc"] = np.concatenate([A.odds3(d, b + "C")[:, k] for k in range(3)])[f.ss.between(*seasons).to_numpy()]
    g = g[np.isfinite(g["odds_" + b]) & np.isfinite(g.mx) & np.isfinite(g.psc) & np.isfinite(g.oc)]
    mv = np.log(g.psc / g.ps)
    for lo in (0.05, 0.10):
        h = g[mv > lo]
        fr[f"{b}_move>{lo}"] = {
            "n": len(h),
            "observed": float((h["odds_" + b] > h.mx + 1e-9).mean()),
            **{
                f"lam{lam}": float((h["odds_" + b] * (h.oc / h["odds_" + b]) ** lam > h.mx + 1e-9).mean())
                for lam in (0.05, 0.1, 0.2, 0.3)
            },
        }
print(json.dumps(fr, indent=1))
res["fractions"] = fr
(A.REPO / "research/audits/r6/out/power.json").write_text(json.dumps(res, indent=1))
