"""Leave-one-division-out CIs and chronological thirds for the auditor's re-derived H1 frames (vs the registered
R3 baseline) and for H1 ridge / H2 res_tiny against the era-refit baseline.

Usage: OMP_NUM_THREADS=1 arch -arm64 uv run python research/audits/r6/loo.py
"""

from __future__ import annotations

import json

import alib as A
import numpy as np
import pandas as pd
from rederive import walk

d = A.universe(A.raw())
r = A.core_frame(d)
era = (d.ss.to_numpy() >= 2019).astype(int)


def era_base(tr, te, k):
    base = A.design(r, k)[0]
    y = r["move"][:, k]
    same = tr[era[tr] == era[te][0]]
    use = same if len(same) >= 5000 else tr
    return A.ols(base[use], y[use], base[te])


eb = walk(d, r, "base", base_fn=era_base)[["mid", "kl_base"]].rename(columns={"kl_base": "kl_era"})
frames = {}
for m in ("ols", "ridge"):
    frames[f"{m}_vs_R3"] = pd.read_parquet(A.OUT / f"rederive_{m}.parquet")
s = pd.read_parquet(A.REPO / "data/research/r6/closing-line-prediction-cross-book-gbm-v1__hgbres_tiny.parquet")
s = s[s.segment == "pinnacle"].rename(columns={"match_id": "mid"})
frames["res_tiny_vs_R3"] = s.assign(delta=s.delta_kl_bits)[["mid", "match_date", "division", "kl_model", "delta"]]
for nm in ("ridge_vs_R3", "res_tiny_vs_R3"):
    f = frames[nm].merge(eb, on="mid")
    frames[nm.replace("R3", "era")] = f.assign(delta=f.kl_era - f.kl_model)
res = {}
for nm, f in frames.items():
    f = f.sort_values("match_date").reset_index(drop=True)
    days = pd.to_datetime(f.match_date).dt.strftime("%Y-%m-%d").to_numpy()
    loo = {}
    for dv in sorted(f.division.unique()):
        k = (f.division != dv).to_numpy()
        loo[dv] = A.boot(f.delta.to_numpy()[k], days[k], n=1000)
    thirds = [float(x.mean()) for x in np.array_split(f.delta.to_numpy(), 3)]
    lo = min(v[1] for v in loo.values())
    res[nm] = {
        "n": len(f),
        "mean": float(f.delta.mean()),
        "thirds": thirds,
        "min_loo_ci_lo": lo,
        "worst_loo_div": min(loo, key=lambda z: loo[z][1]),
        "divisions_positive": int((f.groupby("division").delta.mean() > 0).sum()),
    }
    print(nm, json.dumps(res[nm]))
(A.REPO / "research/audits/r6/out/loo.json").write_text(json.dumps(res, indent=1))
