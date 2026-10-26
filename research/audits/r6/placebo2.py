"""Follow-up to placebo.py: P2/P3 against the era-refit baseline, and a 30-draw null distribution of the
study's P1 (test-only within-day permutation) for ridge against both baselines.

Usage: OMP_NUM_THREADS=1 arch -arm64 uv run python research/audits/r6/placebo2.py
"""

from __future__ import annotations

import json

import alib as A
import numpy as np
import placebo as P
from rederive import walk


def main() -> None:
    d = A.universe(A.raw())
    r = A.core_frame(d)
    dates = d.match_date.dt.strftime("%Y-%m-%d").to_numpy()
    ss = d.ss.to_numpy()
    era = (ss >= 2019).astype(int)

    def era_base(tr, te, k):
        base = A.design(r, k)[0]
        y = r["move"][:, k]
        same = tr[era[tr] == era[te][0]]
        use = same if len(same) >= 5000 else tr
        return A.ols(base[use], y[use], base[te])

    state = {"perm": P.within_day_perm(dates, np.random.default_rng(20261021))}

    def p1(btr, bte, tr, te, k):
        return btr, bte[np.searchsorted(te, state["perm"][te])]

    def p2(btr, bte, tr, te, k):
        return btr[np.searchsorted(tr, state["perm"][tr])], bte[np.searchsorted(te, state["perm"][te])]

    res = {}
    for model in ("ols", "ridge"):
        for nm, tf in (("P2", p2),):
            w = walk(d, r, model, base_fn=era_base, book_tf=tf)
            res[f"{model}_{nm}_vs_era"] = A.boot(w.delta.to_numpy(), w.match_date.dt.strftime("%Y-%m-%d").to_numpy())
            print(model, nm, "vs era", A.fmt(res[f"{model}_{nm}_vs_era"]))
    null_r3, null_era = [], []
    for i in range(30):
        state["perm"] = P.within_day_perm(dates, np.random.default_rng(3000 + i))
        null_r3.append(float(walk(d, r, "ridge", book_tf=p1).delta.mean()))
        null_era.append(float(walk(d, r, "ridge", base_fn=era_base, book_tf=p1).delta.mean()))
    res["ridge_P1_null30_vs_R3"] = {
        "mean": float(np.mean(null_r3)),
        "sd": float(np.std(null_r3)),
        "min": min(null_r3),
        "max": max(null_r3),
    }
    res["ridge_P1_null30_vs_era"] = {
        "mean": float(np.mean(null_era)),
        "sd": float(np.std(null_era)),
        "min": min(null_era),
        "max": max(null_era),
    }
    print(json.dumps(res, indent=1))
    (A.REPO / "research/audits/r6/out/placebo2.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
