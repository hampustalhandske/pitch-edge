"""Placebos and baseline-strength checks for H1 (OLS and ridge), auditor's own pipeline (alib/rederive.walk).

P1  Study's placebo: cross-book block of the TEST rows permuted within match-day; models trained on real features.
P2  Cross-book block permuted within match-day in TRAIN and TEST (the model can never learn book information).
P3  Certainly-stale: each test row's cross-book block replaced by the block of the same fixture's home team's
    previous match in the panel at least 300 days earlier (≈ the previous season); unmatched rows keep a
    within-day permuted block.
B1  Era-refit baseline: per-outcome OLS of the move on [1, g_avg, g_max] fitted only on training seasons from the
    same Avg/Max source era (Betbrain <= 2018/19, football-data Avg/Max >= 2019/20). The 2019/20 test season has no
    same-era training rows, so it is reported separately (pooled baseline there) and Δ is also given on 2020-2024.
B2  Rich consensus baseline: B1 plus Pinnacle overround and max booksum (no individual book).
Each model is compared with each baseline on identical rows. Seeds fixed.

Usage: OMP_NUM_THREADS=1 arch -arm64 uv run python research/audits/r6/placebo.py
"""

from __future__ import annotations

import json

import alib as A
import numpy as np
import pandas as pd
from rederive import walk


def within_day_perm(dates: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    perm = np.arange(len(dates))
    for _, idx in pd.Series(np.arange(len(dates))).groupby(dates):
        idx = idx.to_numpy()
        perm[idx] = rng.permutation(idx)
    return perm


def main() -> None:
    d = A.universe(A.raw())
    r = A.core_frame(d)
    dates = d.match_date.dt.strftime("%Y-%m-%d").to_numpy()
    ss = d.ss.to_numpy()
    res: dict = {}

    def summarise(w: pd.DataFrame, name: str) -> None:
        days = w.match_date.dt.strftime("%Y-%m-%d").to_numpy()
        late = (w.ss >= 2020).to_numpy()
        res[name] = {
            "all": A.boot(w.delta.to_numpy(), days),
            "2020_24": A.boot(w.delta.to_numpy()[late], days[late]),
            "2019": float(w.delta[w.ss == 2019].mean()),
            "gain_vs_open": float((w.kl_open - w.kl_model).mean()),
            "base_gain_vs_open": float((w.kl_open - w.kl_base).mean()),
        }
        print(
            f"{name:34s} all {A.fmt(res[name]['all'])}  2020-24 {A.fmt(res[name]['2020_24'])}  "
            f"model-vs-open {res[name]['gain_vs_open']:+.3e} base-vs-open {res[name]['base_gain_vs_open']:+.3e}"
        )

    # global permutation used for P1/P2 (one fixed permutation, applied to the full panel)
    rng = np.random.default_rng(20261021)
    perm = within_day_perm(dates, rng)

    def p1(btr, bte, tr, te, k):
        return btr, bte[np.searchsorted(te, perm[te])]  # within-day permutation stays inside te (whole days)

    def p2(btr, bte, tr, te, k):
        return btr[np.searchsorted(tr, perm[tr])], bte[np.searchsorted(te, perm[te])]

    # P3 stale: previous-season block of the same home team
    hist = d[["home_team", "match_date"]].reset_index()
    stale_idx = np.full(len(d), -1)
    for _team, g in hist.groupby("home_team"):
        md = g.match_date.to_numpy()
        ix = g["index"].to_numpy()
        for j in range(len(g)):
            cand = np.flatnonzero(md <= md[j] - np.timedelta64(300, "D"))
            if len(cand):
                stale_idx[ix[j]] = ix[cand[-1]]
    res["P3_stale_match_rate"] = float((stale_idx >= 0).mean())
    fallback = perm.copy()

    def p3_factory():
        books = {k: A.design(r, k)[1] for k in range(3)}
        names = A.design(r, 0)[2]

        def p3(btr, bte, tr, te, k):
            src = np.where(stale_idx[te] >= 0, stale_idx[te], fallback[te])
            raw = books[k][src]
            _, bte2 = A.centre_over(books[k][tr], raw, names)
            return btr, bte2

        return p3

    # baselines
    era = (ss >= 2019).astype(int)

    def era_base(extra: bool):
        def fn(tr, te, k):
            base = A.design(r, k)[0]
            if extra:
                base = np.column_stack([base, r["pin_over"], r["max_sum"]])
            y = r["move"][:, k]
            e_te = era[te][0]
            same = tr[era[tr] == e_te]
            use = same if len(same) >= 5000 else tr
            return A.ols(base[use], y[use], base[te])

        return fn

    for model in ("ols", "ridge"):
        summarise(walk(d, r, model), f"{model}: real vs R3 baseline")
        summarise(walk(d, r, model, book_tf=p1), f"{model}: P1 test-permuted")
        summarise(walk(d, r, model, book_tf=p2), f"{model}: P2 train+test permuted")
        summarise(walk(d, r, model, book_tf=p3_factory()), f"{model}: P3 stale (prev season)")
        summarise(walk(d, r, model, base_fn=era_base(False)), f"{model}: real vs era baseline")
        summarise(walk(d, r, model, base_fn=era_base(False), book_tf=p1), f"{model}: P1 vs era baseline")
        summarise(walk(d, r, model, base_fn=era_base(True)), f"{model}: real vs era+overround baseline")
        summarise(walk(d, r, model, base_fn=era_base(True), book_tf=p1), f"{model}: P1 vs era+overround baseline")
    # many within-day permutations for the P1 null distribution (ols, cheap)
    null = []
    for i in range(50):
        perm = within_day_perm(dates, np.random.default_rng(1000 + i))
        w = walk(d, r, "ols", book_tf=p1)
        null.append(float(w.delta.mean()))
    res["P1_ols_null_50"] = {"mean": float(np.mean(null)), "min": float(np.min(null)), "max": float(np.max(null))}
    null = []
    for i in range(50):
        perm = within_day_perm(dates, np.random.default_rng(2000 + i))
        w = walk(d, r, "ols", book_tf=p2)
        null.append(float(w.delta.mean()))
    res["P2_ols_null_50"] = {"mean": float(np.mean(null)), "min": float(np.min(null)), "max": float(np.max(null))}
    print(res["P1_ols_null_50"], res["P2_ols_null_50"], res["P3_stale_match_rate"])
    (A.REPO / "research/audits/r6/out/placebo.json").write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
