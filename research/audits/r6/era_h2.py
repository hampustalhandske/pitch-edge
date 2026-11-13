"""H2 (study's counted res_tiny / tiny frames) and H1 against the auditor's era-refit baseline, by season; and the
stale placebo P3 for ridge against the era baseline.

Usage: OMP_NUM_THREADS=1 arch -arm64 uv run python research/audits/r6/era_h2.py
"""

from __future__ import annotations

import json

import alib as A
import numpy as np
import pandas as pd
import placebo as P
from rederive import walk

d = A.universe(A.raw())
r = A.core_frame(d)
ss = d.ss.to_numpy()
era = (ss >= 2019).astype(int)


def era_base(tr, te, k):
    base = A.design(r, k)[0]
    y = r["move"][:, k]
    same = tr[era[tr] == era[te][0]]
    use = same if len(same) >= 5000 else tr
    return A.ols(base[use], y[use], base[te])


eb = walk(d, r, "base", base_fn=era_base)[["mid", "kl_base", "ss", "match_date"]].rename(columns={"kl_base": "kl_era"})
res = {}
for tag, f in (
    ("ols", "closing-line-prediction-cross-book-linear-v1__ols"),
    ("ridge", "closing-line-prediction-cross-book-linear-v1__ridge"),
    ("hgb_tiny", "closing-line-prediction-cross-book-gbm-v1__hgb_tiny"),
    ("res_tiny", "closing-line-prediction-cross-book-gbm-v1__hgbres_tiny"),
):
    s = pd.read_parquet(A.REPO / "data/research/r6" / f"{f}.parquet")
    s = s[s.segment == "pinnacle"].merge(eb, left_on="match_id", right_on="mid", validate="one_to_one")
    s["d_era"] = s.kl_era - s.kl_model
    days = pd.to_datetime(s.match_date_x).dt.strftime("%Y-%m-%d").to_numpy()
    late = (s.ss >= 2020).to_numpy()
    res[tag] = {
        "n": len(s),
        "vs_era": A.boot(s.d_era.to_numpy(), days),
        "vs_era_2020_24": A.boot(s.d_era.to_numpy()[late], days[late]),
        "by_season": s.groupby("ss").d_era.mean().round(7).to_dict(),
    }
    print(tag, A.fmt(res[tag]["vs_era"]), A.fmt(res[tag]["vs_era_2020_24"]), res[tag]["by_season"])

# P3 stale placebo vs era for ridge (previous-season block of the same home team)
hist = d[["home_team", "match_date"]].reset_index()
stale = np.full(len(d), -1)
for _, g in hist.groupby("home_team"):
    md, ix = g.match_date.to_numpy(), g["index"].to_numpy()
    for j in range(len(g)):
        c = np.flatnonzero(md <= md[j] - np.timedelta64(300, "D"))
        if len(c):
            stale[ix[j]] = ix[c[-1]]
fb = P.within_day_perm(d.match_date.dt.strftime("%Y-%m-%d").to_numpy(), np.random.default_rng(20261021))
books = {k: A.design(r, k)[1] for k in range(3)}
names = A.design(r, 0)[2]


def p3(btr, bte, tr, te, k):
    src = np.where(stale[te] >= 0, stale[te], fb[te])
    return btr, A.centre_over(books[k][tr], books[k][src], names)[1]


for m in ("ols", "ridge"):
    w = walk(d, r, m, base_fn=era_base, book_tf=p3)
    res[f"P3_{m}_vs_era"] = A.boot(w.delta.to_numpy(), w.match_date.dt.strftime("%Y-%m-%d").to_numpy())
    print("P3", m, "vs era", A.fmt(res[f"P3_{m}_vs_era"]))
(A.REPO / "research/audits/r6/out/era_h2.json").write_text(json.dumps(res, indent=1, default=str))
