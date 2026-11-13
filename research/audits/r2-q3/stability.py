"""Stability (thirds, divisions, LODO), eligibility drops, and the 2025-07+ Betfair post segment, re-derived."""

import numpy as np
import pandas as pd
from alib import EVAL, boot, fmt, load, power, rule, three, valid

d = load()
w0 = d[(d.match_date >= EVAL[0]) & (d.match_date <= EVAL[1])].reset_index(drop=True)
ps, pc, px = three(w0, "PS"), three(w0, "PSC"), three(w0, "MX")
res_ok = w0.ft_result.isin(list("HDA")).to_numpy()
ok = valid(ps) & valid(pc) & valid(px) & res_ok
print("eval main matches", len(w0), "eligible", ok.sum(), "dropped", (~ok).sum())
drop = w0[~ok]
print("  dropped by division:", drop.division.value_counts().head(8).to_dict())
print(
    "  result mix kept:",
    w0[ok].ft_result.value_counts(normalize=True).round(3).to_dict(),
    " dropped:",
    drop.ft_result.value_counts(normalize=True).round(3).to_dict(),
)
w, ps, pc, px = w0[ok].reset_index(drop=True), ps[ok], pc[ok], px[ok]
bet, j, e, pr = rule(px, ps, 0.02)
r = np.arange(len(w))
t = pd.DataFrame({"clv": np.log(pr * power(pc)[r, j]), "day": w.match_date, "div": w.division})[bet]
lo, hi = pd.Timestamp(EVAL[0]), pd.Timestamp(EVAL[1])
span = (hi - lo) / 3
th = np.where(t.day < lo + span, 1, np.where(t.day < lo + 2 * span, 2, 3))
for k in (1, 2, 3):
    g = t[th == k]
    print(f"  third {k}: {fmt(*boot(g.clv.to_numpy(), g.day), len(g))}")
per = t.groupby("div").clv.agg(["size", "mean"])
print(
    f"  divisions positive: {(per['mean'] > 0).sum()}/{len(per)}  range {per['mean'].min():.4f}..{per['mean'].max():.4f}"
)
lodo = [t[t["div"] != dv].clv.mean() for dv in per.index]
print(f"  LODO range {min(lodo):.4f}..{max(lodo):.4f}")
# cap sensitivity
for cap in (None, 1.25):
    b2, j2, e2, p2 = rule(px, ps, 0.02, cap=cap)
    print(f"  cap={cap}: {fmt(*boot(np.log(p2 * power(pc)[r, j2])[b2], w.match_date[b2]), b2.sum())}")
# post segment: Betfair Exchange open as fair, BFE close as benchmark
p = d[(d.match_date >= "2025-07-01") & (d.match_date <= "2026-10-19")].reset_index(drop=True)
for src in ("Max", "B365"):
    a, c, x = three(p, "BFE"), three(p, "BFEC"), three(p, src)
    k = valid(a) & valid(c) & valid(x) & p.ft_result.isin(list("HDA")).to_numpy()
    f = p[k].reset_index(drop=True)
    b2, j2, e2, p2 = rule(x[k], a[k], 0.02)
    rr = np.arange(len(f))
    print(f"  post {src} th=0.02: {fmt(*boot(np.log(p2 * power(c[k])[rr, j2])[b2], f.match_date[b2]), b2.sum())}")
