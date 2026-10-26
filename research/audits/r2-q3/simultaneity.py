"""Direct test of whether the aggregator Max was captured after Pinnacle's open quote.

If the aggregator snapshot (which contains Pinnacle) were taken later than football-data's PS open, then on
outcomes where Pinnacle later SHORTENED its price, Pinnacle's (later, shorter) quote could no longer equal PS_open
and Max < PS_open would appear whenever Pinnacle was the top book; P(Max == PS_open) would also fall with the size of
Pinnacle's subsequent move. Under simultaneous capture both are flat in the subsequent move.
Also: lead/lag slopes for a mean of the 5 named soft books (directly captured) vs the aggregator Avg."""

import numpy as np
import pandas as pd
from alib import EVAL, load, three

d = load()
w = d[(d.match_date >= EVAL[0]) & (d.match_date <= EVAL[1])].reset_index(drop=True)
ps, psc, mx = three(w, "PS").ravel(), three(w, "PSC").ravel(), three(w, "Max").ravel()
m = np.isfinite(ps) & np.isfinite(psc) & np.isfinite(mx)
mv = np.log(psc[m] / ps[m])
t = pd.DataFrame({"mv": mv, "eq": np.abs(mx[m] - ps[m]) < 1e-9, "lt": mx[m] < ps[m] - 1e-9})
t["bin"] = pd.cut(t.mv, [-9, -0.10, -0.05, -0.02, -0.005, 0.005, 0.02, 0.05, 0.10, 9])
print(
    t.groupby("bin", observed=True)
    .agg(n=("eq", "size"), share_max_eq_PSopen=("eq", "mean"), share_max_lt_PSopen=("lt", "mean"))
    .round(4)
)

# named-soft-book mean vs aggregator Avg
names = ["B365", "BW", "IW", "WH", "VC"]
o = np.stack([np.log(three(w, b)) for b in names])
oc = np.stack([np.log(three(w, b + "C")) for b in names])
na, nac = np.nanmean(o, axis=0), np.nanmean(oc, axis=0)
for lab, x, xc in (("Named5", na, nac), ("Avg", np.log(three(w, "Avg")), np.log(three(w, "AvgC")))):
    lps, lpsc = np.log(three(w, "PS")), np.log(three(w, "PSC"))
    k = np.isfinite(x) & np.isfinite(xc) & np.isfinite(lps) & np.isfinite(lpsc) & np.isfinite(na) & np.isfinite(nac)
    dev = (x - lps)[k]
    dev -= dev.mean()
    print(
        lab,
        f"a(PS->X)={np.polyfit(dev, (lpsc - lps)[k], 1)[0]:.3f} b(X->PS)={np.polyfit(-dev, (xc - x)[k], 1)[0]:.3f} n={k.sum():d}",
    )
# multivariate: does Avg's deviation predict the Pinnacle move beyond the named-5 deviation?
lps, lpsc, la = np.log(three(w, "PS")), np.log(three(w, "PSC")), np.log(three(w, "Avg"))
k = np.isfinite(la) & np.isfinite(na) & np.isfinite(lps) & np.isfinite(lpsc)
X = np.column_stack([np.ones(k.sum()), (na - lps)[k], (la - lps)[k]])
beta = np.linalg.lstsq(X, (lpsc - lps)[k], rcond=None)[0]
print("PS move ~ named5 dev + Avg dev:", beta.round(3))
