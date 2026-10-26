"""Simultaneity / staleness test, evaluation window.

For a source X with open and close quotes, per outcome (log price, so margins cancel in differences):
  dev   = log X_open - log PS_open                 (X's deviation from Pinnacle at the snapshot)
  a_X   = OLS slope of (log PS_close - log PS_open) on dev     -> Pinnacle moves toward X   (X leads / X later)
  b_X   = OLS slope of (log X_close - log X_open) on -dev      -> X moves toward Pinnacle   (X stale)
A stale soft quote shows b >> a. A quote captured after Pinnacle's open shows a >> b (X already contains the
move). Both computed on outcomes with all quotes present; dev is demeaned within book to strip the margin.
Also: CLV of the max rule by weekday and by the size of Pinnacle's open->close move."""

import numpy as np
import pandas as pd
from alib import EVAL, EXPL, boot, fmt, load, power, rule, three, valid

d = load()


def slopes(w, X, XC):
    ps, psc, x, xc = (np.log(three(w, k)) for k in ("PS", "PSC", X, XC))
    m = np.isfinite(ps) & np.isfinite(psc) & np.isfinite(x) & np.isfinite(xc)
    dev = (x - ps)[m]
    dev = dev - dev.mean()
    mp = (psc - ps)[m]
    mx = (xc - x)[m]
    a = np.polyfit(dev, mp, 1)[0]
    b = np.polyfit(-dev, mx, 1)[0]
    return a, b, m.sum()


for wname, (lo, hi) in {"evaluation": EVAL, "exploratory": EXPL}.items():
    w = d[(d.match_date >= lo) & (d.match_date <= hi)].reset_index(drop=True)
    print(f"== {wname}")
    pairs = [("B365", "B365C"), ("BW", "BWC"), ("IW", "IWC"), ("WH", "WHC"), ("VC", "VCC")]
    pairs += [("Avg", "AvgC"), ("Max", "MaxC")] if wname == "evaluation" else []
    for X, XC in pairs:
        if f"{XC}H" not in w or w[f"{XC}H"].notna().mean() < 0.3:
            print(f"  {X:5s} no close quotes in this window")
            continue
        a, b, n = slopes(w, X, XC)
        print(f"  {X:5s} a(PS->X)={a:+.3f}  b(X->PS)={b:+.3f}  n_outcomes={n}")
    ps, pc, px = three(w, "PS"), three(w, "PSC"), three(w, "MX")
    ok = valid(ps) & valid(pc) & valid(px) & w.ft_result.isin(list("HDA")).to_numpy()
    w, ps, pc, px = w[ok].reset_index(drop=True), ps[ok], pc[ok], px[ok]
    bet, j, e, pr = rule(px, ps, 0.02)
    r = np.arange(len(w))
    clv = np.log(pr * power(pc)[r, j])
    move = np.abs(np.log(power(pc)[r, j] / power(ps)[r, j]))
    t = pd.DataFrame({"clv": clv, "move": move, "day": w.match_date, "wd": w.match_date.dt.day_name()})[bet]
    print("  by weekday:")
    for k, g in t.groupby("wd"):
        print(f"    {k:9s} n={len(g):5d} CLV {fmt(*boot(g.clv.to_numpy(), g.day))}")
    t["mq"] = pd.qcut(t.move, 5, labels=False)
    print("  by quintile of |Pinnacle open->close move| on the backed side:")
    for k, g in t.groupby("mq"):
        print(f"    q{k} move~{g.move.median():.3f} n={len(g)} CLV {fmt(*boot(g.clv.to_numpy(), g.day))}")
