"""Who sets the max on the bets the rule makes, and does CLV differ by setter?

For each evaluation-window bet (max, theta=0.02): flag whether the max price equals the same-outcome open quote
of any individually named book (achievable at that named book at the same snapshot) or exceeds all of them
(an aggregator-only price). Also: how often Max == Pinnacle's own open quote (simultaneity of the aggregator
with the Pinnacle capture)."""

import numpy as np
import pandas as pd
from alib import EVAL, EXPL, boot, fmt, load, power, rule, three, valid

d = load()
NAMED = ["B365", "BW", "IW", "WH", "VC", "LB", "BF", "1XB", "BS", "GB", "SJ"]
for wname, (lo, hi) in {"evaluation": EVAL, "exploratory": EXPL}.items():
    w = d[(d.match_date >= lo) & (d.match_date <= hi)].reset_index(drop=True)
    ps, pc, px = three(w, "PS"), three(w, "PSC"), three(w, "MX")
    ok = valid(ps) & valid(pc) & valid(px) & w.ft_result.isin(list("HDA")).to_numpy()
    w, ps, pc, px = w[ok].reset_index(drop=True), ps[ok], pc[ok], px[ok]
    eq_ps = np.abs(px - ps) < 1e-9
    print(
        f"\n== {wname}: share of outcomes with Max == PS open: {eq_ps.mean():.3f}; Max < PS: {(px < ps - 1e-9).mean():.4f}"
    )
    bet, j, e, pr = rule(px, ps, 0.02)
    r = np.arange(len(w))
    clv = np.log(pr * power(pc)[r, j])
    named = np.full(len(w), np.nan)
    who = np.full(len(w), "agg-only", dtype=object)
    for b in NAMED:
        v = three(w, b)[r, j]
        hit = (np.abs(v - pr) < 1e-9) & (who == "agg-only")
        who[hit] = b
        named = np.fmax(named, v)
    gap = np.log(pr / named)
    t = pd.DataFrame({"who": who, "clv": clv, "edge": e, "gap": gap, "day": w.match_date})[bet]
    print("bets", len(t))
    for k, g in t.groupby("who"):
        if len(g) >= 50:
            print(
                f"  max set by {k:9s} n={len(g):6d} share={len(g) / len(t):.3f} edge_open={g.edge.mean():.4f}"
                f" CLV {fmt(*boot(g.clv.to_numpy(), g.day))}  ratio CLV/edge={g.clv.mean() / g.edge.mean():.2f}"
            )
    a = t[t.who == "agg-only"]
    print(f"  agg-only: median log(max/best named) = {a.gap.median():.4f}")
    # bets where the best NAMED book alone would also have cleared theta
    best_named_edge = np.log(named * power(ps)[r, j])
    m = bet & (best_named_edge > 0.02)
    print(
        f"  bets where best named book ALSO clears theta at same outcome: n={m.sum()} CLV(at named price) "
        f"{fmt(*boot(np.log(named * power(pc)[r, j])[m], w.match_date[m]))}"
    )
