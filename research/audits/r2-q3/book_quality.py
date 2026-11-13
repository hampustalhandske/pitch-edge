"""For single-book candidates (evaluation, theta=0.02): share of bets whose quote exceeds the aggregator Max at the
same snapshot (a quote the aggregator did not see: different capture time or error) and CLV with/without them;
also CLV of the bets after a 1.10 cap on price / Pinnacle open."""

import numpy as np
import pandas as pd
from alib import EVAL, boot, fmt, load, power, rule, three, valid

d = load()
w0 = d[(d.match_date >= EVAL[0]) & (d.match_date <= EVAL[1])].reset_index(drop=True)
for b in ("WH", "BW", "IW", "1XB", "B365", "VC"):
    ps, pc, px, mx = three(w0, "PS"), three(w0, "PSC"), three(w0, b), three(w0, "Max")
    ok = valid(ps) & valid(pc) & valid(px) & valid(mx) & w0.ft_result.isin(list("HDA")).to_numpy()
    w, ps, pc, px, mx = w0[ok].reset_index(drop=True), ps[ok], pc[ok], px[ok], mx[ok]
    bet, j, e, pr = rule(px, ps, 0.02)
    r = np.arange(len(w))
    clv = np.log(pr * power(pc)[r, j])
    above = pr > mx[r, j] + 1e-9
    t = pd.DataFrame({"clv": clv, "above": above, "day": w.match_date, "r": pr / ps[r, j]})[bet]
    g = t[~t.above]
    c = t[t.r <= 1.10]
    print(
        f"{b:5s} n={len(t):5d} share quote>Max={t.above.mean():.3f} CLV(all) {fmt(*boot(t.clv.to_numpy(), t.day))}"
        f" CLV(quote<=Max) {fmt(*boot(g.clv.to_numpy(), g.day), len(g))} CLV(cap1.10) {fmt(*boot(c.clv.to_numpy(), c.day), len(c))}"
    )
