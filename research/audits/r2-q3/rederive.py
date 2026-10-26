"""Re-derive the R2-Q3 grid (max/bet365 x theta, both windows) from raw odds_all columns."""

import numpy as np
from alib import EVAL, EXPL, boot, fmt, load, power, rule, three, valid

d = load()
for wname, (lo, hi) in {"exploratory": EXPL, "evaluation": EVAL}.items():
    w = d[(d.match_date >= lo) & (d.match_date <= hi)].reset_index(drop=True)
    for src, pre in (("max", "MX"), ("bet365", "B365")):
        ps, pc, px = three(w, "PS"), three(w, "PSC"), three(w, pre)
        ok = valid(ps) & valid(pc) & valid(px) & w.ft_result.isin(list("HDA")).to_numpy()
        f = w[ok].reset_index(drop=True)
        ps, pc, px = ps[ok], pc[ok], px[ok]
        pcl = power(pc)
        for th in (0.0, 0.02, 0.05):
            bet, j, e, pr = rule(px, ps, th)
            r = np.arange(len(f))
            clv = np.log(pr * pcl[r, j])[bet]
            roi = (pr * (f[["YH", "YD", "YA"]].to_numpy()[r, j]) - 1)[bet]
            print(
                wname,
                src,
                th,
                "elig",
                len(f),
                "CLV",
                fmt(*boot(clv, f.match_date[bet]), bet.sum()),
                "ROI",
                fmt(*boot(roi, f.match_date[bet])),
                "mean edge_open",
                round(e[bet].mean(), 4),
            )
