"""Benchmark sensitivity, commission haircut, exchange check, achievable single books, placebos (evaluation window
unless stated). Rule and cap as registered; theta = 0.02 unless stated."""

import numpy as np
import pandas as pd
from alib import DV, EVAL, EXPL, O, boot, fmt, load, power, rule, three, valid

d = load()
ev = d[(d.match_date >= EVAL[0]) & (d.match_date <= EVAL[1])].reset_index(drop=True)


def frame(w, pre):
    ps, pc, px = three(w, "PS"), three(w, "PSC"), three(w, pre)
    ok = valid(ps) & valid(pc) & valid(px) & w.ft_result.isin(list("HDA")).to_numpy()
    return w[ok].reset_index(drop=True), ps[ok], pc[ok], px[ok]


w, ps, pc, px = frame(ev, "MX")
r = np.arange(len(w))
bet, j, e, pr = rule(px, ps, 0.02)
print(f"== 1. benchmark de-vig of the CLOSE (max, theta=0.02, n={bet.sum():d})")
for m in ("power", "shin", "mult"):
    clv = np.log(pr * DV[m](pc)[r, j])[bet]
    print(f"  close {m:5s}: {fmt(*boot(clv, w.match_date[bet]))}")
print("  same de-vig at open AND close (rule re-run):")
for m in ("shin", "mult"):
    b2, j2, e2, p2 = rule(px, ps, 0.02, ref_method=m)
    clv = np.log(p2 * DV[m](pc)[r, j2])[b2]
    print(f"  open+close {m:5s}: {fmt(*boot(clv, w.match_date[b2]), b2.sum())}")
print(
    "  raw (vig-inclusive) Pinnacle close as benchmark, i.e. log(price/PSC):",
    fmt(*boot(np.log(pr / pc[r, j])[bet], w.match_date[bet])),
)
print("  share of bets with price > raw PSC:", round(float((pr > pc[r, j])[bet].mean()), 3))
print(
    "  mean odds of bets:",
    round(float(pr[bet].mean()), 2),
    " outcome mix:",
    pd.Series(np.array(O)[j[bet]]).value_counts(normalize=True).round(3).to_dict(),
)

print("\n== 2. commission haircut on the taken price (as if the max were an exchange back price)")
pcl = power(pc)
for c in (0.02, 0.05):
    eff = 1 + (pr - 1) * (1 - c)
    print(f"  {c:.0%} commission: {fmt(*boot(np.log(eff * pcl[r, j])[bet], w.match_date[bet]))}")


def ov(pre, g):
    return (1 / three(g, pre)).sum(axis=1)


s24 = d[d.season == "2024-2025"]
print(
    "  2024-25 median overround: Max {:.4f}  BFE {:.4f}  PS {:.4f} ; share outcomes Max<BFE {:.3f}".format(
        np.nanmedian(ov("Max", s24)),
        np.nanmedian(ov("BFE", s24)),
        np.nanmedian(ov("PS", s24)),
        np.nanmean(three(s24, "Max") < three(s24, "BFE") - 1e-9),
    )
)

print("\n== 3. achievable single books (rule with price = that book's open quote)")
BOOKS = ["B365", "BW", "IW", "WH", "VC", "BF", "1XB"]
for wn, (lo, hi) in {"exploratory": EXPL, "evaluation": EVAL}.items():
    ww = d[(d.match_date >= lo) & (d.match_date <= hi)].reset_index(drop=True)
    for b in BOOKS + ["NAMEDMAX"]:
        if b == "NAMEDMAX":
            arr = np.fmax.reduce([three(ww, x) for x in BOOKS])
            for o_i, o in enumerate(O):
                ww[f"NAMEDMAX{o}"] = arr[:, o_i]
        if ww[f"{b}H"].notna().sum() < 1000:
            continue
        f, a, c, x = frame(ww, b)
        rr = np.arange(len(f))
        for th in (0.0, 0.02, 0.05):
            bb, jj, ee, pp = rule(x, a, th)
            if bb.sum() < 30:
                continue
            clv = np.log(pp * power(c)[rr, jj])[bb]
            shn = np.log(pp * DV["shin"](c)[rr, jj])[bb]
            print(
                f"  {wn[:4]} {b:8s} th={th:.2f} n={bb.sum():5d} CLV(power) {fmt(*boot(clv, f.match_date[bb]))}"
                f"  CLV(shin) {fmt(*boot(shn, f.match_date[bb]))}"
            )
        if wn == "evaluation":
            bb, jj, ee, pp = rule(x, a, 0.02)
            clv = np.log(pp * power(c)[rr, jj])
            t = pd.DataFrame({"s": f.season, "clv": clv, "day": f.match_date})[bb]
            print(
                "      by season (th=0.02): "
                + "; ".join(f"{s} {100 * g.clv.mean():+.2f}% (n={len(g)})" for s, g in t.groupby("s"))
            )

print("\n== 4. placebos (max, theta=0.02)")
rng = np.random.default_rng(11)
prem = np.log(px / ps)  # per-outcome premium of the max over Pinnacle open
day = w.match_date.to_numpy()
res = {"real": [], "shuffled": []}
for _k in range(20):
    perm = np.arange(len(w))
    for _dd, idx in pd.Series(np.arange(len(w))).groupby(day):
        perm[idx.to_numpy()] = rng.permutation(idx.to_numpy())
    pxs = ps * np.exp(prem[perm])
    b2, j2, e2, p2 = rule(pxs, ps, 0.02)
    clv = np.log(p2 * pcl[r, j2])[b2]
    mv = np.log(pcl[r, j2] / power(ps)[r, j2])[b2]
    res["shuffled"].append((clv.mean(), e2[b2].mean(), mv.mean(), b2.sum()))
real_mv = np.log(pcl[r, j] / power(ps)[r, j])[bet]
print(
    f"  real    : CLV {np.log(pr * pcl[r, j])[bet].mean():.4f} edge_open {e[bet].mean():.4f} Pinnacle move on backed side {real_mv.mean():.4f} n={bet.sum():d}"
)
s = np.array(res["shuffled"])
print(
    f"  shuffled premium within match-day (20 perms): CLV {s[:, 0].mean():.4f} edge_open {s[:, 1].mean():.4f} move {s[:, 2].mean():.4f} (sd {s[:, 2].std():.4f}) n~{int(s[:, 3].mean()):d}"
)
print("  real move CI:", fmt(*boot(real_mv, w.match_date[bet])))
# random book as fair: bet365 open (power) as the fair reference instead of Pinnacle
f2, a2, c2, x2 = frame(ev, "MX")
b365 = three(f2, "B365")
okb = valid(b365)
bb, jj, ee, pp = rule(x2[okb], b365[okb], 0.02)
rr = np.arange(okb.sum())
clv = np.log(pp * power(c2[okb])[rr, jj])[bb]
print("  fair reference = bet365 open instead of Pinnacle open:", fmt(*boot(clv, f2.match_date[okb][bb]), bb.sum()))
# lagged/stale signal: decide with the max premium from the SAME fixture's previous meeting is not defined;
# instead use a certainly-uninformative premium: the max premium of the reversed outcome order (H<->A)
pxr = ps * np.exp(prem[:, ::-1])
b3, j3, e3, p3 = rule(pxr, ps, 0.02)
mv3 = np.log(pcl[r, j3] / power(ps)[r, j3])[b3]
print(
    f"  premium taken from the mirrored outcome (H<->A): move {mv3.mean():.4f}, CLV {np.log(p3 * pcl[r, j3])[b3].mean():.4f} n={b3.sum():d}"
)
