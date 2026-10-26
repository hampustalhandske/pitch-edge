"""Same-day snapshot check and direction-match comparison (evaluation window, theta=0.02).

(1) For matches on the snapshot weekday (Fri/Tue), by kickoff hour: share where Pinnacle open == close on all three
outcomes (would indicate the 'open' snapshot was taken at/after the close) and CLV of max bets.
(2) Share of bets where Pinnacle's backed-side probability fell open->close (moved toward the taken price),
for max vs bet365, within the same open-edge band."""

import numpy as np
import pandas as pd
from alib import EVAL, boot, fmt, load, power, rule, three, valid

d = load()
w0 = d[(d.match_date >= EVAL[0]) & (d.match_date <= EVAL[1])].reset_index(drop=True)
out = {}
for src in ("MX", "B365"):
    ps, pc, px = three(w0, "PS"), three(w0, "PSC"), three(w0, src)
    ok = valid(ps) & valid(pc) & valid(px) & w0.ft_result.isin(list("HDA")).to_numpy()
    w, ps, pc, px = w0[ok].reset_index(drop=True), ps[ok], pc[ok], px[ok]
    bet, j, e, pr = rule(px, ps, 0.02)
    r = np.arange(len(w))
    out[src] = pd.DataFrame(
        {
            "e": e,
            "toward": power(pc)[r, j] < power(ps)[r, j],
            "clv": np.log(pr * power(pc)[r, j]),
            "same": (np.abs(ps - pc) < 1e-9).all(axis=1),
            "wd": w.match_date.dt.dayofweek,
            "ko": w.kickoff_time,
            "day": w.match_date,
        }
    )[bet]
m = out["MX"]
m["hr"] = pd.to_numeric(m.ko.str[:2], errors="coerce")
sd = m[m.wd.isin([1, 4])]
print("same-day (Tue/Fri) max bets by UK kickoff hour:")
for lab, g in sd.groupby(pd.cut(sd.hr, [-1, 15, 18, 24]), observed=True):
    print(f"  ko {lab}: n={len(g)} PSopen==PSclose share={g.same.mean():.3f} CLV {fmt(*boot(g.clv.to_numpy(), g.day))}")
print("all bets PSopen==PSclose share:", round(m.same.mean(), 4))
print("direction match (Pinnacle moved toward the taken price), by open-edge band:")
for lo_, hi_ in ((0.02, 0.03), (0.03, 0.05), (0.05, 0.5)):
    row = []
    for s, t in out.items():
        g = t[(t.e > lo_) & (t.e <= hi_)]
        row.append(f"{s}: n={len(g)} toward={g.toward.mean():.3f} CLV/edge={g.clv.mean() / g.e.mean():.2f}")
    print(f"  edge ({lo_},{hi_}]  " + " | ".join(row))
