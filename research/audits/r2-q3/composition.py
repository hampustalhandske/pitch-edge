"""What is the 'max' price? Per season: how often Max < a named book's open quote (impossible if both were
captured at the same instant over a superset of books), how often Max == a named book, Max vs Betfair Exchange."""

import numpy as np
import pandas as pd
from alib import O, load

d = load()
d = d[d.match_date >= "2012-07-01"]
BOOKS = ["PS", "B365", "BW", "IW", "WH", "VC", "LB", "BF", "1XB", "CL", "BS", "GB", "SJ"]
rows = []
for s, g in d.groupby("season"):
    r = {"season": s, "n": len(g)}
    mx = np.concatenate([g[f"MX{o}"].to_numpy() for o in O])
    named_max = np.full(len(mx), np.nan)
    for b in BOOKS:
        if f"{b}H" not in g or g[f"{b}H"].notna().mean() < 0.5:
            continue
        v = np.concatenate([g[f"{b}{o}"].to_numpy() for o in O])
        ok = np.isfinite(v) & np.isfinite(mx)
        r[f"mx<{b}"] = round(float((mx[ok] < v[ok] - 1e-9).mean()), 3)
        named_max = np.fmax(named_max, v)
    ok = np.isfinite(named_max) & np.isfinite(mx)
    r["mx>allnamed"] = round(float((mx[ok] > named_max[ok] + 1e-9).mean()), 3)
    r["mx/namedmax"] = round(float(np.median(mx[ok] / named_max[ok])), 4)
    if g["BFEH"].notna().mean() > 0.5:
        v = np.concatenate([g[f"BFE{o}"].to_numpy() for o in O])
        ok = np.isfinite(v) & np.isfinite(mx)
        r["mx<BFE"] = round(float((mx[ok] < v[ok] - 1e-9).mean()), 3)
        r["mx==BFE"] = round(float((np.abs(mx[ok] - v[ok]) < 1e-9).mean()), 3)
    # overround of the max 'book'
    ov = (1 / g[[f"MX{o}" for o in O]]).sum(axis=1)
    r["max_overround_med"] = round(float(ov.median()), 4)
    r["max_overround<0.95"] = round(float((ov < 0.95).mean()), 4)
    rows.append(r)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
print(pd.DataFrame(rows).set_index("season"))
