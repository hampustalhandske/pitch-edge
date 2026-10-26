"""Exploratory (seasons before 2019/20 only): is the external `home_elo`/`away_elo` column pre-match?

For every team's consecutive appearances i -> i+1, regress the change in its external Elo on the surprise of match i and
on the surprise of match i+1 (points scored minus the Elo-expected score from the pre-match rating). A pre-match rating
moves with surprise i and not with surprise i+1; a post-match rating does the opposite.
"""
import numpy as np
import pandas as pd

f = pd.read_parquet("data/research/r7/features.parquet")
f["date"] = pd.to_datetime(f["date"])
f = f[(f["date"] < "2019-07-01") & f["home_elo"].notna() & f["away_elo"].notna()]
rows = []
for side, opp, hadv in (("home", "away", 60.0), ("away", "home", -60.0)):
    elo, oelo = f[f"{side}_elo"], f[f"{opp}_elo"]
    exp = 1 / (1 + 10 ** ((oelo - elo - hadv) / 400.0)) if side == "home" else 1 / (1 + 10 ** ((oelo + 60.0 - elo) / 400.0))
    if side == "home":
        score = np.where(f.home_goals > f.away_goals, 1.0, np.where(f.home_goals == f.away_goals, 0.5, 0.0))
    else:
        score = np.where(f.away_goals > f.home_goals, 1.0, np.where(f.home_goals == f.away_goals, 0.5, 0.0))
    rows.append(pd.DataFrame({"team": f[f"{side}_team"], "date": f["date"], "mid": f["match_id"], "elo": elo, "surprise": score - exp}))
L = pd.concat(rows).sort_values(["team", "date", "mid"])
g = L.groupby("team")
L["d_next"] = g["elo"].shift(-1) - L["elo"]
L["s_next"] = g["surprise"].shift(-1)
L = L.dropna(subset=["d_next", "s_next"])
X = np.column_stack([np.ones(len(L)), L["surprise"], L["s_next"]])
b, *_ = np.linalg.lstsq(X, L["d_next"].to_numpy(), rcond=None)
print(f"n={len(L)}  d_elo(i->i+1) = {b[0]:+.2f} + {b[1]:+.2f}*surprise_i + {b[2]:+.2f}*surprise_(i+1)")
