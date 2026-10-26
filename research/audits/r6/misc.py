"""Misc checks: duplicate fixtures dropped, early same-day kickoffs (snapshot could post-date kickoff), coarser
(ISO-week) block bootstrap of the headline Δs, Δ by division for the auditor's ridge vs the era baseline.

Usage: OMP_NUM_THREADS=1 arch -arm64 uv run python research/audits/r6/misc.py
"""

from __future__ import annotations

import json

import alib as A
import pandas as pd

raw = A.raw()
d0 = raw[(raw.match_date >= "2012-07-01") & (raw.match_date <= "2025-06-30")]
key = d0.division + "|" + d0.match_date.dt.strftime("%Y-%m-%d") + "|" + d0.home_team + "|" + d0.away_team
res = {"duplicate_fixture_rows": int(key.duplicated().sum())}
w = pd.read_parquet(A.OUT / "rederive_ridge.parquet")
hrs = pd.read_parquet(A.OUT / "hours.parquet").drop_duplicates("mid").set_index("mid").hrs
w["hrs"] = hrs.reindex(w.mid).to_numpy()
early = w[w.hrs <= 0]
res["kickoff_at_or_before_16h_snapshot"] = {
    "n": len(early),
    "delta_mean": float(early.delta.mean()),
    "kl_open": float(early.kl_open.mean()),
    "kl_open_all": float(w.kl_open.mean()),
}
res["kickoff_before_13h_snapshot_day"] = {
    "n": int((w.hrs <= -3).sum()),
    "delta_mean": float(w.delta[w.hrs <= -3].mean()),
}
wk = w.match_date.dt.strftime("%G-%V").to_numpy()
st = {}
for tag, f in (
    ("ols", "closing-line-prediction-cross-book-linear-v1__ols"),
    ("ridge", "closing-line-prediction-cross-book-linear-v1__ridge"),
    ("hgb_tiny", "closing-line-prediction-cross-book-gbm-v1__hgb_tiny"),
    ("res_tiny", "closing-line-prediction-cross-book-gbm-v1__hgbres_tiny"),
):
    s = pd.read_parquet(A.REPO / "data/research/r6" / f"{f}.parquet")
    s = s[s.segment == "pinnacle"]
    st[tag] = {
        "iso_week": A.boot(s.delta_kl_bits.to_numpy(), pd.to_datetime(s.match_date).dt.strftime("%G-%V").to_numpy()),
        "division_season": A.boot(s.delta_kl_bits.to_numpy(), (s.division + s.season_start.astype(str)).to_numpy()),
    }
res["coarse_bootstrap_study_frames"] = st
print(json.dumps(res, indent=1))
(A.REPO / "research/audits/r6/out/misc.json").write_text(json.dumps(res, indent=1))
