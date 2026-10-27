"""Point-in-time assertions for the R7 inputs (writes data/research/r7/pit.json; raises on failure).

1. Feature-builder truncation invariance: features rebuilt from a spine cut at CUT (matches with date < CUT) equal the
   full-spine features for every match before CUT, for every column the models read. A feature that peeked at a later
   result, a later match or a full-sample statistic would differ.
2. Walk-forward truncation invariance: the predictions of the run on the frame cut at 2020-06-30 equal the full run's
   for every match up to the cut (raw and calibrated probabilities, all three models).
3. Fold discipline: for every prediction train_max_date < fold_start <= match date and match date - fold_start < 14 days.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pitch_edge.data.storage import Warehouse
from pitch_edge.features.build import FeatureBuilder, feature_columns
from pitch_edge.features.context import load_context
from pitch_edge.pipeline import DEFAULT_BACKTEST_LEAGUES

OUT = Path(__file__).resolve().parents[3] / "data" / "research" / "r7"
CUT = pd.Timestamp("2022-01-01")
MODELS = ("dixon_coles", "gbdt", "stochastic_strength")


def feature_truncation() -> dict:
    full = pd.read_parquet(OUT / "features.parquet")
    full["date"] = pd.to_datetime(full["date"])
    wh = Warehouse(read_only=True)
    matches = wh.matches_with_closing_odds(bookmakers=("PS", "B365", "Avg", "Mkt", "Max", "BFE"))
    matches = matches[matches["league_code"].isin(DEFAULT_BACKTEST_LEAGUES)]
    matches = matches[pd.to_datetime(matches["date"]) >= pd.Timestamp("2015-07-01")]
    matches = matches[pd.to_datetime(matches["date"]) < CUT]
    ctx = load_context(wh, matches)
    cut = FeatureBuilder().build(matches, weather=wh.read("weather"), venues=wh.read("venues"), context=ctx)
    cols = [c for c in feature_columns(full, include_market=True) if c in cut.columns and c != "league_id"]
    unused = ("pv_", "rot_", "sv_")  # feature groups no `ask` model reads (GBDTMatchModel excludes them)
    a = full[full["date"] < CUT].set_index("match_id")[cols].sort_index()
    b = cut.set_index("match_id")[cols].sort_index()
    assert list(a.index) == list(b.index), "row sets differ"
    # (referee, date) pairs that occur on two matches the same day (a data-entry artefact): the shift(1) order inside
    # such a pair is arbitrary, so its two rows may legitimately differ
    ref = full[full["date"] < CUT].dropna(subset=["referee"])
    dup = ref[ref.duplicated(["referee", "date"], keep=False)]["match_id"]
    dup_ids = set(dup)
    differing_used, differing_unused, unexplained, cols_unexplained = {}, {}, [], set()
    for c in cols:
        x, y = a[c].to_numpy(dtype=float), b[c].to_numpy(dtype=float)
        bad = ~np.isclose(x, y, rtol=1e-9, atol=1e-9, equal_nan=True)
        if not bad.any():
            continue
        (differing_unused if c.startswith(unused) else differing_used)[c] = int(bad.sum())
        if not c.startswith(unused):
            miss = [m for m in a.index[bad] if m not in dup_ids]
            unexplained += miss
            if miss:
                cols_unexplained.add(c)
    ev = full[(full["date"] >= "2019-07-01") & (full["date"] <= "2025-06-30")].dropna(subset=["referee"])
    ev_dup_rows = int(ev.duplicated(["referee", "date"], keep=False).sum())
    return {
        "cut": str(CUT.date()),
        "rows_compared": int(len(a)),
        "columns_compared": len(cols),
        "columns_differing": {c: n for c, n in differing_used.items() if c in cols_unexplained},
        "used_columns_differing_only_on_same_day_referee_pairs": {
            c: n for c, n in differing_used.items() if c not in cols_unexplained
        },
        "differing_rows_not_a_same_day_referee_pair": sorted(set(unexplained)),
        "unused_group_columns_differing": differing_unused,
        "evaluation_window_rows_in_same_day_referee_pairs": ev_dup_rows,
    }


def prediction_truncation() -> dict:
    out = {}
    for m in MODELS:
        full = pd.read_parquet(OUT / f"pred_{m}.parquet").set_index("match_id")
        cut = pd.read_parquet(OUT / f"pred_{m}_cut.parquet").set_index("match_id")
        sub = full.loc[full.index.isin(cut.index)].sort_index()
        cut = cut.sort_index()
        assert list(sub.index) == list(cut.index), f"{m}: matches differ"
        cols = [f"raw_{o}" for o in ("home", "draw", "away")] + [f"p_{o}" for o in ("home", "draw", "away")]
        diff = float(np.nanmax(np.abs(sub[cols].to_numpy(dtype=float) - cut[cols].to_numpy(dtype=float))))
        out[m] = {"rows_compared": int(len(sub)), "max_abs_diff": diff}
    return out


def fold_discipline() -> dict:
    out = {}
    for m in MODELS:
        p = pd.read_parquet(OUT / f"pred_{m}.parquet")
        ok1 = bool((p["train_max_date"] < p["fold_start"]).all())
        ok2 = bool((p["fold_start"] <= p["date"]).all())
        ok3 = bool(((p["date"] - p["fold_start"]).dt.days < 14).all())
        out[m] = {"rows": int(len(p)), "train_before_fold": ok1, "fold_before_match": ok2, "within_14d": ok3}
        assert ok1 and ok2 and ok3, f"{m}: fold discipline violated"
    return out


def main() -> None:
    res = {
        "feature_truncation": feature_truncation(),
        "prediction_truncation": prediction_truncation(),
        "fold_discipline": fold_discipline(),
    }
    (OUT / "pit.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    ft = res["feature_truncation"]
    assert not ft["columns_differing"] and not ft["differing_rows_not_a_same_day_referee_pair"], (
        "feature builder is not truncation-invariant"
    )
    for m, r in res["prediction_truncation"].items():
        assert r["max_abs_diff"] < 1e-9, f"{m}: predictions depend on later rows ({r['max_abs_diff']})"


if __name__ == "__main__":
    main()
