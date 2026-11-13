"""EXPLORATORY ONLY (pre-registration design work). Reads match rows dated <= 2019-06-30 and nothing later.

Walk-forward over exploratory seasons 2014/15-2018/19 (fit on all earlier seasons from 2012/13) for the
baseline and every candidate model, plus H3-style per-book slopes of the Pinnacle open->close move on the
book's open deviation. Every variant run here (default four, or the models named on the command line,
appended) is listed in data/research/r6/explore/variants.csv so the forking
paths are visible. Usage: explore.py [model ...]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r6lib as L  # noqa: E402

from pitch_edge.research.stats import block_bootstrap  # noqa: E402

OUT = L.OUT / "explore"


def slope(x: np.ndarray, y: np.ndarray) -> float:
    xc = x - x.mean()
    return float(np.sum(xc * (y - y.mean())) / np.sum(xc * xc))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = L.load_panel()
    print("panel", panel.shape, panel.groupby("segment").size().to_dict())
    print("by season", panel.groupby("season_start").size().to_dict())
    ex = panel[panel["match_date"] <= L.EXPLORATORY_END].copy()
    assert ex["match_date"].max() <= L.EXPLORATORY_END
    del panel  # nothing after the exploratory window is used below
    rows = []
    seasons = [2014, 2015, 2016, 2017, 2018]
    models = sys.argv[1:] or ["ols", "ridge", "hgb_shallow", "hgb_deeper"]
    prev = pd.read_csv(OUT / "variants.csv") if (OUT / "variants.csv").exists() and sys.argv[1:] else None
    for model in models:
        t0 = time.time()
        s = L.add_r2_value(L.walk_forward(ex, model, "power", seasons))
        day = s["match_date"].dt.strftime("%Y-%m-%d").to_numpy()
        b = block_bootstrap(s["delta_kl_bits"].to_numpy(), day, n_boot=1000, seed=L.SEED)
        r2 = block_bootstrap(s["delta_r2_move"].to_numpy(), day, n_boot=1000, seed=L.SEED)
        per = s.groupby("season_start")["delta_kl_bits"].mean().to_dict()
        row = {
            "model": model,
            "devig": "power",
            "seasons": "2014-2018",
            "n": len(s),
            "delta_kl_bits": b.estimate,
            "lo": b.lo,
            "hi": b.hi,
            "delta_r2_pp": 100 * r2.estimate,
            "r2_lo": 100 * r2.lo,
            "r2_hi": 100 * r2.hi,
            "kl_open": s["kl_open"].mean(),
            "kl_base": s["kl_base"].mean(),
            "kl_model": s["kl_model"].mean(),
            "r2_base": 1 - s["sse_base"].sum() / s["sse0"].sum(),
            "r2_model": 1 - s["sse_model"].sum() / s["sse0"].sum(),
            "by_season": json.dumps({int(k): round(v, 7) for k, v in per.items()}),
            "lambdas": s["ridge_lambda"].iloc[0] if model == "ridge" else "",
            "secs": round(time.time() - t0, 1),
        }
        rows.append(row)
        print({k: (round(v, 7) if isinstance(v, float) else v) for k, v in row.items()})
    new = pd.DataFrame(rows)
    (pd.concat([prev, new], ignore_index=True) if prev is not None else new).to_csv(OUT / "variants.csv", index=False)
    if sys.argv[1:]:
        return

    # H3-style descriptive slopes on the exploratory window (all of 2012/13-2018/19).
    f = L.build_features(ex, "power")
    t = L.targets(ex, "power")
    day = np.repeat(ex["match_date"].dt.strftime("%Y-%m-%d").to_numpy()[:, None], 3, axis=1)
    g_avg = np.column_stack([f.base[k][:, 0] for k in range(3)])
    out = {}
    for b in L.CORE_BOOKS:
        x = f.dev[b]
        ok = np.isfinite(x) & np.isfinite(t["move"]) & (np.abs(x) <= L.DEV_CLIP)
        xs, ys, gs = x[ok], t["move"][ok], g_avg[ok]
        a = np.column_stack([np.ones(ok.sum()), xs, gs])
        partial = float(np.linalg.lstsq(a, ys, rcond=None)[0][1])
        out[b] = {
            "n_rows": int(ok.sum()),
            "marginal": slope(xs, ys),
            "partial_given_avg": partial,
            "sd_dev": float(xs.std()),
            "mean_abs_dev": float(np.abs(xs).mean()),
        }
        print(b, {k: round(v, 4) if isinstance(v, float) else v for k, v in out[b].items()})
    a_ok = np.isfinite(g_avg) & np.isfinite(t["move"])
    out["avg_consensus"] = {"marginal": slope(g_avg[a_ok], t["move"][a_ok])}
    print("avg", out["avg_consensus"])
    (OUT / "h3_slopes.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
