"""EXPLORATORY ONLY (not counted): R4 on the exploratory window (kickoff 2026-01-20 -> 2026-02-28).

Reads `panel.parquet`, keeps rows with kickoff before EXPLORATORY_END (asserted), and for each of the six planned
grid points reports: rows/decisions, target spread, the in-sample crowd coefficients with match-day block-bootstrap
CIs, and a single pseudo-OOS split (train < 2026-02-10, test 2026-02-10 -> 2026-02-28) Δ OOS R². Also the
bounce check that motivates the target: the slope of the move measured from p_H on the own recent move versus the
same slope for the move measured from p_0 (the first bar after H). Writes `exploratory.json`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "r5-common"))

from harness import exact_slope_ci  # noqa: E402  (r5-common)
from model import H1_BASE, H1_CROWD, H2_BASE, H2_CROWD, design, h1_frame, h2_frame, ols, per_decision  # noqa: E402
from panel import DATA  # noqa: E402

EXPLORATORY_END = pd.Timestamp("2026-03-01", tz="UTC")
SPLIT = "2026-02-10"


def boot_coefs(df: pd.DataFrame, cols: list[str], n_boot: int = 1000, seed: int = 1) -> dict:
    fe = sorted(df.h_before.astype(int).unique(), reverse=True)
    X = design(df, cols, fe)
    b = ols(X, df.y.to_numpy())
    k = len(cols)
    blocks = df.block.to_numpy()
    uniq = np.unique(blocks)
    idx = {u: np.where(blocks == u)[0] for u in uniq}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        pick = np.concatenate([idx[u] for u in rng.choice(uniq, len(uniq))])
        draws.append(ols(X[pick], df.y.to_numpy()[pick])[-k:])
    lo, hi = np.percentile(np.array(draws), [2.5, 97.5], axis=0)
    return {c: [round(float(b[-k + i]), 5), round(float(lo[i]), 5), round(float(hi[i]), 5)] for i, c in enumerate(cols)}


def pseudo_oos(df: pd.DataFrame, base: list[str], crowd: list[str]) -> dict:
    fe = sorted(df.h_before.astype(int).unique(), reverse=True)
    tr = df[df.ko < pd.Timestamp(SPLIT, tz="UTC")]
    te = df[df.ko >= pd.Timestamp(SPLIT, tz="UTC")].copy()
    te["pred_base"] = design(te, base, fe) @ ols(design(tr, base, fe), tr.y.to_numpy())
    te["pred_full"] = design(te, base + crowd, fe) @ ols(design(tr, base + crowd, fe), tr.y.to_numpy())
    te["ybar_train"] = tr.y.mean()
    _, s = per_decision(te)
    return {k: round(v, 5) if isinstance(v, float) else v for k, v in s.items()}


def main() -> None:
    panel = pd.read_parquet(DATA / "panel.parquet")
    panel = panel[panel.ko < EXPLORATORY_END]
    assert panel.ko.max() < EXPLORATORY_END
    out: dict = {"label": "EXPLORATORY, not counted", "window": ["2026-01-20", "2026-02-28"]}
    for L in (6, 24):
        f1 = h1_frame(panel, L)
        out[f"h1_L{L}"] = {
            "rows": len(f1),
            "decisions": int(f1.decision_id.nunique()),
            "days": int(f1.block.nunique()),
            "y_mean": round(float(f1.y.mean()), 5),
            "y_share_zero": round(float((f1.y == 0).mean()), 4),
            "in_sample_coefs": boot_coefs(f1, H1_BASE + H1_CROWD),
            "pseudo_oos": pseudo_oos(f1, H1_BASE, H1_CROWD),
        }
        for kind in ("mention", "sentiment"):
            f2 = h2_frame(panel, L, kind)
            out[f"h2_{kind}_L{L}"] = {
                "rows": len(f2),
                "decisions": int(f2.decision_id.nunique()),
                "days": int(f2.block.nunique()),
                "y_sd": round(float(f2.y.std()), 5),
                "skew_sd": round(float(f2["skew"].std()), 4),
                "in_sample_coefs": boot_coefs(f2, H2_BASE + H2_CROWD),
                "pseudo_oos": pseudo_oos(f2, H2_BASE, H2_CROWD),
            }
    # Bounce check (one row per match-hour, home side): move from p_H vs move from p_0, on the own 6h move.
    d = panel.dropna(subset=["p_H_home", "p_H6_home", "p_0_home", "p_end_home"])
    x = (d.p_H_home - d.p_H6_home).to_numpy()
    blocks = d.ko.dt.strftime("%Y-%m-%d").to_numpy()
    out["bounce_check_home"] = {
        "slope_move_from_pH": exact_slope_ci(x, (d.p_end_home - d.p_H_home).to_numpy(), blocks, n_boot=1000),
        "slope_move_from_p0": exact_slope_ci(x, (d.p_end_home - d.p_0_home).to_numpy(), blocks, n_boot=1000),
    }
    (DATA / "exploratory.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    sys.exit(main())
