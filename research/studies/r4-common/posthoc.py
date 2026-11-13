"""POST-HOC, not counted: descriptive follow-ups on the R4 evaluation window, run after the six registered trials.

1. The crowd-feature coefficients fitted on the evaluation rows (with the baseline), with match-day block-bootstrap
   CIs, and the implied move per one standard deviation of the feature, set against Polymarket's crossing cost
   (taker fee 0.05 p (1 - p) at p = 0.5 plus a one-tick half-spread: ~1.75 cents per side).
2. Δ OOS R² of the walk-forward predictions (from the registered runs) by real league (top 6) and by hours to
   kickoff (24-13h vs 12-2h).
Writes `posthoc.json`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "r5-common"))

from explore import boot_coefs  # noqa: E402
from model import H1_BASE, H1_CROWD, H2_BASE, H2_CROWD, h1_frame, h2_frame, per_decision  # noqa: E402
from panel import DATA  # noqa: E402

from pitch_edge.research.stats import block_bootstrap  # noqa: E402

EVAL_FROM = pd.Timestamp("2026-03-01", tz="UTC")
COST_CENTS = 100 * (0.05 * 0.25 + 0.005)
IDS = {
    "h1": "r4-bsky-attention-size-v1",
    "mention": "r4-bsky-mention-skew-v1",
    "sentiment": "r4-bsky-sentiment-skew-v1",
}


def _ci(dec: pd.DataFrame) -> list:
    if dec.block.nunique() < 2:
        return [round(float(dec.value.mean()), 6), None, None, len(dec)]
    b = block_bootstrap(dec.value.to_numpy(), dec.block.to_numpy(), n_boot=2000, seed=7)
    return [round(float(b.estimate), 6), round(float(b.lo), 6), round(float(b.hi), 6), int(b.n)]


def main() -> None:
    panel = pd.read_parquet(DATA / "panel.parquet")
    ev = panel[panel.ko >= EVAL_FROM]
    out: dict = {"label": "POST-HOC, not counted", "crossing_cost_cents_per_side": COST_CENTS}
    for kind, reg_id in IDS.items():
        for L in (6, 24):
            if kind == "h1":
                f, cols, crowd = h1_frame(ev, L), H1_BASE + H1_CROWD, H1_CROWD
            else:
                f, cols, crowd = h2_frame(ev, L, kind), H2_BASE + H2_CROWD, H2_CROWD
            co = boot_coefs(f, cols)
            sd = {c: float(f[c].std()) for c in crowd}
            pred = pd.read_parquet(DATA / reg_id / f"pred_L{L}.parquet")
            by_league = {}
            for lg in pred.league.value_counts().head(6).index:
                d, _ = per_decision(pred[pred.league == lg])
                by_league[lg] = _ci(d)
            by_h = {}
            for lab, m in (("24-13h", pred.h_before >= 13), ("12-2h", pred.h_before <= 12)):
                d, _ = per_decision(pred[m])
                by_h[lab] = _ci(d)
            out[f"{kind}_L{L}"] = {
                "coef_eval_in_sample": {c: co[c] for c in crowd},
                "feature_sd": sd,
                "implied_move_cents_per_sd": {c: round(100 * co[c][0] * sd[c], 4) for c in crowd},
                "delta_oos_r2_by_league": by_league,
                "delta_oos_r2_by_hours_to_ko": by_h,
                "y_sd_cents": round(100 * float(f.y.std()), 3),
            }
    (DATA / "posthoc.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    sys.exit(main())
