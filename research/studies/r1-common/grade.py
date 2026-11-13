"""Family-level grading for polymarket-efficiency: Benjamini-Hochberg (q = 0.10) across every evaluation trial.

Reads every `research/studies/<id>/trials.csv` whose registration names family `polymarket-efficiency`,
keeps the evaluation-window rows (exploratory rows are design runs and are listed separately), and applies
BH to the one-sided p-values. A trial with fewer than MIN_DAYS distinct match-days has no usable bootstrap
p-value (with one or two blocks the bootstrap has no spread) and enters BH with p = 1. Writes
data/research/r1/family_grade.csv.
"""

from __future__ import annotations

import pandas as pd
import yaml
from common import DATA, REGISTRY, REPO, bh

FAMILY = "polymarket-efficiency"
MIN_DAYS = 20


def main() -> None:
    ids = []
    for p in sorted(REGISTRY.glob("*.yaml")):
        reg = yaml.safe_load(p.read_text())
        if reg.get("family") == FAMILY:
            ids.append(reg["id"])
    frames = []
    for i in ids:
        path = REPO / "research" / "studies" / i / "trials.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
    led = pd.concat(frames, ignore_index=True)
    print("ledger rows by id/window:\n", led.groupby(["id", "window"]).size())
    ev = led[led.window == "evaluation"].copy()
    ev["p_for_bh"] = ev.p_one_sided.where(ev.n_days >= MIN_DAYS, 1.0).fillna(1.0)
    ev["p_bh"], ev["bh_reject_q10"] = bh(ev.p_for_bh.to_numpy(), 0.10)
    cols = [
        "id",
        "config",
        "n",
        "n_days",
        "value",
        "ci_lo",
        "ci_hi",
        "p_one_sided",
        "p_for_bh",
        "p_bh",
        "bh_reject_q10",
    ]
    ev[cols].to_csv(DATA / "family_grade.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_colwidth", 90, "display.max_rows", 100):
        print(ev[cols])
    print(f"family trials (evaluation): {len(ev)}; BH rejections at q=0.10: {int(ev.bh_reject_q10.sum())}")


if __name__ == "__main__":
    main()
