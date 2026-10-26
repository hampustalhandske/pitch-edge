"""Benjamini-Hochberg (q = 0.10) over a whole multiple-testing family, including trials that predate the
harness ledger.

The R1 polymarket-efficiency trials were recorded in `research/studies/r1-*/trials.csv` before the harness
ledger existed, so `pitch-edge research grade` (which reads only the ledger) does not see them. This script
pools, per family: every primary-metric trial in the ledger (first run per hypothesis and config) and, for
polymarket-efficiency, R1's evaluation-window rows from those CSVs unless the ledger already holds that
R1 id (they are imported into the ledger from 2026-10-20 21:22 on) (a trial with fewer than 20 match-days
enters with p = 1, R1's own rule). It prints and writes data/research/r5/family_bh_<family>.csv.

Usage: arch -arm64 uv run python research/studies/r5-common/family_bh.py <family>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from harness import REPO, ledger  # noqa: E402

from pitch_edge.research.stats import benjamini_hochberg  # noqa: E402

R1_IDS = ["r1-pm-accuracy-v1", "r1-pm-bias-v1", "r1-pm-convergence-v1", "r1-pm-gap-trade-v1"]


def main(family: str) -> None:
    t = ledger().trials(family=family)
    t = t[[json.loads(e).get("is_primary_metric", True) for e in t.extra_json]]
    t = t.sort_values(["run_at", "trial_id"]).groupby(["hypothesis_id", "config_hash"]).head(1)
    rows = [
        {
            "source": "ledger",
            "id": r.hypothesis_id,
            "config": r.config_json,
            "n": r.n,
            "value": r.metric_value,
            "p": r.p_value,
        }
        for r in t.itertuples(index=False)
    ]
    in_ledger = set(t.hypothesis_id)
    if family == "polymarket-efficiency":
        for i in [x for x in R1_IDS if x not in in_ledger]:
            d = pd.read_csv(REPO / "research" / "studies" / i / "trials.csv")
            d = d[d.window == "evaluation"]
            for r in d.itertuples(index=False):
                p = r.p_one_sided if (r.n_days >= 20 and pd.notna(r.p_one_sided)) else 1.0
                rows.append(
                    {"source": "r1 trials.csv", "id": r.id, "config": r.config, "n": r.n, "value": r.value, "p": p}
                )
    df = pd.DataFrame(rows)
    df["p"] = df.p.fillna(1.0)
    df["reject_q10"], df["q_bh"] = benjamini_hochberg(df.p.to_numpy(), q=0.10)
    out = REPO / "data" / "research" / "r5" / f"family_bh_{family}.csv"
    df.to_csv(out, index=False)
    with pd.option_context("display.width", 250, "display.max_colwidth", 70, "display.max_rows", 200):
        print(df.sort_values("p"))
    print(f"{family}: {len(df)} tests, {int(df.reject_q10.sum())} rejected at q = 0.10")


if __name__ == "__main__":
    main(sys.argv[1])
