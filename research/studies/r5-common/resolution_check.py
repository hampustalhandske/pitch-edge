"""Mapping diagnostic: do Kalshi and Polymarket resolve each matched outcome the same way?

Uses hindsight columns (Kalshi `result`, Polymarket `winning_outcome_label`) only to validate the
Kalshi<->Polymarket mapping and to size the resolution risk that a "riskless" cross-venue combination
carries; nothing here feeds a decision. Writes data/research/r5/resolution_check.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from partb import R5, SIDES, universe  # noqa: E402


def main() -> None:
    u, _ = universe()
    km = pd.read_parquet(R5 / "kalshi_markets.parquet").set_index("ticker")
    pm = pd.read_parquet(R5 / "pm_1x2.parquet").drop_duplicates("condition_id").set_index("condition_id")
    rows = []
    for r in u.itertuples(index=False):
        for s in SIDES:
            kr = km.result.get(getattr(r, f"tk_{s}"))
            pl = pm.winning_outcome_label.get(getattr(r, f"cond_{s}"))
            rows.append(
                {
                    "event": r.event_ticker,
                    "side": s,
                    "k": None if kr is None else str(kr).lower(),
                    "p": None if pl is None else str(pl).lower(),
                }
            )
    d = pd.DataFrame(rows)
    both = d[d.k.isin(["yes", "no"]) & d.p.isin(["yes", "no"])]
    ev = both.groupby("event").apply(lambda g: bool((g.k == g.p).all()) and len(g) == 3)
    out = {
        "outcomes": len(d),
        "outcomes_both_resolved_yes_no": len(both),
        "kalshi_result_values": d.k.value_counts(dropna=False).to_dict(),
        "pm_label_values": d.p.value_counts(dropna=False).head(10).to_dict(),
        "outcome_agreement": float((both.k == both.p).mean()),
        "events_fully_compared": int(ev.size),
        "events_all_three_agree": int(ev.sum()),
        "disagreeing_events_sample": ev[~ev].index[:15].tolist(),
    }
    (R5 / "resolution_check.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
