"""POST-HOC diagnostics for r5-xv-arb-v1 (not registered trials; nothing is written to the ledger).

The Polymarket book archive holds about one snapshot per market every four hours, so the "book in force"
at H (and at H + 1h) is often hours old. A stale Polymarket quote can both create a cross-venue signal and
"fill" it an hour later at a price nobody may have been offering. This splits the book-priced P&L by the age
of the Polymarket book on the signal's Polymarket legs at H and at H + 1h, and by whether the H + 1h fill uses
a newer Polymarket snapshot than the signal did. Writes data/research/r5/r5-xv-arb-v1/diagnostics.json.

Usage: arch -arm64 uv run python research/studies/r5-xv-arb-v1/diagnostics.py
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

from partb import R5, panel, universe  # noqa: E402
from run import signals, wide  # noqa: E402

from pitch_edge.research.stats import block_bootstrap  # noqa: E402


def summary(x: pd.DataFrame) -> dict:
    if len(x) < 2:
        return {"n": len(x)}
    b = block_bootstrap(x.pnl.to_numpy(), x.match_day.to_numpy(), n_boot=5000, seed=20261020)
    return {
        "n": b.n,
        "events": int(x.event_ticker.nunique()),
        "days": b.n_blocks,
        "mean_pnl": b.estimate,
        "lo": b.lo,
        "hi": b.hi,
    }


def main() -> None:
    u, _ = universe()
    u = u[u.window == "evaluation"]
    pan = panel(u)
    age = pan.set_index(["event_ticker", "side", "H"]).b_age_h
    w = wide(pan, "book")
    out: dict = {}
    for typ in ("cross", "dutch"):
        sg = signals(w, typ)
        sg = sg[np.isfinite(sg.edge) & np.isfinite(sg.pnl)].copy()
        pm_sides = [[leg.split("_")[2] for leg in legs.split("+") if "_P_" in leg] for legs in sg.legs]
        a_now, a_next = [], []
        for r, sides in zip(sg.itertuples(index=False), pm_sides, strict=True):
            an = [age.get((r.event_ticker, s, r.H), np.nan) for s in sides]
            ax = [age.get((r.event_ticker, s, r.H + pd.Timedelta(hours=1)), np.nan) for s in sides]
            a_now.append(max(an) if an else np.nan)
            a_next.append(max(ax) if ax else np.nan)
        sg["pm_age_at_signal_h"] = a_now
        sg["pm_age_at_fill_h"] = a_next
        # A fill uses a newer Polymarket snapshot than the signal iff its age is smaller than signal age + 1h.
        sg["new_pm_snapshot_at_fill"] = sg.pm_age_at_fill_h < sg.pm_age_at_signal_h + 1 - 1e-6
        for th in (0.0, 0.01, 0.02):
            s = sg[sg.edge > th]
            key = f"{typ}|{th}"
            out[key] = {
                "all": summary(s),
                "pm_book_fresh_at_signal_and_fill_le_1h": summary(
                    s[(s.pm_age_at_signal_h <= 1) & (s.pm_age_at_fill_h <= 1)]
                ),
                "fill_uses_new_pm_snapshot": summary(s[s.new_pm_snapshot_at_fill]),
                "fill_uses_same_pm_snapshot": summary(s[~s.new_pm_snapshot_at_fill]),
                "pm_age_at_signal_quantiles": s.pm_age_at_signal_h.quantile([0.25, 0.5, 0.75]).round(2).tolist()
                if len(s)
                else None,
                "edge_at_signal_max": float(s.edge.max()) if len(s) else None,
                "top_events_share_of_signals": float(s.event_ticker.value_counts().head(5).sum() / max(len(s), 1)),
            }
    dest = R5 / "r5-xv-arb-v1"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "diagnostics.json").write_text(json.dumps(out, indent=2, default=str))
    for k, v in out.items():
        print(k)
        for kk, vv in v.items():
            print("   ", kk, vv)


if __name__ == "__main__":
    main()
