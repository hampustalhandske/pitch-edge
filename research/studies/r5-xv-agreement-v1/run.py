"""r5-xv-agreement-v1: do Kalshi and Polymarket agree on the same soccer 1X2 outcome in the same hour?

Per (event, side, H) with a Kalshi quote and a fresh Polymarket trade in the hour ending at H:
  gap  = Kalshi mid - Polymarket last trade
  cost = Kalshi taker fee + Polymarket taker fee (both at the average price) + Kalshi half-spread
         + Polymarket half-spread proxy max(0.5c, half the bar's high-low)   (research/STANDARDS.md proxy)
  exceed = |gap| > cost   (the mids are further apart than it costs to buy one side and sell the other)
Primary value per row: exceed - 0.05 (H1: more than 5% of outcome-hours exceed the round-trip cost).
Secondaries: mean signed gap, mean |gap|, and the exceedance share with the true Polymarket book (mid and
half-spread) where the book exists (2026-04-09 -> 2026-06-10, 2026-09-29).

Usage: arch -arm64 uv run python research/studies/r5-xv-agreement-v1/run.py {exploratory|evaluation}
  exploratory: prints the design numbers for the exploratory window; nothing is written to the ledger
  evaluation:  records one counted trial per hours-to-kickoff bucket (plus secondaries) in the ledger
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "r5-common"))

from harness import record  # noqa: E402
from partb import CODE_FILES, R5, panel, snapshot_tag, universe  # noqa: E402
from prices import kalshi_fee, pm_fee  # noqa: E402

REG_ID = "r5-xv-agreement-v1"
THRESHOLD = 0.05
BUCKETS = {"1-6h": (0, 6), "7-24h": (6, 24), "25-48h": (24, 48)}


def frames(pan: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(last-trade frame, book frame) with gap, cost and exceed per row."""
    f = pan[pan.p_fresh.notna() & pan.mid.notna()].copy()
    pbar = (f.mid + f.p_fresh) / 2
    f["gap"] = f.mid - f.p_fresh
    f["hs_k"] = (f.ask - f.bid) / 2
    f["hs_p"] = np.maximum(0.005, (f.hi - f.lo) / 2)
    f["cost"] = kalshi_fee(pbar) + pm_fee(pbar) + f.hs_k + f.hs_p
    f["exceed"] = (f.gap.abs() > f.cost).astype(float)
    b = pan[pan.b_mid.notna() & pan.mid.notna()].copy()
    pbar = (b.mid + b.b_mid) / 2
    b["gap"] = b.mid - b.b_mid
    b["cost"] = kalshi_fee(pbar) + pm_fee(pbar) + (b.ask - b.bid) / 2 + (b.b_ask - b.b_bid) / 2
    b["exceed"] = (b.gap.abs() > b.cost).astype(float)
    for x in (f, b):
        x["decision_id"] = x.event_ticker + "|" + x.side + "|" + x.H.dt.strftime("%Y-%m-%dT%H")
        x["block"] = x.match_day
    return f, b


def describe(f: pd.DataFrame) -> dict:
    g = f.gap
    return {
        "n": len(f),
        "events": int(f.event_ticker.nunique()),
        "days": int(f.block.nunique()),
        "share_exceed": float(f.exceed.mean()) if len(f) else None,
        "mean_gap": float(g.mean()) if len(f) else None,
        "mean_abs_gap": float(g.abs().mean()) if len(f) else None,
        "abs_gap_quantiles": {str(q): float(g.abs().quantile(q)) for q in (0.5, 0.75, 0.9, 0.95, 0.99)}
        if len(f)
        else None,
        "mean_cost": float(f.cost.mean()) if len(f) else None,
    }


def main(window: str) -> None:
    u, drops = universe()
    u = u[u.window == window]
    pan = panel(u)
    f, b = frames(pan)
    stale = {
        "kalshi_rows_with_mid": int(pan.mid.notna().sum()),
        "share_without_fresh_pm_trade": float(1 - pan.p_fresh.notna()[pan.mid.notna()].mean()),
        "pm_last_trade_age_h_quantiles": {str(q): float(pan.p_age_h.quantile(q)) for q in (0.25, 0.5, 0.75, 0.9)},
        "kalshi_rows_without_two_sided_quote": int(pan.mid.isna().sum()),
    }
    detail = {"window": window, "drops": drops, "events": len(u), "staleness": stale, "buckets": {}}
    for name, (lo, hi) in BUCKETS.items():
        fs = f[(f.hours_to_ko > lo) & (f.hours_to_ko <= hi)].reset_index(drop=True)
        bs = b[(b.hours_to_ko > lo) & (b.hours_to_ko <= hi)].reset_index(drop=True)
        detail["buckets"][name] = {"lasttrade": describe(fs), "book": describe(bs)}
        if window != "evaluation":
            continue
        snap = snapshot_tag()
        cf = CODE_FILES + [Path(__file__)]
        days = fs.block.to_numpy()

        def share_on(m: np.ndarray, fs=fs) -> float:
            return float(fs.exceed.to_numpy()[m].mean() - THRESHOLD)

        thirds = np.array_split(np.sort(fs.block.unique()), 3)
        lg = fs.series.to_numpy()
        vc = pd.Series(lg).value_counts()
        extra = {
            "clean": True,
            "within_ceiling": True,
            "subperiod_estimates": [share_on(fs.block.isin(t).to_numpy()) for t in thirds],
            "leave_one_league_out": {s: share_on(lg != s) for s in vc[vc >= 200].index},
            "by_series_share_exceed": {s: float(fs.exceed[lg == s].mean()) for s in vc[vc >= 200].index},
            "describe": detail["buckets"][name],
            "staleness": stale,
        }
        cfg = {"hours_to_kickoff": name}
        record(
            REG_ID,
            cfg,
            pd.DataFrame({"decision_id": fs.decision_id, "block": days, "value": fs.exceed - THRESHOLD}),
            "share_exceeding_roundtrip_cost_minus_5pct",
            snap,
            extra,
            cf,
        )
        record(
            REG_ID,
            cfg,
            pd.DataFrame({"decision_id": fs.decision_id, "block": days, "value": fs.gap}),
            "mean_signed_gap_kalshi_minus_pm",
            snap,
            {"clean": True},
            cf,
        )
        record(
            REG_ID,
            cfg,
            pd.DataFrame({"decision_id": fs.decision_id, "block": days, "value": fs.gap.abs()}),
            "mean_abs_gap",
            snap,
            {"clean": True},
            cf,
        )
        if len(bs) > 1:
            record(
                REG_ID,
                cfg,
                pd.DataFrame({"decision_id": bs.decision_id, "block": bs.block, "value": bs.exceed - THRESHOLD}),
                "share_exceeding_roundtrip_cost_minus_5pct_book",
                snap,
                {"clean": True},
                cf,
            )
    out = R5 / REG_ID
    out.mkdir(parents=True, exist_ok=True)
    (out / f"detail_{window}.json").write_text(json.dumps(detail, indent=2, default=str))
    print(json.dumps(detail, indent=1, default=str))


if __name__ == "__main__":
    main(sys.argv[1])
