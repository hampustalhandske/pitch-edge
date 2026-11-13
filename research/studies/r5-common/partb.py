"""Part B (cross-venue) panel: Kalshi and Polymarket prices for the same outcome and the same hour.

One row per (Kalshi event, side, H) for hours H on the hour with ko - 48h <= H <= ko (ko = verified kickoff),
holding what each venue showed at H:

- Kalshi: the candle with end_period_ts = H (known at H): yes bid/ask at the close of the hour and their mid.
  Kalshi writes a candle every hour, with quotes even when nothing traded (66% of pre-kickoff hours).
- Polymarket last trade: the bar with hour_start = H - 1h (known at H) -> `p_fresh` (a trade happened in
  the hour) with its high-low; `p_ffill` / `p_age_h` carry the last trade forward with its age (stale).
- Polymarket book (only 2026-04-09 -> 2026-06-10 and 2026-09-29): the snapshot in force at H.

Hindsight: Kalshi `result`, PM resolution and the Betfair close are not in the panel; they are joined only
where a study uses them as a settlement check or benchmark.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "r1-common"))

from mapping import verified_universe  # noqa: E402
from prices import OrderBooks, asset_labels, kalshi_hourly, pm_hourly  # noqa: E402

REPO = HERE.parents[2]
R5 = REPO / "data" / "research" / "r5"
SIDES = ("home", "draw", "away")
EXPLORATORY = ("2025-08-01", "2025-12-31")
EVALUATION = ("2026-01-01", "2026-10-19")
MAX_H = 48
CODE_FILES = [
    HERE / "partb.py",
    HERE / "prices.py",
    HERE / "mapping.py",
    HERE / "harness.py",
    HERE / "pull_prices.py",
    HERE / "pull_meta.py",
]


def snapshot_tag() -> str:
    m = json.loads((R5 / "snapshot_meta.json").read_text())
    p = json.loads((R5 / "snapshot_prices.json").read_text())
    return (
        f"r5 meta {m['pulled_at'][:19]} kalshi_markets={m['kalshi_markets']} pm_1x2={m['pm_1x2']}; prices "
        f"{p['pulled_at'][:19]} candles={p['kalshi_candles']} bars={p['pm_bars']} book={p['pm_book']}"
    )


def universe() -> tuple[pd.DataFrame, dict]:
    u, drops = verified_universe()
    u = u[u.match_day <= EVALUATION[1]].copy()
    u["window"] = np.where(
        u.match_day < EXPLORATORY[0],
        "before",
        np.where(u.match_day <= EXPLORATORY[1], "exploratory", "evaluation"),
    )
    drops["kept_to_2026_10_19"] = len(u)
    return u.reset_index(drop=True), drops


def panel(u: pd.DataFrame, with_book: bool = True) -> pd.DataFrame:
    """The (event, side, H) panel for the events in `u`."""
    kc = kalshi_hourly(pd.read_parquet(R5 / "kalshi_candles.parquet"))
    bars = pd.read_parquet(R5 / "pm_bars.parquet")
    ph = pm_hourly(bars)
    long = []
    for s in SIDES:
        long.append(
            u[["event_ticker", "series", "ko", "match_day", "window", "fd_match_key", f"tk_{s}", f"cond_{s}"]]
            .rename(columns={f"tk_{s}": "ticker", f"cond_{s}": "condition_id"})
            .assign(side=s)
        )
    lu = pd.concat(long, ignore_index=True)
    k = kc.merge(lu, on="ticker", how="inner")
    k = k[(k.H <= k.ko) & (k.H >= k.ko - pd.Timedelta(hours=MAX_H))].copy()
    k["hours_to_ko"] = (k.ko - k.H).dt.total_seconds() / 3600
    # Polymarket bar in the hour ending at H (fresh trade) ...
    k = k.merge(ph.rename(columns={"p": "p_fresh"}), on=["condition_id", "H"], how="left")
    # ... and the last trade known at H, carried forward, with its age.
    ph_sorted = ph.sort_values("H")
    k = k.sort_values("H")
    ff = pd.merge_asof(
        k[["condition_id", "H"]].reset_index(),
        ph_sorted[["condition_id", "H", "p"]].rename(columns={"H": "H_last", "p": "p_ffill"}),
        left_on="H",
        right_on="H_last",
        by="condition_id",
        direction="backward",
    ).set_index("index")
    k = k.join(ff[["H_last", "p_ffill"]])
    k["p_age_h"] = (k.H - k.H_last).dt.total_seconds() / 3600
    assert (k.H_last.isna() | (k.H_last <= k.H)).all(), "PM bar used before it was known"
    if with_book:
        labels = asset_labels(bars, pd.read_parquet(REPO / "data" / "research" / "r1" / "bars.parquet"))
        books = OrderBooks(pd.read_parquet(R5 / "pm_book.parquet"), labels)
        cols = {c: [] for c in ("b_bid", "b_ask", "b_mid", "b_age_h")}
        for r in k[["condition_id", "H"]].itertuples(index=False):
            q = books.at(r.condition_id, r.H)
            if np.isfinite(q["known_at_ns"]):
                assert q["known_at_ns"] <= r.H.value
            ok = books.usable(q, 24.0, max_spread=1.0)
            cols["b_bid"].append(q["bid"] if ok else np.nan)
            cols["b_ask"].append(q["ask"] if ok else np.nan)
            cols["b_mid"].append(q["mid"] if ok else np.nan)
            cols["b_age_h"].append(q["age_h"] if ok else np.nan)
        for c, v in cols.items():
            k[c] = v
    return k.sort_values(["event_ticker", "side", "H"]).reset_index(drop=True)
