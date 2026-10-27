"""Stage-2 pull for R5: prices for the mapped universes, into data/research/r5/.

Writes:
  kalshi_candles.parquet  kalshi_candles_1h rows for every ticker of a verified Kalshi<->PM event, with
                          end_period_ts in [ko - 72h, ko + 1h] (a candle is known at its end_period_ts)
  pm_bars.parquet         pm_soccer_bars_1h rows for the PM conditions of those events, hour_start in
                          [ko - 73h, ko] (a bar is known at hour_start + 1h)
  pm_book.parquet         pm_orderbook_snapshots rows (top of book and depth summaries; the per-level array
                          columns come back as JSON text and only within a narrow condition_id and ts filter,
                          an unfiltered read runs out of memory lake-side)
                          for (a) the R1 matched universe and (b) the PM conditions of verified Kalshi events,
                          restricted to matches kicking off inside the book's coverage, ts in [ko - 72h, ko]
  snapshot_prices.json    pull time and row counts

Usage: arch -arm64 uv run python research/studies/r5-common/pull_prices.py
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "r1-common"))
sys.path.insert(0, str(HERE))

from common import load_matches  # noqa: E402  (r1-common)
from lake import REPO, sql  # noqa: E402
from mapping import verified_universe  # noqa: E402

OUT = REPO / "data" / "research" / "r5"
CHUNK = 250
BOOK_FROM, BOOK_TO = pd.Timestamp("2026-04-09", tz="UTC"), pd.Timestamp("2026-09-30", tz="UTC")
SIDES = ("home", "draw", "away")


def _windowed(
    table: str, key: str, ts_col: str, cols: str, pairs: pd.DataFrame, lo_h: float, hi_h: float
) -> pd.DataFrame:
    """Rows of `table` for (key, ko) pairs with ts_col in [ko + lo_h, ko + hi_h].

    Keys are sorted by kickoff and queried in chunks with a plain IN-list and one time range per chunk
    (join plans against a VALUES list exceed the lake's temp-space limit); the per-key window is then
    applied locally.
    """
    parts = []
    pairs = pairs.drop_duplicates(key).sort_values("ko").reset_index(drop=True)
    n_chunks = (len(pairs) - 1) // CHUNK + 1
    for i in range(0, len(pairs), CHUNK):
        ch = pairs.iloc[i : i + CHUNK]
        lo = (ch.ko.min() + pd.Timedelta(hours=lo_h)).isoformat()
        hi = (ch.ko.max() + pd.Timedelta(hours=hi_h)).isoformat()
        marks = ",".join("?" * len(ch))
        q = f"""select {cols} from {table} t
                where t.{key} in ({marks}) and t.{ts_col} >= ?::TIMESTAMPTZ and t.{ts_col} <= ?::TIMESTAMPTZ"""
        got = sql(q, [*ch[key].tolist(), lo, hi])
        got = got.merge(ch[[key, "ko"]], on=key, how="inner")
        t = got[ts_col]
        got = got[(t >= got.ko + pd.Timedelta(hours=lo_h)) & (t <= got.ko + pd.Timedelta(hours=hi_h))]
        parts.append(got)
        print(f"{table} chunk {i // CHUNK + 1}/{n_chunks}: {len(got)} rows", flush=True)
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    snap: dict = {"pulled_at": datetime.now(UTC).isoformat()}
    u, drops = verified_universe()
    u = u[u.ko <= pd.Timestamp("2026-10-20", tz="UTC")]
    snap["universe_events"] = len(u)
    tk = pd.concat([u[["ko"]].assign(ticker=u[f"tk_{s}"]) for s in SIDES], ignore_index=True)
    kc = _windowed(
        "kalshi_candles_1h",
        "ticker",
        "end_period_ts",
        "t.ticker, t.end_period_ts, t.price_open, t.price_high, t.price_low, t.price_close, t.price_mean, "
        "t.yes_bid_open, t.yes_bid_close, t.yes_ask_open, t.yes_ask_close, t.volume, t.open_interest",
        tk,
        -72,
        1,
    )
    kc.to_parquet(OUT / "kalshi_candles.parquet")
    snap["kalshi_candles"] = len(kc)
    cd = pd.concat([u[["ko"]].assign(condition_id=u[f"cond_{s}"]) for s in SIDES], ignore_index=True)
    bars = _windowed(
        "pm_soccer_bars_1h",
        "condition_id",
        "hour_start",
        "t.asset_id, t.hour_start, t.condition_id, t.outcome_label, t.open, t.high, t.low, t.close, t.volume_usdc, "
        "t.n_trades",
        cd,
        -73,
        0,
    )
    bars.to_parquet(OUT / "pm_bars.parquet")
    snap["pm_bars"] = len(bars)
    # Book: R1 universe + Kalshi-matched conditions, kickoff inside the book coverage.
    r1, _ = load_matches()
    r1c = pd.concat([r1[["ko"]].assign(condition_id=r1[f"cond_{s}"]) for s in SIDES], ignore_index=True)
    allc = pd.concat([r1c, cd], ignore_index=True)
    allc = allc[(allc.ko >= BOOK_FROM) & (allc.ko <= BOOK_TO)]
    book = _windowed(
        "pm_orderbook_snapshots",
        "condition_id",
        "ts",
        "t.asset_id, t.condition_id, t.ts, t.best_bid, t.best_ask, t.mid, t.spread, t.n_bid_levels, t.n_ask_levels, "
        "t.bid_depth_1c, t.ask_depth_1c, t.bid_depth_5c, t.ask_depth_5c, t.source",
        allc,
        -72,
        0,
    )
    book.to_parquet(OUT / "pm_book.parquet")
    snap["pm_book"] = len(book)
    snap["pm_book_conditions_requested"] = int(allc.condition_id.nunique())
    (OUT / "snapshot_prices.json").write_text(json.dumps(snap, indent=2))
    print(snap)


if __name__ == "__main__":
    main()
