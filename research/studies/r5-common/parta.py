"""Part A builders (polymarket-efficiency microstructure follow-up to R1): per-decision frames priced at the
Polymarket order-book mid, with R1's last-trade price on the same rows for paired comparison.

Inputs: R1's snapshot (data/research/r1/: matches, bars) for the universe, kickoff rule, Betfair/sharp-open
de-vig and last-trade PriceBook; R5's order-book pull (data/research/r5/pm_book.parquet) for the mids.
Hindsight columns: the football-data result (`result_side`) and the Betfair close (`q_bf_close_*`) are used
only as targets/benchmarks; every Polymarket price is asserted to be known at or before the decision time.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "r1-common"))
sys.path.insert(0, str(HERE))

from common import DATA as R1_DATA  # noqa: E402
from common import SIDES, PriceBook, load_matches, pm_probs, power_normalise  # noqa: E402
from prices import OrderBooks, asset_labels  # noqa: E402

REPO = HERE.parents[2]
R5 = REPO / "data" / "research" / "r5"
EVAL = (pd.Timestamp("2026-04-09", tz="UTC"), pd.Timestamp("2026-09-30", tz="UTC"))
CODE_FILES = [
    HERE / "parta.py",
    HERE / "prices.py",
    HERE / "harness.py",
    HERE / "pull_prices.py",
    HERE.parent / "r1-common" / "common.py",
]


def snapshot_tag() -> str:
    r1 = json.loads((R1_DATA / "snapshot.json").read_text())
    r5 = json.loads((R5 / "snapshot_prices.json").read_text())
    return f"r1 {r1['pulled_at'][:19]} bars={r1['bars']}; r5 book {r5['pulled_at'][:19]} rows={r5['pm_book']}"


def load() -> tuple[pd.DataFrame, PriceBook, OrderBooks, dict]:
    """R1 matches with kickoff in the book window, R1's last-trade book and the order books."""
    df, drops = load_matches()
    df = df[(df.ko >= EVAL[0]) & (df.ko < EVAL[1])].reset_index(drop=True)
    r1_bars = pd.read_parquet(R1_DATA / "bars.parquet")
    r5_bars = pd.read_parquet(R5 / "pm_bars.parquet")
    labels = asset_labels(r1_bars, r5_bars)
    books = OrderBooks(pd.read_parquet(R5 / "pm_book.parquet"), labels)
    info = {**drops, "matches_in_book_window": len(df), "book_rows_unlabelled": books.n_unlabelled_rows}
    return df, PriceBook(r1_bars), books, info


def mid_probs(
    df: pd.DataFrame, books: OrderBooks, h: float, max_age: float, field: str = "mid"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Power-normalised 3-way book prices (`field` = mid or micro) at kickoff - h, the pass mask (all three
    books usable, raw sum in [0.9, 1.1]) and the raw prices."""
    raw = np.full((len(df), 3), np.nan)
    ok = np.ones(len(df), bool)
    for i, r in enumerate(df.itertuples(index=False)):
        ts = r.ko - pd.Timedelta(hours=h)
        for j, s in enumerate(SIDES):
            q = books.at(getattr(r, f"cond_{s}"), ts)
            if np.isfinite(q["known_at_ns"]):
                assert q["known_at_ns"] <= ts.value
            if not books.usable(q, max_age):
                ok[i] = False
                continue
            raw[i, j] = q[field]
    s = raw.sum(axis=1)
    ok &= np.isfinite(raw).all(axis=1) & (s >= 0.9) & (s <= 1.1)
    pn = power_normalise(np.where(ok[:, None], raw, np.nan))
    ok &= np.isfinite(pn).all(axis=1)
    return pn, ok, raw


def lasttrade_probs(df: pd.DataFrame, pbook: PriceBook, h: float) -> tuple[np.ndarray, np.ndarray]:
    """R1's last-trade rule: last trades <= 24h old, raw sum in [0.9, 1.1], power-normalised."""
    return pm_probs(df, pbook, h, 24.0, 0.9, 1.1, "power")


def subperiods(values: np.ndarray, days: np.ndarray, stat) -> list[float | None]:  # type: ignore[no-untyped-def]
    """`stat` per chronological third of match-days."""
    d = pd.Series(days).reset_index(drop=True)
    cuts = np.array_split(np.sort(d.unique()), 3)
    out: list[float | None] = []
    for c in cuts:
        m = d.isin(c).to_numpy()
        out.append(float(stat(m)) if m.sum() > 1 else None)
    return out


def leave_one_league_out(leagues: np.ndarray, stat, min_rows: int = 20) -> dict[str, float | None]:  # type: ignore[no-untyped-def]
    """`stat` with each league (>= min_rows rows) removed in turn."""
    lg = pd.Series(leagues).reset_index(drop=True)
    vc = lg.value_counts()
    out: dict[str, float | None] = {}
    for name in vc[vc >= min_rows].index:
        m = (lg != name).to_numpy()
        out[str(name)] = float(stat(m)) if m.sum() > 1 else None
    return out


def by_league(leagues: np.ndarray, stat, min_rows: int = 20) -> dict[str, dict]:  # type: ignore[no-untyped-def]
    lg = pd.Series(leagues).reset_index(drop=True)
    vc = lg.value_counts()
    lab = lg.where(lg.map(vc) >= min_rows, "pooled_small")
    return {str(n): {"n": int((lab == n).sum()), "est": float(stat((lab == n).to_numpy()))} for n in lab.unique()}
