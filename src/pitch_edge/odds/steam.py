"""Cross-book / cross-venue divergence and steam detection.

Given odds snapshots over time for the same market (bookmakers, exchanges,
prediction markets), flag (a) *steam*: a sharp reference line moving while
soft books lag, and (b) *divergence*: two venues implying materially
different no-vig probabilities right now. Output is a list of flags for the
dashboard/alert log. Nothing here places a bet.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from pitch_edge.odds.utils import no_vig_probabilities


@dataclass(frozen=True)
class DivergenceFlag:
    match_id: str
    side: str
    sharp_book: str
    soft_book: str
    sharp_probability: float
    soft_probability: float
    gap: float  # sharp - soft (positive: soft book too generous on this side)
    soft_decimal_odds: float
    kind: str  # "divergence" | "steam"


def _no_vig_table(snapshot: pd.DataFrame) -> pd.DataFrame:
    """snapshot: match_id, bookmaker, side (home/draw/away), price. -> match_id, bookmaker, side, prob, price"""
    rows = []
    for (mid, book), g in snapshot.groupby(["match_id", "bookmaker"]):
        prices = g.set_index("side")["price"]
        if not {"home", "draw", "away"}.issubset(prices.index):
            continue
        h, d, a = float(prices["home"]), float(prices["draw"]), float(prices["away"])
        if min(h, d, a) <= 1.0:
            continue
        probs = no_vig_probabilities(h, d, a)
        for side, p, price in zip(("home", "draw", "away"), probs, (h, d, a), strict=True):
            rows.append({"match_id": mid, "bookmaker": book, "side": side, "prob": p, "price": price})
    return pd.DataFrame(rows)


def detect_divergence(snapshot: pd.DataFrame, sharp_book: str = "PS", min_gap: float = 0.03) -> list[DivergenceFlag]:
    table = _no_vig_table(snapshot)
    if table.empty:
        return []
    flags: list[DivergenceFlag] = []
    for mid, g in table.groupby("match_id"):
        sharp = g[g["bookmaker"] == sharp_book].set_index("side")
        if sharp.empty:
            continue
        for book, gb in g[g["bookmaker"] != sharp_book].groupby("bookmaker"):
            gb = gb.set_index("side")
            for side in ("home", "draw", "away"):
                if side not in gb.index or side not in sharp.index:
                    continue
                gap = float(sharp.loc[side, "prob"] - gb.loc[side, "prob"])
                if gap >= min_gap:
                    flags.append(
                        DivergenceFlag(
                            str(mid),
                            side,
                            sharp_book,
                            str(book),
                            float(sharp.loc[side, "prob"]),
                            float(gb.loc[side, "prob"]),
                            gap,
                            float(gb.loc[side, "price"]),
                            "divergence",
                        )
                    )
    return flags


def detect_steam(
    history: pd.DataFrame, sharp_book: str = "PS", move_threshold: float = 0.02, lag_threshold: float = 0.01
) -> list[DivergenceFlag]:
    """history: match_id, bookmaker, side, price, snapshot_ts (>= 2 snapshots per book).

    Steam = sharp book's no-vig probability for a side moved by >= move_threshold
    between its first and last snapshot while a soft book moved < lag_threshold.
    """
    if history.empty:
        return []
    history = history.sort_values("snapshot_ts")
    first = _no_vig_table(history.groupby(["match_id", "bookmaker", "side"], as_index=False).first())
    last = _no_vig_table(history.groupby(["match_id", "bookmaker", "side"], as_index=False).last())
    if first.empty or last.empty:
        return []
    merged = first.merge(last, on=["match_id", "bookmaker", "side"], suffixes=("_first", "_last"))
    merged["move"] = merged["prob_last"] - merged["prob_first"]
    flags: list[DivergenceFlag] = []
    for (mid, side), g in merged.groupby(["match_id", "side"]):
        sharp = g[g["bookmaker"] == sharp_book]
        if sharp.empty or abs(float(sharp["move"].iloc[0])) < move_threshold:
            continue
        sharp_move = float(sharp["move"].iloc[0])
        for _, soft in g[g["bookmaker"] != sharp_book].iterrows():
            if abs(float(soft["move"])) < lag_threshold and sharp_move > 0:
                flags.append(
                    DivergenceFlag(
                        str(mid),
                        str(side),
                        sharp_book,
                        str(soft["bookmaker"]),
                        float(sharp["prob_last"].iloc[0]),
                        float(soft["prob_last"]),
                        float(sharp["prob_last"].iloc[0] - soft["prob_last"]),
                        float(soft["price_last"]),
                        "steam",
                    )
                )
    return flags


def flags_to_frame(flags: list[DivergenceFlag]) -> pd.DataFrame:
    return pd.DataFrame([f.__dict__ for f in flags])
