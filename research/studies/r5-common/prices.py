"""Point-in-time price access for R5: Polymarket order books, Polymarket hourly bars, Kalshi hourly candles.

`OrderBooks(book, labels)` holds, per Polymarket condition, the time-ordered top of book expressed for the
Yes outcome. Both tokens of a condition trade on one complementary book (verified: Yes best bid + No best
ask = 1 to the cent), so a No-token snapshot is converted (yes_bid = 1 - no_ask, yes_ask = 1 - no_bid, depths
swapped). A snapshot is known at its `ts`. The archive stores roughly one snapshot per token-hour, timed at
the last book update in that hour, so the latest snapshot at or before t is the book in force at t and its
age says how long the book has been unchanged (or unobserved). `at(cond, t)` returns that book.

`asset_labels(*bar_frames)` maps asset_id -> 'yes'/'no' from the hourly bars' `outcome_label`; an unlabelled
asset of a condition whose other asset is labelled gets the complement label.

`kalshi_panel(candles, tickers)` and `pm_hourly(bars)` give hourly prices keyed by the hour H at which they
are known: a Kalshi candle with end_period_ts = H, a Polymarket bar with hour_start = H - 1h.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MAX_SPREAD = 0.05


def asset_labels(*bar_frames: pd.DataFrame) -> dict[str, str]:
    lab: dict[str, str] = {}
    cond_assets: dict[str, set[str]] = {}
    for bars in bar_frames:
        b = bars[["asset_id", "condition_id", "outcome_label"]].drop_duplicates()
        for r in b.itertuples(index=False):
            cond_assets.setdefault(r.condition_id, set()).add(r.asset_id)
            ol = str(r.outcome_label).lower()
            if ol in ("yes", "no"):
                lab[r.asset_id] = ol
    return lab


class OrderBooks:
    """Yes-side top of book per condition, as known at a timestamp."""

    def __init__(self, book: pd.DataFrame, labels: dict[str, str]):
        b = book.copy()
        # Complete labels inside each condition: two assets, one labelled -> the other is its complement.
        for _cid, g in b.groupby("condition_id"):
            assets = list(g.asset_id.unique())
            known = [a for a in assets if a in labels]
            if len(assets) == 2 and len(known) == 1:
                other = assets[1 - assets.index(known[0])]
                labels.setdefault(other, "no" if labels[known[0]] == "yes" else "yes")
        b["label"] = b.asset_id.map(labels)
        self.n_unlabelled_rows = int(b.label.isna().sum())
        b = b[b.label.notna()].copy()
        yes = b.label.eq("yes")
        b["bid"] = np.where(yes, b.best_bid, 1 - b.best_ask)
        b["ask"] = np.where(yes, b.best_ask, 1 - b.best_bid)
        b["bid_d1"] = np.where(yes, b.bid_depth_1c, b.ask_depth_1c)
        b["ask_d1"] = np.where(yes, b.ask_depth_1c, b.bid_depth_1c)
        b["is_yes"] = yes
        b = b.sort_values(["condition_id", "ts", "is_yes"]).drop_duplicates(["condition_id", "ts"], keep="last")
        self.series: dict[str, tuple[np.ndarray, ...]] = {}
        for cid, g in b.groupby("condition_id", sort=False):
            self.series[cid] = (
                g.ts.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy().astype("datetime64[ns]").astype(np.int64),
                g.bid.to_numpy(float),
                g.ask.to_numpy(float),
                g.bid_d1.to_numpy(float),
                g.ask_d1.to_numpy(float),
            )

    def at(self, cid: str, ts: pd.Timestamp) -> dict[str, float]:
        """Book in force at `ts`: bid, ask, mid, spread, microprice (depth-1c weighted; NaN when a side has
        no depth), age_h (hours since the snapshot), known_at_ns. All NaN when no snapshot precedes ts."""
        nan = dict.fromkeys(("bid", "ask", "mid", "spread", "micro", "age_h", "known_at_ns"), np.nan)
        s = self.series.get(cid)
        if s is None:
            return nan
        t = pd.Timestamp(ts).value
        i = np.searchsorted(s[0], t, side="right") - 1
        if i < 0:
            return nan
        assert s[0][i] <= t, "book used before it was known"
        bid, ask, bd, ad = s[1][i], s[2][i], s[3][i], s[4][i]
        micro = (bid * ad + ask * bd) / (ad + bd) if (ad > 0 and bd > 0) else np.nan
        return {
            "bid": bid,
            "ask": ask,
            "mid": (bid + ask) / 2,
            "spread": ask - bid,
            "micro": micro,
            "age_h": (t - s[0][i]) / 3.6e12,
            "known_at_ns": float(s[0][i]),
        }

    def usable(self, q: dict[str, float], max_age_h: float, max_spread: float = MAX_SPREAD) -> bool:
        """A two-sided book no older than max_age_h with spread <= max_spread."""
        return bool(
            np.isfinite(q["mid"])
            and q["bid"] > 0
            and q["ask"] < 1
            and q["spread"] <= max_spread + 1e-9
            and q["age_h"] <= max_age_h
        )


def pm_hourly(bars: pd.DataFrame) -> pd.DataFrame:
    """Yes-equivalent hourly Polymarket bar per (condition_id, H) with H = hour_start + 1h (when it is known):
    p (last trade), hi, lo, volume. Yes bars win over No-complement bars in the same hour."""
    b = bars[bars.outcome_label.str.lower().isin(["yes", "no"])].copy()
    yes = b.outcome_label.str.lower().eq("yes")
    b["p"] = np.where(yes, b.close, 1 - b.close)
    b["hi"] = np.where(yes, b.high, 1 - b.low)
    b["lo"] = np.where(yes, b.low, 1 - b.high)
    b["is_yes"] = yes
    b["H"] = b.hour_start + pd.Timedelta(hours=1)
    b = b.sort_values(["condition_id", "H", "is_yes"]).drop_duplicates(["condition_id", "H"], keep="last")
    return b[["condition_id", "H", "p", "hi", "lo", "volume_usdc", "n_trades"]].reset_index(drop=True)


def kalshi_hourly(candles: pd.DataFrame) -> pd.DataFrame:
    """Kalshi quotes per (ticker, H) with H = end_period_ts: bid, ask, mid (NaN unless 0 < bid < ask <= 1),
    last trade price_close, volume."""
    c = candles.copy()
    c["H"] = c.end_period_ts
    bid, ask = c.yes_bid_close, c.yes_ask_close
    ok = (bid > 0) & (ask > bid) & (ask <= 1)
    c["bid"] = bid.where(ok)
    c["ask"] = ask.where(ok)
    c["mid"] = ((bid + ask) / 2).where(ok)
    return c[["ticker", "H", "bid", "ask", "mid", "price_close", "volume", "open_interest"]].reset_index(drop=True)


def pm_fee(p: np.ndarray | float) -> np.ndarray | float:
    """Polymarket taker fee per share bought at price p (research/STANDARDS.md): 0.05 p (1 - p)."""
    return 0.05 * np.asarray(p) * (1 - np.asarray(p))


def kalshi_fee(p: np.ndarray | float) -> np.ndarray | float:
    """Kalshi taker fee per contract bought at price p: 0.07 p (1 - p) (series fee_type `quadratic` /
    `quadratic_with_maker_fees`, fee_multiplier 1, per Kalshi's public series API; per-order rounding up to
    the cent is ignored, i.e. the large-order limit)."""
    return 0.07 * np.asarray(p) * (1 - np.asarray(p))
