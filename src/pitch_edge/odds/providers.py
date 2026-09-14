"""Concrete OddsProvider implementations.

`HistoricalClosingOddsProvider` is real: it reads the bookmaker odds columns
that ship inside football-data.co.uk match rows (see
`pitch_edge.data.sources.football_data_co_uk`). This is genuine closing-line
data, free, no key required, and is what the backtest engine uses for CLV.

`MockLiveOddsProvider` is explicitly synthetic — a stand-in for a paid live
odds API (The Odds API / SportsGameOdds / a funded Betfair account) that
this project doesn't assume the reader has. It is never used in the
backtest and is labeled "synthetic/demo" everywhere the dashboard surfaces
it, per the project's honesty-about-what's-real requirement.

`PolymarketOddsProvider` is real: it reads actual executed Polymarket trade
prices from `pmxt_orderbook` (joined to `matches` via `pmxt_match_map`) and
returns one `OddsQuote` per tick, chronologically — unlike the other two
providers, `get_quotes()` here returns genuine intraday price movement, not
a single snapshot.
"""

from __future__ import annotations

import json
import random

import pandas as pd

from pitch_edge.data.storage import Warehouse
from pitch_edge.odds.base import OddsProvider, OddsQuote

# Trade prices land exactly at 0 or 1 near resolution; clip before inverting to a decimal odds
# so a downstream 1/price never divides by zero or produces a nonsensical odds value.
_MIN_PRICE = 0.001
_MAX_PRICE = 0.999

# Bookmaker column prefixes -> display name, matching football_data_co_uk.py
_BOOKMAKER_NAMES = {
    "B365": "Bet365",
    "BW": "Bet&Win",
    "PS": "Pinnacle",
    "WH": "William Hill",
    "VC": "VC Bet",
    "Max": "Market Best",
    "Avg": "Market Average",
}


class HistoricalClosingOddsProvider(OddsProvider):
    """Reads closing odds embedded in a football-data.co.uk-derived DataFrame."""

    name = "football_data_co_uk_closing"

    def __init__(self, matches_with_odds: pd.DataFrame):
        self._df = matches_with_odds.set_index("match_id", drop=False)

    def get_quotes(self, match_id: str) -> list[OddsQuote]:
        if match_id not in self._df.index:
            return []
        row = self._df.loc[match_id]
        timestamp = str(row.get("date", ""))
        quotes = []
        for prefix, bookmaker in _BOOKMAKER_NAMES.items():
            h_col, d_col, a_col = f"{prefix}H", f"{prefix}D", f"{prefix}A"
            if h_col in row.index and pd.notna(row[h_col]) and pd.notna(row[d_col]) and pd.notna(row[a_col]):
                quotes.append(
                    OddsQuote(
                        match_id=match_id,
                        bookmaker=bookmaker,
                        home_odds=float(row[h_col]),
                        draw_odds=float(row[d_col]),
                        away_odds=float(row[a_col]),
                        timestamp=timestamp,
                        is_closing=True,
                    )
                )
        return quotes


class PolymarketOddsProvider(OddsProvider):
    """Real per-tick Polymarket prices for a match, converted to pseudo-decimal-odds (`1/price`)
    at this one boundary so every downstream consumer (`backtest/kelly.py`, `backtest/metrics.py`,
    `odds/utils.py::no_vig_probabilities`) works unmodified — they're all decimal-odds-shaped.

    Requires `pmxt_match_map` (see `data/alt/pmxt_match_map.py`) to already have resolved the
    match's 3 outcome markets. `clob_token_ids[0]` is treated as the "Yes" token per Gamma API
    convention (each market's own list orders Yes before No)."""

    name = "polymarket"

    def __init__(self, wh: Warehouse):
        self._wh = wh

    def get_quotes(self, match_id: str) -> list[OddsQuote]:
        markets = self._wh.query(
            "SELECT condition_id, outcome_side FROM pmxt_match_map WHERE match_id = ?", [match_id]
        )
        if markets.empty:
            return []
        side_by_condition = dict(zip(markets["condition_id"], markets["outcome_side"], strict=True))
        placeholders = ",".join("?" * len(side_by_condition))
        clob = self._wh.query(
            f"SELECT condition_id, clob_token_ids FROM dim_soccer_markets WHERE condition_id IN ({placeholders})",
            list(side_by_condition),
        )
        yes_token: dict[str, str] = {}
        for row in clob.itertuples(index=False):
            ids = json.loads(row.clob_token_ids) if row.clob_token_ids else []
            if ids:
                yes_token[row.condition_id] = ids[0]
        asset_to_side = {yes_token[cid]: side for cid, side in side_by_condition.items() if cid in yes_token}
        if not asset_to_side:
            return []

        cond_placeholders = ",".join("?" * len(side_by_condition))
        asset_placeholders = ",".join("?" * len(asset_to_side))
        trades = self._wh.query(
            f"""
            SELECT asset_id, timestamp_received, price FROM pmxt_orderbook
            WHERE condition_id IN ({cond_placeholders}) AND asset_id IN ({asset_placeholders})
            ORDER BY timestamp_received
            """,
            list(side_by_condition) + list(asset_to_side),
        )
        if trades.empty:
            return []

        trades["side"] = trades["asset_id"].map(asset_to_side)
        trades["timestamp_received"] = pd.to_datetime(trades["timestamp_received"], utc=True, errors="coerce")
        trades = trades.dropna(subset=["timestamp_received"])
        wide = trades.pivot_table(index="timestamp_received", columns="side", values="price", aggfunc="last")
        wide = wide.reindex(columns=["home", "draw", "away"]).ffill().dropna(how="any")
        if wide.empty:
            return []
        wide = wide.clip(lower=_MIN_PRICE, upper=_MAX_PRICE)

        quotes = []
        n = len(wide)
        for i, (ts, row) in enumerate(wide.iterrows()):
            quotes.append(
                OddsQuote(
                    match_id=match_id,
                    bookmaker=self.name,
                    home_odds=1.0 / row["home"],
                    draw_odds=1.0 / row["draw"],
                    away_odds=1.0 / row["away"],
                    timestamp=ts.isoformat(),
                    is_closing=(i == n - 1),
                )
            )
        return quotes


class MockLiveOddsProvider(OddsProvider):
    """SYNTHETIC. Simulates a live pre-match odds feed by jittering a closing
    line backwards in time. Stands in for a paid live-odds API; do not treat
    output as real market data. Deterministic given a seed for testability.
    """

    name = "mock_live_synthetic"

    def __init__(self, closing_odds_provider: HistoricalClosingOddsProvider, seed: int = 42):
        self._closing = closing_odds_provider
        self._rng = random.Random(seed)

    def get_quotes(self, match_id: str) -> list[OddsQuote]:
        closing = self._closing.get_quotes(match_id)
        jittered = []
        for q in closing:
            drift = 1.0 + self._rng.uniform(-0.05, 0.05)
            jittered.append(
                OddsQuote(
                    match_id=q.match_id,
                    bookmaker=f"{q.bookmaker} (synthetic-live)",
                    home_odds=round(max(q.home_odds * drift, 1.01), 2),
                    draw_odds=round(max(q.draw_odds * drift, 1.01), 2),
                    away_odds=round(max(q.away_odds * drift, 1.01), 2),
                    timestamp=q.timestamp,
                    is_closing=False,
                )
            )
        return jittered
