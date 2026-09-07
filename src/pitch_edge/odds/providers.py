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
"""

from __future__ import annotations

import random

import pandas as pd

from pitch_edge.odds.base import OddsProvider, OddsQuote

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
