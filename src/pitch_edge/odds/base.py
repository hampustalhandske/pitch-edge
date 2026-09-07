"""Common interface for odds providers.

Two flavours exist:
- historical/closing-line providers (used for backtesting) — real data
- live providers (used for the paper-trading alert pipeline) — most
  consumer-grade live odds APIs (The Odds API, SportsGameOdds, etc.) require
  a paid plan for meaningful coverage/refresh rate. Rather than depend on a
  paid key the project can't guarantee a reader has, `MockLiveOddsProvider`
  simulates a live feed from historical closing lines with synthetic
  pre-close drift, clearly labeled as synthetic everywhere it's surfaced
  (dashboard, docs). A real provider (e.g. TheOddsApiProvider) can be dropped
  in later behind this exact interface with zero downstream changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class OddsQuote:
    match_id: str
    bookmaker: str
    home_odds: float
    draw_odds: float
    away_odds: float
    timestamp: str  # ISO-8601
    is_closing: bool = False


class OddsProvider(ABC):
    name: str = "unnamed_provider"

    @abstractmethod
    def get_quotes(self, match_id: str) -> list[OddsQuote]:
        """All bookmaker quotes currently known for a match."""
        raise NotImplementedError
