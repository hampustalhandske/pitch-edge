"""Kalshi connector — a second prediction-market venue (public, keyless, read-only).

Kalshi's trade API exposes series (e.g. KXEPLGAME, KXLALIGASPREAD, KXSERIEABTTS,
KXLIGAMXSCORE) and their markets with yes/no bid-ask in cents. We snapshot the
football series so cross-venue divergence (Kalshi vs Polymarket vs bookmaker
no-vig) can be flagged. Nothing is traded.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from pitch_edge.config import get_settings
from pitch_edge.data.http import CachedHttpClient

logger = logging.getLogger(__name__)

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
FOOTBALL_TOKENS = (
    "EPL", "LALIGA", "SERIEA", "BUNDESLIGA", "LIGUE1", "LIGUE2", "MLS", "LIGAMX", "UCL", "UEFA", "FIFA",
    "LIGAPORTUGAL", "EREDIVISIE", "SCOTTISHPREM", "EFLCHAMPIONSHIP", "SOCCER", "ALLSVENSKAN", "ARGPREM", "BRASIL",
)
EXCLUDE_TOKENS = ("NFL", "NCAAF", "RUGBY", "NBA", "FANTASY", "PROBOWL")


def _num(value) -> float:
    try:
        return float(value) if value is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _dollars(m: dict, key: str) -> float | None:
    """Kalshi moved from integer cents (`yes_bid`) to dollar strings (`yes_bid_dollars`); accept both."""
    if m.get(f"{key}_dollars") not in (None, ""):
        try:
            return float(m[f"{key}_dollars"])
        except (TypeError, ValueError):
            return None
    if m.get(key) is not None:
        try:
            return float(m[key]) / 100.0
        except (TypeError, ValueError):
            return None
    return None


def is_football_series(ticker: str) -> bool:
    t = ticker.upper()
    return any(k in t for k in FOOTBALL_TOKENS) and not any(k in t for k in EXCLUDE_TOKENS)


class KalshiSource:
    name = "kalshi"

    def __init__(self, cache_dir: str | Path | None = None, http: CachedHttpClient | None = None):
        settings = get_settings()
        self.cache_dir = Path(cache_dir) if cache_dir else settings.raw_dir / "kalshi"
        self.http = http or CachedHttpClient(self.cache_dir, min_interval_s=0.5)

    def list_football_series(self) -> list[dict]:
        try:
            payload = self.http.get_json(f"{BASE_URL}/series", params={"category": "Sports"}, cache=False)
        except Exception as exc:  # noqa: BLE001
            logger.info("Kalshi series unavailable: %s", exc)
            return []
        return [s for s in payload.get("series", []) if is_football_series(str(s.get("ticker", "")))]

    def fetch_markets(self, series_ticker: str, status: str = "open", limit: int = 200) -> pd.DataFrame:
        try:
            payload = self.http.get_json(
                f"{BASE_URL}/markets", params={"series_ticker": series_ticker, "status": status, "limit": limit}, cache=False
            )
        except Exception as exc:  # noqa: BLE001
            logger.info("Kalshi markets %s unavailable: %s", series_ticker, exc)
            return pd.DataFrame()
        return self.parse_markets(payload.get("markets", []), series_ticker)

    @staticmethod
    def parse_markets(markets: list[dict], series_ticker: str = "") -> pd.DataFrame:
        rows = []
        now = pd.Timestamp.utcnow().tz_localize(None)
        for m in markets or []:
            bid, ask, last = _dollars(m, "yes_bid"), _dollars(m, "yes_ask"), _dollars(m, "last_price")
            mid = None
            if bid is not None and ask is not None and 0.0 < bid <= ask < 1.0:
                mid = (bid + ask) / 2.0
            elif last is not None and 0.0 < last < 1.0:
                mid = last
            if mid is None:
                continue
            rows.append(
                {
                    "market_id": f"kalshi:{m.get('ticker')}",
                    "venue": "kalshi",
                    "series": series_ticker or str(m.get("ticker", "")).split("-")[0],
                    "event_ticker": m.get("event_ticker"),
                    "question": f"{m.get('title')} — {m.get('rules_primary', '')[:120]}" if m.get("rules_primary") else m.get("title"),
                    "outcome": m.get("yes_sub_title") or "yes",
                    "probability": mid,
                    "decimal_odds": round(1.0 / mid, 3),
                    "yes_bid": bid,
                    "yes_ask": ask,
                    "spread": (ask - bid) if (bid is not None and ask is not None) else None,
                    "volume": _num(m.get("volume_fp", m.get("volume"))),
                    "liquidity": _num(m.get("open_interest_fp", m.get("open_interest"))),
                    "end_date": pd.to_datetime(m.get("close_time"), errors="coerce", utc=True),
                    "snapshot_ts": now,
                }
            )
        return pd.DataFrame(rows)

    def fetch_football_markets(self, max_series: int = 60, status: str = "open") -> pd.DataFrame:
        series = self.list_football_series()
        # prefer match-level series (GAME / SPREAD / TOTAL / BTTS / SCORE) over season futures
        prio = sorted(series, key=lambda s: (0 if any(k in s["ticker"] for k in ("GAME", "SPREAD", "TOTAL", "BTTS", "SCORE", "MOV")) else 1, s["ticker"]))
        frames = [self.fetch_markets(s["ticker"], status=status) for s in prio[:max_series]]
        frames = [f for f in frames if not f.empty]
        if not frames:
            return pd.DataFrame(columns=["market_id", "venue", "series", "question", "outcome", "probability", "decimal_odds", "snapshot_ts"])
        return pd.concat(frames, ignore_index=True)
