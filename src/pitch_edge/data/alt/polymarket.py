"""Polymarket Gamma API connector — prediction-market prices as a cross-venue odds source.

Free, public, keyless read-only API. Prediction markets quote outcome
probabilities directly (0-1 share prices), which we convert to decimal-odds
equivalents so they slot into the same `OddsQuote` shape as bookmaker
prices. This is the "cross-book / cross-exchange divergence" leg of the
brief: a prediction market that has already moved while a soft book hasn't
is flagged (never traded automatically).
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import pandas as pd

from pitch_edge.config import get_settings
from pitch_edge.data.http import CachedHttpClient

logger = logging.getLogger(__name__)

GAMMA_URL = "https://gamma-api.polymarket.com"
# Gamma's tag filter is not reliable (it has returned US-politics markets under "soccer"); keep only
# questions that look like football fixtures or competitions.
_FOOTBALL_Q = re.compile(
    r"\b(vs\.?|v\.?)\b|premier league|la liga|serie a|bundesliga|ligue 1|champions league|europa league|"
    r"allsvenskan|eredivisie|primeira|s[uü]per lig|liga mx|brasileir|copa|world cup|euro 20|mls\b|"
    r"\bfc\b|\bcf\b|football|soccer|goals?\b|clean sheet|relegat|to win the",
    re.I,
)


_NON_FOOTBALL_Q = re.compile(
    r"election|nominat|president|senate|governor|congress|parliament|award|oscar|grammy|emmy|\bnba\b|\bnfl\b|\bmlb\b|"
    r"\bnhl\b|\bufc\b|tennis|golf|\bf1\b|formula|nascar|cricket|rugby|bitcoin|\beth\b|fed\b|rate cut|tariff",
    re.I,
)
_GENERIC_WIN_Q = re.compile(r"\bwin(s|ner)?\b", re.I)


def is_football_question(question: str | None) -> bool:
    """Football-looking question (fixture or competition words), or a generic 'will X win?' that is not
    obviously politics/finance/another sport. Gamma's tag filter alone has returned US-politics markets."""
    if not question:
        return False
    q = str(question)
    if _NON_FOOTBALL_Q.search(q):
        return False
    return bool(_FOOTBALL_Q.search(q) or _GENERIC_WIN_Q.search(q))


class PolymarketSource:
    name = "polymarket"

    def __init__(self, cache_dir: str | Path | None = None, http: CachedHttpClient | None = None):
        settings = get_settings()
        self.cache_dir = Path(cache_dir) if cache_dir else settings.raw_dir / "polymarket"
        self.http = http or CachedHttpClient(self.cache_dir, min_interval_s=1.0)

    def fetch_football_markets(self, limit: int = 200, closed: bool = False, football_only: bool = True) -> pd.DataFrame:
        frames = []
        for tag in ("soccer", "football", "premier-league", "champions-league", "la-liga", "bundesliga", "serie-a"):
            try:
                payload = self.http.get_json(
                    f"{GAMMA_URL}/markets",
                    params={"tag_slug": tag, "closed": str(closed).lower(), "limit": limit},
                    cache=False,
                )
            except Exception as exc:
                logger.info("Polymarket tag %s unavailable: %s", tag, exc)
                continue
            df = self.parse_markets(payload)
            if football_only and not df.empty:
                df = df[df["question"].map(is_football_question)]
            if not df.empty:
                frames.append(df)
        if not frames:
            return pd.DataFrame(columns=["market_id", "question", "outcome", "probability", "decimal_odds", "end_date"])
        return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["market_id", "outcome"])

    @staticmethod
    def parse_markets(payload: list[dict]) -> pd.DataFrame:
        rows = []
        for m in payload or []:
            outcomes = _as_list(m.get("outcomes"))
            prices = _as_list(m.get("outcomePrices"))
            if not outcomes or len(outcomes) != len(prices):
                continue
            for outcome, price in zip(outcomes, prices, strict=True):
                try:
                    p = float(price)
                except (TypeError, ValueError):
                    continue
                if not (0.0 < p < 1.0):
                    continue
                rows.append(
                    {
                        "market_id": str(m.get("id")),
                        "question": m.get("question"),
                        "slug": m.get("slug"),
                        "outcome": outcome,
                        "probability": p,
                        "decimal_odds": round(1.0 / p, 3),
                        "volume": float(m.get("volumeNum") or m.get("volume") or 0.0),
                        "liquidity": float(m.get("liquidityNum") or m.get("liquidity") or 0.0),
                        "end_date": pd.to_datetime(m.get("endDate"), errors="coerce", utc=True),
                        "snapshot_ts": pd.Timestamp.utcnow().tz_localize(None),
                    }
                )
        return pd.DataFrame(rows)


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else []
    except (TypeError, ValueError):
        return []
