"""The Odds API connector — OPTIONAL, keyed. Real (non-synthetic) live bookmaker quotes.

Set `ODDS_API_KEY` to enable. Free tier is rate-limited; requests are NOT disk-cached
(`cache=False`) since a live line changes between snapshots, unlike every other source in
this project — but they still go through `CachedHttpClient`'s throttle/circuit-breaker.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd

from pitch_edge.config import get_settings
from pitch_edge.data.http import CachedHttpClient

logger = logging.getLogger(__name__)

BASE_URL = "https://api.the-odds-api.com/v4"


class OddsApiSource:
    name = "odds_api"

    def __init__(
        self, api_key: str | None = None, cache_dir: str | Path | None = None, http: CachedHttpClient | None = None
    ):
        settings = get_settings()
        self.api_key = api_key or os.environ.get("ODDS_API_KEY")
        self.cache_dir = Path(cache_dir) if cache_dir else settings.raw_dir / "odds_api"
        self.http = http or CachedHttpClient(self.cache_dir, min_interval_s=1.0)

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def _get(self, path: str, params: dict) -> list[dict]:
        if not self.enabled:
            return []
        query = {**params, "apiKey": self.api_key}
        try:
            payload = self.http.get_json(f"{BASE_URL}/{path}", params=query, cache=False)
        except Exception as exc:  # noqa: BLE001
            logger.info("Odds API %s unavailable: %s", path, exc)
            return []
        return payload if isinstance(payload, list) else []

    def live_odds(self, sport_key: str = "soccer_epl", regions: str = "uk,eu", markets: str = "h2h") -> pd.DataFrame:
        rows = []
        snapshot_ts = pd.Timestamp.utcnow()
        for event in self._get(
            f"sports/{sport_key}/odds", {"regions": regions, "markets": markets, "oddsFormat": "decimal"}
        ):
            match_id = event.get("id")
            home_team = event.get("home_team")
            away_team = event.get("away_team")
            for bookmaker in event.get("bookmakers", []):
                book_key = bookmaker.get("key")
                for market in bookmaker.get("markets", []):
                    market_key = market.get("key")
                    for outcome in market.get("outcomes", []):
                        rows.append(
                            {
                                "match_id": match_id,
                                "bookmaker": book_key,
                                "market": market_key,
                                "side": outcome.get("name"),
                                "price": outcome.get("price"),
                                "snapshot_ts": snapshot_ts,
                                "source": self.name,
                                "home_team": home_team,
                                "away_team": away_team,
                            }
                        )
        return pd.DataFrame(
            rows,
            columns=[
                "match_id",
                "bookmaker",
                "market",
                "side",
                "price",
                "snapshot_ts",
                "source",
                "home_team",
                "away_team",
            ],
        )
