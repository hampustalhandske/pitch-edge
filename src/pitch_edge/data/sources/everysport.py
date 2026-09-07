"""Everysport connector — OPTIONAL, keyed (free registration at everysport.com).

Set `EVERYSPORT_API_KEY` to enable. Everysport is the Swedish sports-results
aggregator with the deepest coverage of Swedish football below Allsvenskan
(Superettan, Ettan, Division 2/3). Disabled → every call returns an empty frame.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd

from pitch_edge.config import get_settings
from pitch_edge.data.http import CachedHttpClient
from pitch_edge.data.sources.base import MatchDataSource

logger = logging.getLogger(__name__)

BASE_URL = "https://api.everysport.com/v1"


class EverysportSource(MatchDataSource):
    name = "everysport"

    def __init__(self, api_key: str | None = None, cache_dir: str | Path | None = None, http: CachedHttpClient | None = None):
        settings = get_settings()
        self.api_key = api_key or os.environ.get("EVERYSPORT_API_KEY")
        self.cache_dir = Path(cache_dir) if cache_dir else settings.raw_dir / "everysport"
        self.http = http or CachedHttpClient(self.cache_dir, min_interval_s=1.0)

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def _get(self, path: str, params: dict | None = None) -> dict:
        if not self.enabled:
            return {}
        try:
            payload = self.http.get_json(f"{BASE_URL}/{path}", params={**(params or {}), "apikey": self.api_key})
            return payload if isinstance(payload, dict) else {}
        except Exception as exc:  # noqa: BLE001
            logger.info("Everysport %s unavailable: %s", path, exc)
            return {}

    def leagues(self, sport_id: int = 1, country: str = "Sweden") -> pd.DataFrame:
        rows = [
            {"league_id": lg.get("id"), "league": lg.get("name"), "sport": (lg.get("sport") or {}).get("name")}
            for lg in self._get("leagues", {"sport": sport_id}).get("leagues", [])
            if country.lower() in str(lg.get("name", "") + lg.get("country", "")).lower() or True
        ]
        return pd.DataFrame(rows)

    def fetch_matches(self, league_id: int | None = None, status: str = "finished", **_) -> pd.DataFrame:  # type: ignore[override]
        if league_id is None:
            return pd.DataFrame(columns=list(self.required_columns()))
        rows = []
        for ev in self._get("events", {"league": league_id, "status": status, "limit": 500}).get("events", []):
            home, away = ev.get("homeTeam") or {}, ev.get("visitingTeam") or {}
            rows.append(
                {
                    "match_id": f"es_{ev.get('id')}",
                    "date": pd.to_datetime(ev.get("startDate"), errors="coerce", utc=True).tz_convert(None)
                    if ev.get("startDate")
                    else pd.NaT,
                    "league": f"Sweden - {(ev.get('league') or {}).get('name', league_id)}",
                    "league_code": f"ES{league_id}",
                    "country": "Sweden",
                    "season": str(pd.to_datetime(ev.get("startDate"), errors="coerce").year) if ev.get("startDate") else "",
                    "home_team": home.get("name"),
                    "away_team": away.get("name"),
                    "home_goals": pd.to_numeric(ev.get("homeTeamScore"), errors="coerce"),
                    "away_goals": pd.to_numeric(ev.get("visitingTeamScore"), errors="coerce"),
                    "source": self.name,
                }
            )
        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame(columns=list(self.required_columns()))
        df = df.dropna(subset=["date", "home_goals", "away_goals"]).reset_index(drop=True)
        return self.validate(df) if not df.empty else df
