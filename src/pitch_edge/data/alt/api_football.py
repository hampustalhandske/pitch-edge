"""API-Football connector — OPTIONAL, keyed (free tier: 100 requests/day).

Set `API_FOOTBALL_KEY` to enable. Gives injuries, lineups and substitutions per
fixture — the "who is actually playing" layer that no keyless source provides.
Requests are cached on disk, so the daily budget is spent only on new fixtures.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd

from pitch_edge.config import get_settings
from pitch_edge.data.http import CachedHttpClient

logger = logging.getLogger(__name__)

BASE_URL = "https://v3.football.api-sports.io"
LEAGUE_IDS = {"E0": 39, "D1": 78, "SP1": 140, "I1": 135, "F1": 61, "SWE1": 113, "SWE2": 114, "N1": 88, "P1": 94}


class APIFootballSource:
    name = "api_football"

    def __init__(self, api_key: str | None = None, cache_dir: str | Path | None = None, http: CachedHttpClient | None = None):
        settings = get_settings()
        self.api_key = api_key or os.environ.get("API_FOOTBALL_KEY")
        self.cache_dir = Path(cache_dir) if cache_dir else settings.raw_dir / "api_football"
        self.http = http or CachedHttpClient(self.cache_dir, min_interval_s=1.0)

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def _get(self, path: str, params: dict) -> list[dict]:
        if not self.enabled:
            return []
        try:
            payload = self.http.get_json(f"{BASE_URL}/{path}", params=params, headers={"x-apisports-key": self.api_key})
        except Exception as exc:  # noqa: BLE001
            logger.info("API-Football %s unavailable: %s", path, exc)
            return []
        return payload.get("response", []) if isinstance(payload, dict) else []

    def injuries(self, league_code: str, season: int) -> pd.DataFrame:
        rows = []
        for r in self._get("injuries", {"league": LEAGUE_IDS.get(league_code, league_code), "season": season}):
            rows.append(
                {
                    "fixture_id": (r.get("fixture") or {}).get("id"),
                    "date": pd.to_datetime((r.get("fixture") or {}).get("date"), errors="coerce", utc=True),
                    "team": (r.get("team") or {}).get("name"),
                    "player": (r.get("player") or {}).get("name"),
                    "type": (r.get("player") or {}).get("type"),
                    "reason": (r.get("player") or {}).get("reason"),
                    "league_code": league_code,
                    "source": self.name,
                }
            )
        return pd.DataFrame(rows)

    def lineups(self, fixture_id: int) -> pd.DataFrame:
        rows = []
        for team in self._get("fixtures/lineups", {"fixture": fixture_id}):
            for slot, group in (("start", team.get("startXI", [])), ("bench", team.get("substitutes", []))):
                for p in group:
                    pl = p.get("player", {})
                    rows.append(
                        {
                            "fixture_id": fixture_id,
                            "team": (team.get("team") or {}).get("name"),
                            "formation": team.get("formation"),
                            "player": pl.get("name"),
                            "position": pl.get("pos"),
                            "slot": slot,
                            "source": self.name,
                        }
                    )
        return pd.DataFrame(rows)

    def substitutions(self, fixture_id: int) -> pd.DataFrame:
        rows = []
        for e in self._get("fixtures/events", {"fixture": fixture_id, "type": "subst"}):
            rows.append(
                {
                    "fixture_id": fixture_id,
                    "minute": (e.get("time") or {}).get("elapsed"),
                    "team": (e.get("team") or {}).get("name"),
                    "player_out": (e.get("player") or {}).get("name"),
                    "player_in": (e.get("assist") or {}).get("name"),
                    "source": self.name,
                }
            )
        return pd.DataFrame(rows)
