"""TheSportsDB connector (free tier, public key "3"; no registration).

Coverage we use: leagues by country (Sweden: Allsvenskan 4347, Division 1 North/
South), season fixtures/results with venue + round, and team squads with player
bios (position, DOB, nationality). Rate-limited; every response is cached.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from pitch_edge.config import get_settings
from pitch_edge.data.http import CachedHttpClient

logger = logging.getLogger(__name__)

BASE_URL = "https://www.thesportsdb.com/api/v1/json/3"
SWEDISH_LEAGUES = {4347: "Allsvenskan", 4674: "Division 1 North", 4845: "Division 1 South", 4756: "Svenska Cupen"}


class TheSportsDBSource:
    name = "thesportsdb"

    def __init__(self, cache_dir: str | Path | None = None, http: CachedHttpClient | None = None):
        settings = get_settings()
        self.cache_dir = Path(cache_dir) if cache_dir else settings.raw_dir / "thesportsdb"
        self.http = http or CachedHttpClient(self.cache_dir, min_interval_s=2.0)

    def _json(self, url: str, params: dict) -> dict:
        """The free tier answers some endpoints with an HTML rate-limit page — treat that as empty."""
        try:
            payload = self.http.get_json(url, params=params)
            return payload if isinstance(payload, dict) else {}
        except Exception as exc:  # noqa: BLE001
            logger.info("TheSportsDB %s unavailable: %s", url.rsplit("/", 1)[-1], exc)
            return {}


    def season_events(self, league_id: int, season: str) -> pd.DataFrame:
        payload = self._json(f"{BASE_URL}/eventsseason.php", params={"id": league_id, "s": season})
        rows = []
        for e in payload.get("events") or []:
            rows.append(
                {
                    "event_id": f"tsdb_{e.get('idEvent')}",
                    "league_id": league_id,
                    "season": season,
                    "round": e.get("intRound"),
                    "date": pd.to_datetime(e.get("dateEvent"), errors="coerce"),
                    "kickoff_time": (e.get("strTime") or "")[:5] or None,
                    "home_team": e.get("strHomeTeam"),
                    "away_team": e.get("strAwayTeam"),
                    "home_goals": pd.to_numeric(e.get("intHomeScore"), errors="coerce"),
                    "away_goals": pd.to_numeric(e.get("intAwayScore"), errors="coerce"),
                    "venue": e.get("strVenue"),
                    "status": e.get("strStatus"),
                    "source": self.name,
                }
            )
        return pd.DataFrame(rows)

    def team_players(self, team_name: str) -> pd.DataFrame:
        payload = self._json(f"{BASE_URL}/searchplayers.php", params={"t": team_name})
        rows = []
        for p in payload.get("player") or []:
            rows.append(
                {
                    "player_id": f"tsdb_{p.get('idPlayer')}",
                    "player": p.get("strPlayer"),
                    "team": p.get("strTeam"),
                    "position": p.get("strPosition"),
                    "nationality": p.get("strNationality"),
                    "date_of_birth": pd.to_datetime(p.get("dateBorn"), errors="coerce"),
                    "height": p.get("strHeight"),
                    "status": p.get("strStatus"),
                    "source": self.name,
                }
            )
        return pd.DataFrame(rows)

    def team_lookup(self, team_name: str) -> pd.DataFrame:
        payload = self._json(f"{BASE_URL}/searchteams.php", params={"t": team_name})
        rows = []
        for t in payload.get("teams") or []:
            rows.append(
                {
                    "team_id": f"tsdb_{t.get('idTeam')}",
                    "team": t.get("strTeam"),
                    "league": t.get("strLeague"),
                    "stadium": t.get("strStadium"),
                    "stadium_capacity": pd.to_numeric(t.get("intStadiumCapacity"), errors="coerce"),
                    "location": t.get("strLocation") or t.get("strStadiumLocation"),
                    "founded": pd.to_numeric(t.get("intFormedYear"), errors="coerce"),
                    "source": self.name,
                }
            )
        return pd.DataFrame(rows)
