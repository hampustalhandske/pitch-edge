"""StatsBomb Open Data connector (github.com/statsbomb/open-data).

Free professional-grade *event* data, published for research/education
(attribution required — see StatsBomb's open-data licence). We read the raw
JSON straight from GitHub through the cached HTTP client rather than via
`statsbombpy`, so every file is fetched exactly once, works offline after
that, and is trivially mockable in tests. Event frames feed the GNN passing
network / xT layer; match frames feed the same MatchDataSource interface as
the rest of the sources.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from pitch_edge.config import get_settings
from pitch_edge.data.http import CachedHttpClient
from pitch_edge.data.sources.base import MatchDataSource

logger = logging.getLogger(__name__)

BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"


class StatsBombOpenDataSource(MatchDataSource):
    name = "statsbomb_open_data"

    def __init__(self, cache_dir: str | Path | None = None, http: CachedHttpClient | None = None):
        settings = get_settings()
        self.cache_dir = Path(cache_dir) if cache_dir else settings.raw_dir / "statsbomb"
        self.http = http or CachedHttpClient(self.cache_dir, min_interval_s=0.3)

    def list_competitions(self) -> pd.DataFrame:
        return pd.DataFrame(self.http.get_json(f"{BASE_URL}/competitions.json"))

    def fetch_matches(  # type: ignore[override]
        self, competition_id: int | None = None, season_id: int | None = None, **_
    ) -> pd.DataFrame:
        if competition_id is None or season_id is None:
            raise ValueError("competition_id and season_id are required (see list_competitions())")
        payload = self.http.get_json(f"{BASE_URL}/matches/{competition_id}/{season_id}.json")
        rows = []
        for m in payload:
            if m.get("home_score") is None or m.get("away_score") is None:
                continue
            rows.append(
                {
                    "match_id": f"sb_{m['match_id']}",
                    "statsbomb_match_id": int(m["match_id"]),
                    "date": pd.to_datetime(m.get("match_date"), errors="coerce"),
                    "kickoff_time": m.get("kick_off"),
                    "league": f"{m['competition']['country_name']} - {m['competition']['competition_name']}",
                    "competition_id": competition_id,
                    "season_id": season_id,
                    "season": m["season"]["season_name"],
                    "home_team": m["home_team"]["home_team_name"],
                    "away_team": m["away_team"]["away_team_name"],
                    "home_goals": float(m["home_score"]),
                    "away_goals": float(m["away_score"]),
                    "referee": (m.get("referee") or {}).get("name"),
                    "stadium": (m.get("stadium") or {}).get("name"),
                    "source": self.name,
                }
            )
        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame(columns=list(self.required_columns()))
        return self.validate(df)

    def fetch_events(self, statsbomb_match_id: int) -> pd.DataFrame:
        """Flat event frame for one match (type, team, player, minute, location, pass end, xG...)."""
        payload = self.http.get_json(f"{BASE_URL}/events/{statsbomb_match_id}.json")
        rows = []
        for e in payload:
            loc = e.get("location") or [None, None]
            pas = e.get("pass") or {}
            shot = e.get("shot") or {}
            end = pas.get("end_location") or [None, None]
            recipient = pas.get("recipient") or {}
            rows.append(
                {
                    "statsbomb_match_id": statsbomb_match_id,
                    "event_id": e.get("id"),
                    "index": e.get("index"),
                    "period": e.get("period"),
                    "minute": e.get("minute"),
                    "second": e.get("second"),
                    "type": (e.get("type") or {}).get("name"),
                    "possession": e.get("possession"),
                    "team": (e.get("team") or {}).get("name"),
                    "player": (e.get("player") or {}).get("name"),
                    "player_id": (e.get("player") or {}).get("id"),
                    "position": (e.get("position") or {}).get("name"),
                    "x": loc[0], "y": loc[1],
                    "pass_end_x": end[0], "pass_end_y": end[1],
                    "pass_recipient": recipient.get("name"),
                    "pass_recipient_id": recipient.get("id"),
                    "pass_outcome": (pas.get("outcome") or {}).get("name"),
                    "pass_length": pas.get("length"),
                    "shot_xg": shot.get("statsbomb_xg"),
                    "shot_outcome": (shot.get("outcome") or {}).get("name"),
                    "under_pressure": bool(e.get("under_pressure", False)),
                }
            )
        return pd.DataFrame(rows)

