"""Club-Football-Match-Data-2000-2025 connector (github.com/xgabora).

An open, MIT-style community compilation of ~230k club matches (2000-2025)
across 30+ divisions, built mostly on football-data.co.uk with pre-computed
Elo ratings (per match, both teams), rolling form, and match stats. It's
a superset of our spine for historical Elo-at-kickoff without having to
replay Club Elo's history ourselves, and it extends coverage to divisions
football-data.co.uk only publishes as recent seasons.

Single ~43 MB CSV, cached once. Team names follow football-data.co.uk, so
they join onto the spine without aliasing.
"""

from __future__ import annotations

import logging
from io import StringIO
from pathlib import Path

import pandas as pd

from pitch_edge.config import get_settings
from pitch_edge.data.http import CachedHttpClient
from pitch_edge.data.sources.base import MatchDataSource
from pitch_edge.data.sources.football_data_co_uk import EXTRA_DIVISION_COUNTRIES, LEAGUE_CODES

_DIVISIONS: dict[str, tuple[str, str]] = {**LEAGUE_CODES, **EXTRA_DIVISION_COUNTRIES}

logger = logging.getLogger(__name__)

MATCHES_URL = "https://raw.githubusercontent.com/xgabora/Club-Football-Match-Data-2000-2025/main/data/Matches.csv"
ELO_URL = "https://raw.githubusercontent.com/xgabora/Club-Football-Match-Data-2000-2025/main/data/EloRatings.csv"

COLUMN_MAP = {
    "HomeElo": "home_elo", "AwayElo": "away_elo",
    "Form3Home": "home_form3", "Form5Home": "home_form5", "Form3Away": "away_form3", "Form5Away": "away_form5",
    "HTHome": "ht_home_goals", "HTAway": "ht_away_goals",
    "HomeShots": "home_shots", "AwayShots": "away_shots", "HomeTarget": "home_shots_on_target",
    "AwayTarget": "away_shots_on_target", "HomeFouls": "home_fouls", "AwayFouls": "away_fouls",
    "HomeCorners": "home_corners", "AwayCorners": "away_corners", "HomeYellow": "home_yellows",
    "AwayYellow": "away_yellows", "HomeRed": "home_reds", "AwayRed": "away_reds",
    "OddHome": "MktH", "OddDraw": "MktD", "OddAway": "MktA",
    "MaxHome": "MaxH", "MaxDraw": "MaxD", "MaxAway": "MaxA",
    "Over25": "Mkt>2.5", "Under25": "Mkt<2.5", "MaxOver25": "Max>2.5", "MaxUnder25": "Max<2.5",
    "HandiSize": "ah_line", "HandiHome": "ah_home_odds", "HandiAway": "ah_away_odds",
}


class ClubFootballMatchDataSource(MatchDataSource):
    name = "club_football_match_data"

    def __init__(self, cache_dir: str | Path | None = None, http: CachedHttpClient | None = None):
        settings = get_settings()
        self.cache_dir = Path(cache_dir) if cache_dir else settings.raw_dir / "club_football_match_data"
        self.http = http or CachedHttpClient(self.cache_dir, min_interval_s=settings.http_min_interval_s)

    def fetch_matches(  # type: ignore[override]
        self,
        divisions: list[str] | None = None,
        min_year: int = 2000,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        text = self.http.get_text(MATCHES_URL, force=force_refresh)
        raw = pd.read_csv(StringIO(text), low_memory=False)
        if divisions:
            raw = raw[raw["Division"].isin(divisions)]
        out = pd.DataFrame(
            {
                "date": pd.to_datetime(raw["MatchDate"], errors="coerce"),
                "league_code": raw["Division"].astype(str),
                "country": raw["Division"].astype(str).map(lambda c: _DIVISIONS.get(c, (None,))[0]),
                "league": raw["Division"].astype(str).map(
                    lambda c: f"{_DIVISIONS[c][0]} - {_DIVISIONS[c][1]}" if c in _DIVISIONS else c
                ),
                "home_team": raw["HomeTeam"].astype(str).str.strip(),
                "away_team": raw["AwayTeam"].astype(str).str.strip(),
                "home_goals": pd.to_numeric(raw["FTHome"], errors="coerce"),
                "away_goals": pd.to_numeric(raw["FTAway"], errors="coerce"),
                "source": self.name,
            }
        )
        for src, dst in COLUMN_MAP.items():
            if src in raw.columns:
                out[dst] = pd.to_numeric(raw[src], errors="coerce")
        out = out.dropna(subset=["date", "home_goals", "away_goals"])
        out = out[out["date"].dt.year >= min_year]
        # season = start year of the Jul-Jun football year
        start_year = out["date"].dt.year - (out["date"].dt.month < 7).astype(int)
        out["season"] = start_year.astype(str) + "/" + ((start_year + 1) % 100).map(lambda y: f"{y:02d}")
        out["match_id"] = (
            "cfmd_" + out["league_code"] + "_" + out["date"].dt.strftime("%Y%m%d") + "_"
            + out["home_team"].str.replace(r"[^A-Za-z0-9]+", "", regex=True) + "_"
            + out["away_team"].str.replace(r"[^A-Za-z0-9]+", "", regex=True)
        )
        out = out.drop_duplicates(subset="match_id").reset_index(drop=True)
        return self.validate(out)

