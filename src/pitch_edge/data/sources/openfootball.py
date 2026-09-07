"""openfootball connector (github.com/openfootball/football.json).

Public-domain fixtures and results as JSON, one file per league-season.
Useful as (a) an independent cross-check of the spine's results and (b) a
source of *upcoming* fixtures for the current season, which football-data.co.uk
only publishes once played. Team names carry "FC" suffixes and full names,
so they go through `TeamNameResolver` before joining.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from pitch_edge.config import get_settings
from pitch_edge.data.http import CachedHttpClient
from pitch_edge.data.sources.base import MatchDataSource

logger = logging.getLogger(__name__)

BASE_URL = "https://raw.githubusercontent.com/openfootball/football.json/master"

LEAGUES: dict[str, str] = {  # openfootball code -> display name
    "en.1": "England - Premier League",
    "en.2": "England - Championship",
    "de.1": "Germany - Bundesliga",
    "de.2": "Germany - 2. Bundesliga",
    "es.1": "Spain - La Liga",
    "es.2": "Spain - Segunda Division",
    "it.1": "Italy - Serie A",
    "it.2": "Italy - Serie B",
    "fr.1": "France - Ligue 1",
    "fr.2": "France - Ligue 2",
    "nl.1": "Netherlands - Eredivisie",
    "pt.1": "Portugal - Primeira Liga",
    "at.1": "Austria - Bundesliga",
    "ch.1": "Switzerland - Super League",
    "be.1": "Belgium - Jupiler Pro League",
    "tr.1": "Turkey - Super Lig",
    "gr.1": "Greece - Super League",
    "hu.1": "Hungary - NB I",
    "cz.1": "Czechia - First League",
    "ru.1": "Russia - Premier League",
    "sco.1": "Scotland - Premiership",
    "mx.1": "Mexico - Liga MX",
    "br.1": "Brazil - Serie A",
    "ar.1": "Argentina - Primera Division",
    "jp.1": "Japan - J1 League",
    "cn.1": "China - Super League",
}


def season_dir(start_year: int) -> str:
    return f"{start_year}-{(start_year + 1) % 100:02d}"


EUROPE_URL = "https://raw.githubusercontent.com/openfootball/europe/master"
_TXT_DATE_RE = re.compile(r"^\s*(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+([A-Z][a-z]{2})\s+(\d{1,2})(?:\s+(\d{4}))?\s*$")
_TXT_MATCH_RE = re.compile(
    r"^\s*(?:(\d{1,2}[.:]\d{2})\s+)?(.+?)\s+v\s+(.+?)(?:(?:\s{2,}|\s+)(\d+)-(\d+)(?:\s*\((\d+)-(\d+)\))?)?\s*$"
)
_TXT_HEADER_RE = re.compile(r"^=\s*(.+?)\s+(\d{4}(?:/\d{2})?)\s*$")


def parse_openfootball_txt(
    text: str, league_code: str, league_name: str, source: str = "openfootball", include_unplayed: bool = False
) -> pd.DataFrame:
    """Parse the openfootball plain-text schedule format (matchday blocks, date lines, 'A v B 2-1 (1-0)')."""
    rows = []
    year = None
    season = ""
    current_date = None
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        h = _TXT_HEADER_RE.match(line)
        if h:
            season = h.group(2)
            year = int(season[:4])
            continue
        d = _TXT_DATE_RE.match(line)
        if d:
            y = int(d.group(4)) if d.group(4) else year
            try:
                current_date = pd.Timestamp(f"{d.group(2)} {d.group(3)} {y}")
            except ValueError:
                current_date = None
            continue
        m = _TXT_MATCH_RE.match(line)
        if m and current_date is not None:
            kickoff, home, away = m.group(1), m.group(2).strip(), m.group(3).strip()
            played = m.group(4) is not None
            if not played and not include_unplayed:
                continue
            hg = float(m.group(4)) if played else None
            ag = float(m.group(5)) if played else None
            slug = lambda s: re.sub(r"[^A-Za-z0-9]+", "", str(s))  # noqa: E731
            rows.append(
                {
                    "match_id": f"of_{league_code}_{current_date.strftime('%Y%m%d')}_{slug(home)}_{slug(away)}",
                    "date": current_date,
                    "kickoff_time": kickoff.replace(".", ":") if kickoff else None,
                    "league": league_name,
                    "league_code": league_code,
                    "country": league_name.split(" - ")[0],
                    "season": season,
                    "home_team": home,
                    "away_team": away,
                    "home_goals": hg,
                    "away_goals": ag,
                    "ht_home_goals": float(m.group(6)) if m.group(6) else None,
                    "ht_away_goals": float(m.group(7)) if m.group(7) else None,
                    "source": source,
                }
            )
    return pd.DataFrame(rows)


def _extract_score(m: dict) -> tuple[int, int] | None:
    """openfootball has used several score shapes over the years:
    {"score": {"ft": [h, a]}}, {"score": [h, a]}, {"score1": h, "score2": a}."""
    score = m.get("score")
    if isinstance(score, dict):
        ft = score.get("ft")
        if isinstance(ft, list | tuple) and len(ft) == 2 and None not in ft:
            return int(ft[0]), int(ft[1])
        return None
    if isinstance(score, list | tuple) and len(score) == 2 and None not in score:
        return int(score[0]), int(score[1])
    if m.get("score1") is not None and m.get("score2") is not None:
        return int(m["score1"]), int(m["score2"])
    return None


class OpenFootballSource(MatchDataSource):
    name = "openfootball"

    def __init__(self, cache_dir: str | Path | None = None, http: CachedHttpClient | None = None):
        settings = get_settings()
        self.cache_dir = Path(cache_dir) if cache_dir else settings.raw_dir / "openfootball"
        self.http = http or CachedHttpClient(self.cache_dir, min_interval_s=0.5)

    def fetch_matches(  # type: ignore[override]
        self,
        league: str = "en.1",
        seasons: list[int] | None = None,
        include_unplayed: bool = False,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        seasons = seasons or [2023, 2024]
        frames = []
        for start_year in seasons:
            url = f"{BASE_URL}/{season_dir(start_year)}/{league}.json"
            try:
                payload = self.http.get_json(url, force=force_refresh)
            except Exception as exc:
                logger.warning("openfootball %s %s unavailable: %s", league, start_year, exc)
                continue
            frames.append(self._normalize(payload, league, start_year, include_unplayed))
        if not frames:
            return pd.DataFrame(columns=list(self.required_columns()))
        df = pd.concat(frames, ignore_index=True)
        if include_unplayed:
            return df  # cannot pass goal-null validation by design; caller handles fixtures
        return self.validate(df)

    def _normalize(self, payload: dict, league: str, start_year: int, include_unplayed: bool) -> pd.DataFrame:
        rows = []
        matches = payload.get("matches") or [m for r in payload.get("rounds", []) for m in r.get("matches", [])]
        for m in matches:
            score = _extract_score(m)
            if score is None and not include_unplayed:
                continue
            home = m.get("team1")
            away = m.get("team2")
            if isinstance(home, dict):
                home = home.get("name")
            if isinstance(away, dict):
                away = away.get("name")
            rows.append(
                {
                    "date": pd.to_datetime(m.get("date"), errors="coerce"),
                    "kickoff_time": m.get("time"),
                    "round": m.get("round"),
                    "league": LEAGUES.get(league, league),
                    "league_code": league,
                    "season": f"{start_year}/{(start_year + 1) % 100:02d}",
                    "home_team": home,
                    "away_team": away,
                    "home_goals": float(score[0]) if score else None,
                    "away_goals": float(score[1]) if score else None,
                    "source": self.name,
                }
            )
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df = df.dropna(subset=["date", "home_team", "away_team"])
        slug = lambda s: re.sub(r"[^A-Za-z0-9]+", "", str(s))  # noqa: E731
        df["match_id"] = (
            "of_" + league.replace(".", "") + "_" + df["date"].dt.strftime("%Y%m%d") + "_"
            + df["home_team"].map(slug) + "_" + df["away_team"].map(slug)
        )
        return df.drop_duplicates(subset="match_id").reset_index(drop=True)

    @staticmethod
    def extract_score(m: dict) -> tuple[int, int] | None:
        return _extract_score(m)

    # ------------------------------------------------------------ europe .txt
    def fetch_europe_txt(
        self, country: str, filename: str, league_code: str, league_name: str, include_unplayed: bool = False
    ) -> pd.DataFrame:
        """Parse an openfootball/europe text schedule, e.g. sweden/2025_se1.txt (Allsvenskan)."""
        url = f"{EUROPE_URL}/{country}/{filename}"
        text = self.http.get_text(url)
        df = parse_openfootball_txt(
            text, league_code=league_code, league_name=league_name, source=self.name, include_unplayed=include_unplayed
        )
        if df.empty or include_unplayed:
            return df
        return self.validate(df)

    def fetch_sweden(self, seasons: tuple[int, ...] = (2023, 2024, 2025), include_unplayed: bool = False) -> pd.DataFrame:
        frames = []
        tiers = ((1, "SWE1", "Sweden - Allsvenskan"), (2, "SWE2", "Sweden - Superettan"), (3, "SWE3", "Sweden - Ettan"))
        for year in seasons:
            for tier, code, name in tiers:
                for suffix in [f"{year}_se{tier}.txt"] if tier < 3 else [f"{year}_se3s.txt", f"{year}_se3n.txt"]:
                    try:
                        frames.append(self.fetch_europe_txt("sweden", suffix, code, name, include_unplayed=include_unplayed))
                    except Exception as exc:  # noqa: BLE001 - file may not exist for that season
                        logger.info("openfootball sweden %s: %s", suffix, exc)
        frames = [f for f in frames if not f.empty]
        return pd.concat(frames, ignore_index=True).drop_duplicates("match_id") if frames else pd.DataFrame()

    def fetch_upcoming_fixtures(self, league: str, start_year: int) -> pd.DataFrame:
        df = self.fetch_matches(league=league, seasons=[start_year], include_unplayed=True)
        if df.empty:
            return df
        return df[df["home_goals"].isna()].reset_index(drop=True)
