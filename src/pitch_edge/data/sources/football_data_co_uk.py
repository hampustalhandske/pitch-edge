"""football-data.co.uk connector — the backtesting spine.

Free, keyless, static CSV per league/season, explicitly published for
reuse. Two file families:

* **Main leagues** (`/mmz4281/<season>/<code>.csv`): 22 divisions across
  England (E0-E3, EC), Scotland (SC0-SC3), Germany (D1, D2), Italy (I1, I2),
  Spain (SP1, SP2), France (F1, F2), Netherlands (N1), Belgium (B1),
  Portugal (P1), Turkey (T1), Greece (G1). Results + match stats (shots,
  fouls, corners, cards, referee for E0) + odds from many bookmakers, with
  both *early* (e.g. `PSH`) and *closing* (e.g. `PSCH`) Pinnacle prices in
  recent seasons — a genuine bet-vs-close pair for honest CLV.
* **"Extra" leagues** (`/new/<COUNTRY>.csv`): one file per country covering
  all seasons since 2012 for Argentina, Austria, Brazil, China, Denmark,
  Finland, Ireland, Japan, Mexico, Norway, Poland, Romania, Russia, Sweden,
  Switzerland, USA. These are the under-covered, information-asymmetric
  markets the project brief says the edge actually lives in — and they ship
  with Pinnacle closing, market max/avg, and Betfair Exchange closing prices.
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

BASE_URL = "https://www.football-data.co.uk"

LEAGUE_CODES: dict[str, tuple[str, str]] = {  # code -> (country, league name)
    "E0": ("England", "Premier League"),
    "E1": ("England", "Championship"),
    "E2": ("England", "League One"),
    "E3": ("England", "League Two"),
    "EC": ("England", "National League"),
    "SC0": ("Scotland", "Premiership"),
    "SC1": ("Scotland", "Championship"),
    "SC2": ("Scotland", "League One"),
    "SC3": ("Scotland", "League Two"),
    "D1": ("Germany", "Bundesliga"),
    "D2": ("Germany", "2. Bundesliga"),
    "SP1": ("Spain", "La Liga"),
    "SP2": ("Spain", "Segunda Division"),
    "I1": ("Italy", "Serie A"),
    "I2": ("Italy", "Serie B"),
    "F1": ("France", "Ligue 1"),
    "F2": ("France", "Ligue 2"),
    "N1": ("Netherlands", "Eredivisie"),
    "B1": ("Belgium", "Jupiler Pro League"),
    "P1": ("Portugal", "Primeira Liga"),
    "T1": ("Turkey", "Super Lig"),
    "G1": ("Greece", "Super League"),
}

# Division codes used by the xgabora compilation for the same "extra" leagues (a few differ from the file stems).
EXTRA_DIVISION_COUNTRIES: dict[str, tuple[str, str]] = {
    "ARG": ("Argentina", "Liga Profesional"), "AUT": ("Austria", "Bundesliga"), "BRA": ("Brazil", "Serie A"),
    "CHN": ("China", "Super League"), "DEN": ("Denmark", "Superliga"), "DNK": ("Denmark", "Superliga"),
    "FIN": ("Finland", "Veikkausliiga"), "IRL": ("Ireland", "Premier Division"), "JAP": ("Japan", "J1 League"),
    "JPN": ("Japan", "J1 League"), "MEX": ("Mexico", "Liga MX"), "NOR": ("Norway", "Eliteserien"),
    "POL": ("Poland", "Ekstraklasa"), "ROM": ("Romania", "Liga I"), "ROU": ("Romania", "Liga I"),
    "RUS": ("Russia", "Premier League"), "SWE": ("Sweden", "Allsvenskan"), "SUI": ("Switzerland", "Super League"),
    "SWZ": ("Switzerland", "Super League"), "USA": ("USA", "MLS"),
}

EXTRA_LEAGUE_FILES: dict[str, str] = {  # file stem -> country
    "ARG": "Argentina", "AUT": "Austria", "BRA": "Brazil", "CHN": "China", "DNK": "Denmark",
    "FIN": "Finland", "IRL": "Ireland", "JPN": "Japan", "MEX": "Mexico", "NOR": "Norway",
    "POL": "Poland", "ROU": "Romania", "RUS": "Russia", "SWE": "Sweden", "SWZ": "Switzerland", "USA": "USA",
}

# 1X2 odds columns: <bookmaker><C?><H|D|A>. "C" marks a closing price.
_ODDS_1X2_RE = re.compile(r"^(B365|BW|IW|PS|WH|VC|Max|Avg|Mkt|BFE|BF|BS|GB|LB|SB|SJ|SY|SO|1XB|P)(C?)(H|D|A)$")
# Totals (over/under 2.5) columns: <bookmaker><C?><>|<>2.5
_ODDS_OU_RE = re.compile(r"^(B365|P|Max|Avg|Mkt|BFE|PC|MaxC|AvgC|B365C|BFEC|GB|BW|IW|LB|SB|SJ|WH|VC)([<>])2\.5$")

STAT_COLUMNS = {
    "HTHG": "ht_home_goals", "HTAG": "ht_away_goals",
    "HS": "home_shots", "AS": "away_shots", "HST": "home_shots_on_target", "AST": "away_shots_on_target",
    "HF": "home_fouls", "AF": "away_fouls", "HC": "home_corners", "AC": "away_corners",
    "HY": "home_yellows", "AY": "away_yellows", "HR": "home_reds", "AR": "away_reds",
    "Referee": "referee", "Attendance": "attendance", "Time": "kickoff_time",
}


def season_code(start_year: int) -> str:
    """2023 -> '2324'."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def season_label(start_year: int) -> str:
    return f"{start_year}/{(start_year + 1) % 100:02d}"


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", str(text))


class FootballDataCoUkSource(MatchDataSource):
    name = "football_data_co_uk"

    def __init__(self, cache_dir: str | Path | None = None, http: CachedHttpClient | None = None):
        settings = get_settings()
        self.cache_dir = Path(cache_dir) if cache_dir else settings.raw_dir / "football_data_co_uk"
        self.http = http or CachedHttpClient(self.cache_dir, min_interval_s=settings.http_min_interval_s)

    # ------------------------------------------------------------- main files
    def fetch_matches(  # type: ignore[override]
        self,
        league: str = "E0",
        seasons: list[int] | None = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        if league not in LEAGUE_CODES:
            raise ValueError(f"Unknown football-data.co.uk league code {league!r}")
        seasons = seasons or [2022, 2023, 2024]
        frames = []
        for start_year in seasons:
            url = f"{BASE_URL}/mmz4281/{season_code(start_year)}/{league}.csv"
            try:
                raw_bytes = self.http.get_bytes(url, force=force_refresh)
            except Exception as exc:  # 404 for seasons/leagues that don't exist yet
                logger.warning("Skipping %s: %s", url, exc)
                continue
            raw = _read_csv(raw_bytes)
            if raw.empty:
                continue
            frames.append(self._normalize_main(raw, league, start_year))
        if not frames:
            return self.validate(pd.DataFrame(columns=list(self.required_columns())))
        return self.validate(pd.concat(frames, ignore_index=True))

    def _normalize_main(self, raw: pd.DataFrame, league: str, start_year: int) -> pd.DataFrame:
        raw = raw.dropna(how="all").dropna(subset=["Date", "HomeTeam", "AwayTeam"])
        country, league_name = LEAGUE_CODES[league]
        out = pd.DataFrame(
            {
                "date": pd.to_datetime(raw["Date"], dayfirst=True, format="mixed", errors="coerce"),
                "country": country,
                "league": f"{country} - {league_name}",
                "league_code": league,
                "season": season_label(start_year),
                "home_team": raw["HomeTeam"].astype(str).str.strip(),
                "away_team": raw["AwayTeam"].astype(str).str.strip(),
                "home_goals": pd.to_numeric(raw["FTHG"], errors="coerce"),
                "away_goals": pd.to_numeric(raw["FTAG"], errors="coerce"),
                "source": self.name,
            }
        )
        out = _attach_extras(out, raw)
        out = out.dropna(subset=["date", "home_goals", "away_goals"]).reset_index(drop=True)
        out["match_id"] = (
            "fd_" + league + "_" + season_code(start_year) + "_" + out["date"].dt.strftime("%Y%m%d")
            + "_" + out["home_team"].map(_slug) + "_" + out["away_team"].map(_slug)
        )
        return out

    # ------------------------------------------------------------ extra files
    def fetch_extra_league(self, country_file: str, force_refresh: bool = False) -> pd.DataFrame:
        """One of the `/new/<COUNTRY>.csv` developing-market files (all seasons)."""
        if country_file not in EXTRA_LEAGUE_FILES:
            raise ValueError(f"Unknown extra-league file {country_file!r}; options: {sorted(EXTRA_LEAGUE_FILES)}")
        url = f"{BASE_URL}/new/{country_file}.csv"
        raw = _read_csv(self.http.get_bytes(url, force=force_refresh))
        raw = raw.dropna(how="all").dropna(subset=["Date", "Home", "Away"])
        country = EXTRA_LEAGUE_FILES[country_file]
        league_name = raw["League"].astype(str).str.strip() if "League" in raw else pd.Series(country, index=raw.index)
        out = pd.DataFrame(
            {
                "date": pd.to_datetime(raw["Date"], dayfirst=True, format="mixed", errors="coerce"),
                "country": country,
                "league": country + " - " + league_name,
                "league_code": country_file,
                "season": raw["Season"].astype(str).str.strip().str.replace(r"/(\d{2})(\d{2})$", r"/\2", regex=True),
                "home_team": raw["Home"].astype(str).str.strip(),
                "away_team": raw["Away"].astype(str).str.strip(),
                "home_goals": pd.to_numeric(raw["HG"], errors="coerce"),
                "away_goals": pd.to_numeric(raw["AG"], errors="coerce"),
                "source": self.name,
            }
        )
        out = _attach_extras(out, raw)
        out = out.dropna(subset=["date", "home_goals", "away_goals"]).reset_index(drop=True)
        out["match_id"] = (
            "fdx_" + country_file + "_" + out["date"].dt.strftime("%Y%m%d")
            + "_" + out["home_team"].map(_slug) + "_" + out["away_team"].map(_slug)
        )
        out = out.drop_duplicates(subset="match_id").reset_index(drop=True)
        return self.validate(out)


def _read_csv(raw_bytes: bytes) -> pd.DataFrame:
    text = raw_bytes.decode("utf-8-sig", errors="replace")
    from io import StringIO

    try:
        df = pd.read_csv(StringIO(text), on_bad_lines="skip", low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    df = df.loc[:, [c for c in df.columns if not str(c).startswith("Unnamed")]]
    return df


def _attach_extras(out: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    for src, dst in STAT_COLUMNS.items():
        if src in raw.columns:
            out[dst] = raw[src] if dst in ("referee", "kickoff_time") else pd.to_numeric(raw[src], errors="coerce")
    for col in raw.columns:
        if _ODDS_1X2_RE.match(str(col)) or _ODDS_OU_RE.match(str(col)):
            out[col] = pd.to_numeric(raw[col], errors="coerce")
    return out


def odds_wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """Melt bookmaker odds columns into the canonical long odds table.

    Produces one row per (match, bookmaker, market, side) with an is_closing
    flag, so both sides of every market are stored (needed for no-vig fair
    probabilities and CLV later).
    """
    side_map = {"H": "home", "D": "draw", "A": "away", ">": "over", "<": "under"}
    records = []
    odds_cols = [c for c in df.columns if _ODDS_1X2_RE.match(str(c)) or _ODDS_OU_RE.match(str(c))]
    if not odds_cols:
        return pd.DataFrame(columns=["match_id", "bookmaker", "market", "side", "price", "is_closing", "snapshot_ts"])
    sub = df[["match_id", "date", *odds_cols]]
    melted = sub.melt(id_vars=["match_id", "date"], var_name="col", value_name="price").dropna(subset=["price"])
    melted = melted[melted["price"] > 1.0]

    def parse(col: str) -> tuple[str, str, str, bool]:
        m = _ODDS_1X2_RE.match(col)
        if m:
            book, closing, side = m.groups()
            return book, "1x2", side_map[side], closing == "C"
        m = _ODDS_OU_RE.match(col)
        assert m is not None
        book, side = m.groups()
        closing = book.endswith("C") and book not in ("PC",) or book == "PC"
        book = book[:-1] if book.endswith("C") and len(book) > 1 else book
        return book, "totals_2.5", side_map[side], closing

    parsed = melted["col"].map(parse)
    melted["bookmaker"] = parsed.map(lambda t: t[0])
    melted["market"] = parsed.map(lambda t: t[1])
    melted["side"] = parsed.map(lambda t: t[2])
    melted["is_closing"] = parsed.map(lambda t: t[3])
    melted["snapshot_ts"] = melted["date"]
    records = melted[["match_id", "bookmaker", "market", "side", "price", "is_closing", "snapshot_ts"]]
    return records.reset_index(drop=True)
