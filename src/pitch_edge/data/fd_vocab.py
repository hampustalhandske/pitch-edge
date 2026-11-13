"""football-data.co.uk vocabulary shared by the warehouse spine and the other match sources.

Holds the division catalogue (main leagues and the developing-market "extra" files), the season-label and
fixture-id conventions of the `matches` table, and the wide-to-long melt that turns a football-data odds
row (columns like `PSH`, `PSCH`, `B365>2.5`) into canonical `odds` rows. Nothing here reads a network or a
file: football-data rows reach the warehouse through the lake mirror (`pitch_edge.lake.spine`).
"""

from __future__ import annotations

import re

import pandas as pd

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

# Division code -> (country, league name) for the "extra" leagues (a few codes differ from the file stems).
EXTRA_DIVISION_COUNTRIES: dict[str, tuple[str, str]] = {
    "ARG": ("Argentina", "Liga Profesional"),
    "AUT": ("Austria", "Bundesliga"),
    "BRA": ("Brazil", "Serie A"),
    "CHN": ("China", "Super League"),
    "DEN": ("Denmark", "Superliga"),
    "DNK": ("Denmark", "Superliga"),
    "FIN": ("Finland", "Veikkausliiga"),
    "IRL": ("Ireland", "Premier Division"),
    "JAP": ("Japan", "J1 League"),
    "JPN": ("Japan", "J1 League"),
    "MEX": ("Mexico", "Liga MX"),
    "NOR": ("Norway", "Eliteserien"),
    "POL": ("Poland", "Ekstraklasa"),
    "ROM": ("Romania", "Liga I"),
    "ROU": ("Romania", "Liga I"),
    "RUS": ("Russia", "Premier League"),
    "SWE": ("Sweden", "Allsvenskan"),
    "SUI": ("Switzerland", "Super League"),
    "SWZ": ("Switzerland", "Super League"),
    "USA": ("USA", "MLS"),
}

EXTRA_LEAGUE_FILES: dict[str, str] = {  # file stem -> country
    "ARG": "Argentina",
    "AUT": "Austria",
    "BRA": "Brazil",
    "CHN": "China",
    "DNK": "Denmark",
    "FIN": "Finland",
    "IRL": "Ireland",
    "JPN": "Japan",
    "MEX": "Mexico",
    "NOR": "Norway",
    "POL": "Poland",
    "ROU": "Romania",
    "RUS": "Russia",
    "SWE": "Sweden",
    "SWZ": "Switzerland",
    "USA": "USA",
}

# 1X2 odds columns: <bookmaker><C?><H|D|A>. "C" marks a closing price.
ODDS_1X2_RE = re.compile(r"^(B365|BW|IW|PS|WH|VC|Max|Avg|Mkt|BFE|BF|BS|GB|LB|SB|SJ|SY|SO|1XB|P)(C?)(H|D|A)$")
# Totals (over/under 2.5) columns: <bookmaker><C?><>|<>2.5
ODDS_OU_RE = re.compile(r"^(B365|P|Max|Avg|Mkt|BFE|PC|MaxC|AvgC|B365C|BFEC|GB|BW|IW|LB|SB|SJ|WH|VC)([<>])2\.5$")


def season_code(start_year: int) -> str:
    """2023 -> '2324'."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def season_label(start_year: int) -> str:
    """2023 -> '2023/24'."""
    return f"{start_year}/{(start_year + 1) % 100:02d}"


def slug(text: str) -> str:
    """Team name with every non-alphanumeric character removed (the fixture-id form)."""
    return re.sub(r"[^A-Za-z0-9]+", "", str(text))


def odds_wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """Melt bookmaker odds columns into the canonical long odds table.

    Takes a frame with `match_id`, `date` and football-data odds columns; returns one row per (match,
    bookmaker, market, side) with columns match_id, bookmaker, market, side, price, is_closing, snapshot_ts.
    `is_closing` marks a `...C...` column (a close is hindsight, an open is not) and `snapshot_ts` is the
    match date at 00:00, for opens and closes alike. Prices <= 1 are dropped.
    """
    side_map = {"H": "home", "D": "draw", "A": "away", ">": "over", "<": "under"}
    records = []
    odds_cols = [c for c in df.columns if ODDS_1X2_RE.match(str(c)) or ODDS_OU_RE.match(str(c))]
    if not odds_cols:
        return pd.DataFrame(columns=["match_id", "bookmaker", "market", "side", "price", "is_closing", "snapshot_ts"])
    sub = df[["match_id", "date", *odds_cols]]
    melted = sub.melt(id_vars=["match_id", "date"], var_name="col", value_name="price").dropna(subset=["price"])
    melted = melted[melted["price"] > 1.0]

    def parse(col: str) -> tuple[str, str, str, bool]:
        m = ODDS_1X2_RE.match(col)
        if m:
            book, closing, side = m.groups()
            return book, "1x2", side_map[side], closing == "C"
        m = ODDS_OU_RE.match(col)
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
