"""Ingestion orchestrator: every source -> warehouse, with per-source health logging.

`ingest_everything` is what the CLI, the scheduler and the LangGraph
"Data Scout" node call. Each source runs in isolation; one failing source is
logged to `pipeline_runs` and never blocks the others.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, datetime

import pandas as pd

from pitch_edge.data.alt.news import NewsScanner
from pitch_edge.data.alt.polymarket import PolymarketSource
from pitch_edge.data.alt.venues import VenueGeocoder
from pitch_edge.data.alt.weather import OpenMeteoWeather
from pitch_edge.data.sources.club_elo import ClubEloSource
from pitch_edge.data.sources.club_football_match_data import ClubFootballMatchDataSource
from pitch_edge.data.sources.football_data_co_uk import (
    EXTRA_DIVISION_COUNTRIES,
    EXTRA_LEAGUE_FILES,
    LEAGUE_CODES,
    FootballDataCoUkSource,
    odds_wide_to_long,
)
from pitch_edge.data.sources.openfootball import OpenFootballSource
from pitch_edge.data.sources.statsbomb import StatsBombOpenDataSource
from pitch_edge.data.storage import Warehouse
from pitch_edge.data.teams import TeamNameResolver

logger = logging.getLogger(__name__)

MATCH_CORE_COLUMNS = [
    "match_id", "date", "country", "league", "league_code", "season", "home_team", "away_team",
    "home_goals", "away_goals", "source", "referee", "kickoff_time", "attendance",
    "ht_home_goals", "ht_away_goals", "home_shots", "away_shots", "home_shots_on_target", "away_shots_on_target",
    "home_fouls", "away_fouls", "home_corners", "away_corners", "home_yellows", "away_yellows",
    "home_reds", "away_reds", "home_elo", "away_elo", "home_form3", "home_form5", "away_form3", "away_form5",
]

# StatsBomb open competitions with full event data, small enough to ingest end-to-end.
DEFAULT_STATSBOMB_COMPETITIONS: list[tuple[int, int]] = [
    (11, 90),   # La Liga 2020/21
    (11, 42),   # La Liga 2019/20
    (2, 27),    # Premier League 2015/16
    (43, 106),  # FIFA World Cup 2022
    (55, 43),   # UEFA Euro 2020
    (9, 281),   # Bundesliga 2023/24
]


def _run(wh: Warehouse, source: str, fn: Callable[[], int]) -> int:
    started = datetime.now(UTC).replace(tzinfo=None)
    try:
        rows = fn()
        wh.log_run(source, "ok", rows=rows, started_at=started)
        logger.info("%s: %d new rows", source, rows)
        return rows
    except Exception as exc:  # noqa: BLE001 - we want every failure recorded, never fatal
        logger.exception("%s failed", source)
        wh.log_run(source, "error", error=repr(exc), started_at=started)
        return 0


def store_matches(wh: Warehouse, df: pd.DataFrame) -> int:
    """Persist matches + long odds. football-data.co.uk rows are the preferred spine: if the same
    fixture (home, away, date) already exists from a secondary source (xgabora), the old row and its
    odds are replaced and its Elo/form columns carried over — so ingestion order never matters."""
    if df.empty:
        return 0
    df = df.copy()
    if df["source"].iloc[0] == "football_data_co_uk" and wh.table_exists("matches"):
        df = _replace_secondary_duplicates(wh, df)
    core = df[[c for c in MATCH_CORE_COLUMNS if c in df.columns]]
    inserted = wh.upsert("matches", core)
    odds = odds_wide_to_long(df)
    if not odds.empty:
        wh.upsert("odds", odds)
    return inserted


def _replace_secondary_duplicates(wh: Warehouse, df: pd.DataFrame) -> pd.DataFrame:
    wh._conn.register("incoming_keys", df[["home_team", "away_team", "date"]].assign(d=df["date"].dt.strftime("%Y-%m-%d")))
    existing_cols = set(wh.columns("matches"))
    carry_cols = ["home_elo", "away_elo", "home_form5", "away_form5"]
    select_carry = ", ".join(f"m.{c}" if c in existing_cols else f"NULL AS {c}" for c in carry_cols)
    dupes = wh.query(
        f"""
        SELECT m.match_id, m.home_team, m.away_team, strftime(m.date, '%Y-%m-%d') AS d, {select_carry}
        FROM matches m JOIN incoming_keys k
          ON m.home_team = k.home_team AND m.away_team = k.away_team AND strftime(m.date, '%Y-%m-%d') = k.d
        WHERE m.source <> 'football_data_co_uk'
        """
    )
    wh._conn.unregister("incoming_keys")
    if dupes.empty:
        return df
    ids = dupes["match_id"].tolist()
    wh._conn.execute("DELETE FROM matches WHERE match_id IN (SELECT UNNEST(?::VARCHAR[]))", [ids])
    if wh.table_exists("odds"):
        wh._conn.execute("DELETE FROM odds WHERE match_id IN (SELECT UNNEST(?::VARCHAR[]))", [ids])
    carry = dupes.set_index(["home_team", "away_team", "d"])[["home_elo", "away_elo", "home_form5", "away_form5"]]
    key = pd.MultiIndex.from_arrays([df["home_team"], df["away_team"], df["date"].dt.strftime("%Y-%m-%d")])
    for col in carry.columns:
        carried = carry[col].reindex(key).to_numpy()
        df[col] = df[col].fillna(pd.Series(carried, index=df.index)) if col in df else carried
    logger.info("replaced %d secondary-source rows with football-data.co.uk rows", len(ids))
    return df


# ---------------------------------------------------------------------- sources
def ingest_football_data(wh: Warehouse, leagues: list[str] | None = None, seasons: list[int] | None = None,
                         extra_leagues: bool = True) -> dict[str, int]:
    src = FootballDataCoUkSource()
    leagues = leagues or list(LEAGUE_CODES)
    seasons = seasons or list(range(2005, 2026))
    counts: dict[str, int] = {}
    for league in leagues:
        counts[league] = _run(
            wh,
            f"football_data_co_uk:{league}",
            lambda league=league: store_matches(wh, src.fetch_matches(league=league, seasons=seasons)),
        )
    if extra_leagues:
        for stem in EXTRA_LEAGUE_FILES:
            counts[stem] = _run(
                wh, f"football_data_co_uk:new/{stem}", lambda stem=stem: store_matches(wh, src.fetch_extra_league(stem))
            )
    return counts


def ingest_club_football_match_data(wh: Warehouse) -> int:
    src = ClubFootballMatchDataSource()

    def go() -> int:
        df = src.fetch_matches()
        # Only keep divisions/dates that add information (Elo, form, or leagues we don't have).
        existing = wh.query("SELECT DISTINCT home_team, away_team, CAST(date AS DATE) AS d FROM matches") if wh.table_exists("matches") else pd.DataFrame()
        if not existing.empty:
            key_existing = set(zip(existing["home_team"], existing["away_team"], existing["d"].astype(str), strict=True))
            key_new = list(zip(df["home_team"], df["away_team"], df["date"].dt.strftime("%Y-%m-%d"), strict=True))
            dup_mask = pd.Series([k in key_existing for k in key_new], index=df.index)
            # Attach Elo/form to spine rows instead of inserting duplicates.
            dupes = df[dup_mask]
            if not dupes.empty:
                _attach_elo_to_spine(wh, dupes)
            df = df[~dup_mask]
        return store_matches(wh, df)

    return _run(wh, "club_football_match_data", go)


def _attach_elo_to_spine(wh: Warehouse, dupes: pd.DataFrame) -> None:
    wh._conn.register("elo_df", dupes[["home_team", "away_team", "date", "home_elo", "away_elo", "home_form5", "away_form5"]].assign(d=lambda x: x["date"].dt.strftime("%Y-%m-%d")))
    for col in ("home_elo", "away_elo", "home_form5", "away_form5"):
        if col not in wh.columns("matches"):
            wh._conn.execute(f'ALTER TABLE matches ADD COLUMN "{col}" DOUBLE')
    wh._conn.execute(
        """
        UPDATE matches SET home_elo = e.home_elo, away_elo = e.away_elo,
                           home_form5 = e.home_form5, away_form5 = e.away_form5
        FROM elo_df e
        WHERE matches.home_team = e.home_team AND matches.away_team = e.away_team
          AND strftime(matches.date, '%Y-%m-%d') = e.d
        """
    )
    wh._conn.unregister("elo_df")


def ingest_club_elo(wh: Warehouse, dates: list[str] | None = None) -> int:
    src = ClubEloSource()
    dates = dates or [f"{y}-{m:02d}-01" for y in range(2010, 2026) for m in (1, 7)]

    def go() -> int:
        total = 0
        teams = wh.query("SELECT DISTINCT home_team AS t FROM matches")["t"].tolist() if wh.table_exists("matches") else []
        resolver = TeamNameResolver(teams)
        for d in dates:
            try:
                df = src.fetch_ratings_by_date(d)
            except Exception as exc:
                logger.warning("club elo %s: %s", d, exc)
                continue
            df["team_raw"] = df["team"]
            df["team"] = df["team_raw"].map(lambda n: resolver.resolve(n) or n)
            df["date"] = df["as_of"]
            df["source"] = "club_elo"
            total += wh.upsert("team_ratings", df[["team", "team_raw", "date", "elo", "rank", "country", "level", "source"]].rename(columns={"rank": "elo_rank"}))
        return total

    return _run(wh, "club_elo", go)


def ingest_openfootball(wh: Warehouse, leagues: list[str] | None = None, seasons: list[int] | None = None) -> int:
    src = OpenFootballSource()
    leagues = leagues or ["en.1", "de.1", "es.1", "it.1", "fr.1", "at.1", "ch.1", "hu.1", "cz.1", "mx.1", "br.1", "jp.1"]
    seasons = seasons or list(range(2015, 2026))

    def go() -> int:
        total = 0
        for lg in leagues:
            df = src.fetch_matches(league=lg, seasons=seasons)
            if df.empty:
                continue
            total += wh.upsert("openfootball_matches", df, keys=["match_id"])
        return total

    return _run(wh, "openfootball", go)


def ingest_statsbomb(wh: Warehouse, competitions: list[tuple[int, int]] | None = None, with_events: bool = True,
                     max_matches_per_competition: int | None = None) -> int:
    src = StatsBombOpenDataSource()
    competitions = competitions or DEFAULT_STATSBOMB_COMPETITIONS

    def go() -> int:
        total = 0
        for comp_id, season_id in competitions:
            matches = src.fetch_matches(competition_id=comp_id, season_id=season_id)
            if matches.empty:
                continue
            total += wh.upsert("statsbomb_matches", matches)
            if not with_events:
                continue
            ids = matches["statsbomb_match_id"].tolist()
            if max_matches_per_competition:
                ids = ids[:max_matches_per_competition]
            for mid in ids:
                try:
                    ev = src.fetch_events(int(mid))
                except Exception as exc:
                    logger.warning("events %s: %s", mid, exc)
                    continue
                if not ev.empty:
                    wh.upsert("statsbomb_events", ev, keys=["event_id"])
        return total

    return _run(wh, "statsbomb_open_data", go)


def ingest_venues_and_weather(wh: Warehouse, leagues: list[str] | None = None) -> int:
    leagues = leagues or ["E0", "E1", "D1", "SP1", "I1", "F1", "N1", "P1", "B1", "SC0", "T1"]

    def go() -> int:
        matches = wh.read("matches", "league_code IN (" + ",".join("?" * len(leagues)) + ")", leagues)
        if matches.empty:
            return 0
        teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
        divisions = {**LEAGUE_CODES, **EXTRA_DIVISION_COUNTRIES}
        by_code = matches["league_code"].map(lambda c: divisions.get(str(c), (None, None))[0])
        country_col = matches["country"] if "country" in matches else pd.Series(index=matches.index, dtype=object)
        countries = country_col.fillna(by_code).dropna().unique().tolist()
        coords = VenueGeocoder().build_team_coordinates(teams, countries)
        wh.upsert("venues", coords)
        wx = OpenMeteoWeather().weather_for_matches(
            matches[["match_id", "date", "home_team", *(["kickoff_time"] if "kickoff_time" in matches else [])]], coords
        )
        return wh.upsert("weather", wx) if not wx.empty else 0

    return _run(wh, "venues_weather", go)


def ingest_news(wh: Warehouse) -> int:
    scanner = NewsScanner()

    def go() -> int:
        items = scanner.fetch_all_feeds()
        teams = wh.query("SELECT DISTINCT home_team AS t FROM matches")["t"].tolist() if wh.table_exists("matches") else []
        scored = scanner.score(items, teams)
        return wh.upsert("news_items", scored)

    return _run(wh, "news_rss", go)


def ingest_polymarket(wh: Warehouse) -> int:
    src = PolymarketSource()
    return _run(wh, "polymarket", lambda: wh.upsert("market_snapshots", src.fetch_football_markets().assign(venue="polymarket")))


def ingest_kalshi(wh: Warehouse, max_series: int = 60) -> int:
    from pitch_edge.data.alt.kalshi import KalshiSource

    src = KalshiSource()
    return _run(wh, "kalshi", lambda: wh.upsert("market_snapshots", src.fetch_football_markets(max_series=max_series)))


def ingest_sweden(wh: Warehouse, seasons: tuple[int, ...] = (2023, 2024, 2025, 2026)) -> int:
    """Swedish Allsvenskan / Superettan / Ettan results from openfootball/europe (text schedules)."""
    src = OpenFootballSource()

    def go() -> int:
        df = src.fetch_sweden(seasons)
        if df.empty:
            return 0
        # de-dup against spine rows (football-data.co.uk SWE file, when reachable) by fixture key
        if wh.table_exists("matches"):
            existing = wh.query("SELECT home_team, away_team, CAST(date AS DATE) AS d FROM matches WHERE league_code IN ('SWE','SWE1','SWE2','SWE3')")
            if not existing.empty:
                resolver = TeamNameResolver(sorted(set(existing["home_team"]) | set(existing["away_team"])))
                key_existing = set(zip(existing["home_team"], existing["away_team"], existing["d"].astype(str), strict=True))
                keys = [
                    (resolver.resolve(h) or h, resolver.resolve(a) or a, d)
                    for h, a, d in zip(df["home_team"], df["away_team"], df["date"].dt.strftime("%Y-%m-%d"), strict=True)
                ]
                df = df[[k not in key_existing for k in keys]]
        return store_matches(wh, df)

    return _run(wh, "openfootball:sweden", go)


def ingest_thesportsdb_sweden(wh: Warehouse, seasons: tuple[str, ...] = ("2024", "2025", "2026")) -> int:
    from pitch_edge.data.sources.thesportsdb import SWEDISH_LEAGUES, TheSportsDBSource

    src = TheSportsDBSource()

    def go() -> int:
        total = 0
        for league_id in SWEDISH_LEAGUES:
            for season in seasons:
                ev = src.season_events(league_id, season)
                if not ev.empty:
                    total += wh.upsert("tsdb_events", ev)
        teams = wh.query("SELECT DISTINCT home_team AS t FROM matches WHERE league_code LIKE 'SWE%'")["t"].tolist() if wh.table_exists("matches") else []
        for team in teams[:40]:
            info = src.team_lookup(team)
            if not info.empty:
                wh.upsert("tsdb_teams", info)
            players = src.team_players(team)
            if not players.empty:
                total += wh.upsert("tsdb_players", players)
        return total

    return _run(wh, "thesportsdb:sweden", go)


def ingest_transfermarkt_open(wh: Warehouse, tables: tuple[str, ...] = ("competitions", "clubs", "players", "games", "game_events", "appearances", "game_lineups", "player_valuations")) -> int:
    """Open Transfermarkt extract (dcaribou): player history, substitutions, cards, valuations, lineups."""
    from pitch_edge.data.sources.transfermarkt_open import TransfermarktOpenSource

    src = TransfermarktOpenSource()

    def go() -> int:
        src.download()
        total = 0
        from pitch_edge.data.sources.transfermarkt_open import TABLES

        for t in tables:
            try:
                df = src.read_table(t)
                if df.empty:
                    continue
                keys = [k for k in TABLES[t][0] if k in df.columns] or [df.columns[0]]
                total += wh.upsert(f"tm_{t}", df, keys=keys)
                logger.info("tm_%s: %d rows", t, len(df))
            except Exception as exc:  # noqa: BLE001 - one table must not block the others
                logger.warning("tm_%s failed: %s", t, exc)
        return total

    return _run(wh, "transfermarkt_open", go)


# ------------------------------------------------------------- phase 5 sources
WIKI_LEAGUES = ["E0", "E1", "D1", "SP1", "I1", "F1", "N1", "P1", "B1", "SC0", "T1"]


def ingest_wikipedia_attention(
    wh: Warehouse, leagues: list[str] | None = None, since: str = "2015-07-01", recent_days: int | None = None
) -> int:
    """Daily English-Wikipedia pageviews for every club in `leagues` (article titles resolved once and
    stored in `wiki_articles`). `recent_days` limits the window for the hourly scheduler job."""
    from pitch_edge.data.alt.wikipedia_attention import WikipediaAttentionSource

    src = WikipediaAttentionSource()
    leagues = leagues or WIKI_LEAGUES

    def go() -> int:
        if not wh.table_exists("matches"):
            return 0
        ph = ",".join("?" * len(leagues))
        teams = wh.query(
            f"SELECT DISTINCT home_team AS t FROM matches WHERE league_code IN ({ph}) AND date >= ?", [*leagues, since]
        )["t"].tolist()
        known = wh.read("wiki_articles") if wh.table_exists("wiki_articles") else pd.DataFrame()
        todo = [t for t in teams if known.empty or t not in set(known["team"])]
        if todo:
            resolved = src.resolve_titles(todo)
            wh.upsert("wiki_articles", resolved)
            known = pd.concat([known, resolved], ignore_index=True) if not known.empty else resolved
        titles = known[known["team"].isin(teams)].dropna(subset=["article"])
        end = pd.Timestamp.utcnow().tz_localize(None).normalize() - pd.Timedelta(days=1)
        start = end - pd.Timedelta(days=recent_days) if recent_days else pd.Timestamp(since)
        pv = src.pageviews_for_teams(titles, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        return wh.upsert("wiki_pageviews", pv) if not pv.empty else 0

    return _run(wh, "wikipedia_pageviews", go)


def ingest_referee_announcements(wh: Warehouse) -> int:
    """Poll public referee-appointment pages; store first-seen timestamps."""
    from pitch_edge.data.alt.referee_announcements import RefereeAnnouncementSource

    src = RefereeAnnouncementSource()

    def go() -> int:
        teams = wh.query("SELECT DISTINCT home_team AS t FROM matches")["t"].tolist() if wh.table_exists("matches") else []
        found = src.poll(teams)
        if found.empty:
            return 0
        return wh.upsert("referee_announcements", found)  # first-seen wins: later polls do not overwrite announced_at

    return _run(wh, "referee_announcements", go)


def ingest_api_football_context(wh: Warehouse, leagues: list[str] | None = None, season: int | None = None) -> int:
    """Injuries per league-season when `API_FOOTBALL_KEY` is set (no-op without it, logged as such)."""
    from pitch_edge.data.alt.api_football import LEAGUE_IDS, APIFootballSource

    src = APIFootballSource()
    leagues = leagues or [c for c in ("E0", "D1", "SP1", "I1", "F1") if c in LEAGUE_IDS]
    season = season or (datetime.now(UTC).year - (1 if datetime.now(UTC).month < 7 else 0))

    def go() -> int:
        if not src.enabled:
            logger.info("API_FOOTBALL_KEY not set — skipping injuries/lineups (see `pitch-edge setup`)")
            return 0
        total = 0
        for code in leagues:
            inj = src.injuries(code, season)
            if not inj.empty:
                inj = inj.assign(season=season)
                total += wh.upsert("api_football_injuries", inj)
        return total

    return _run(wh, "api_football", go)


# ------------------------------------------------------------------------ all
def ingest_everything(
    wh: Warehouse,
    leagues: list[str] | None = None,
    seasons: list[int] | None = None,
    include_statsbomb: bool = True,
    include_weather: bool = True,
    statsbomb_max_matches: int | None = None,
    include_players: bool = True,
) -> dict[str, int]:
    report: dict[str, int] = {}
    report.update(ingest_football_data(wh, leagues, seasons))
    report["club_football_match_data"] = ingest_club_football_match_data(wh)
    report["club_elo"] = ingest_club_elo(wh)
    report["openfootball"] = ingest_openfootball(wh)
    if include_statsbomb:
        report["statsbomb"] = ingest_statsbomb(wh, max_matches_per_competition=statsbomb_max_matches)
    if include_weather:
        report["venues_weather"] = ingest_venues_and_weather(wh)
    report["sweden_openfootball"] = ingest_sweden(wh)
    report["news"] = ingest_news(wh)
    report["polymarket"] = ingest_polymarket(wh)
    report["kalshi"] = ingest_kalshi(wh)
    if include_players:
        report["transfermarkt_open"] = ingest_transfermarkt_open(wh)
        report["thesportsdb_sweden"] = ingest_thesportsdb_sweden(wh)
    report["wikipedia_pageviews"] = ingest_wikipedia_attention(wh)
    report["referee_announcements"] = ingest_referee_announcements(wh)
    report["api_football"] = ingest_api_football_context(wh)
    return report
