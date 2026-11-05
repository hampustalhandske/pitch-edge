"""Ingestion orchestrator: every source -> warehouse, with per-source health logging.

`ingest_everything` is what the CLI, the scheduler and the LangGraph
"Data Scout" node call. Each source runs in isolation; one failing source is
logged to `pipeline_runs` and never blocks the others.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd

from pitch_edge.config import get_settings
from pitch_edge.data.alt.news import NewsScanner
from pitch_edge.data.alt.venues import VenueGeocoder
from pitch_edge.data.alt.weather import OpenMeteoWeather
from pitch_edge.data.fd_vocab import EXTRA_DIVISION_COUNTRIES, LEAGUE_CODES, odds_wide_to_long
from pitch_edge.data.quality import find_duplicate_fixtures
from pitch_edge.data.sources.club_elo import ClubEloSource
from pitch_edge.data.sources.openfootball import OpenFootballSource
from pitch_edge.data.sources.statsbomb import StatsBombOpenDataSource
from pitch_edge.data.storage import Warehouse, _clean_frame
from pitch_edge.data.teams import TeamNameResolver

logger = logging.getLogger(__name__)


def latest_season_start(today: date | None = None) -> int:
    """Start year of the European season under way on `today` (default: today, UTC): seasons start on 1 July,
    so 2026-10-28 gives 2026 (2026/27) and 2026-05-01 gives 2025 (2025/26)."""
    today = today or datetime.now(UTC).date()
    return today.year if today.month >= 7 else today.year - 1


MATCH_CORE_COLUMNS = [
    "match_id",
    "date",
    "country",
    "league",
    "league_code",
    "season",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "source",
    "referee",
    "kickoff_time",
    "attendance",
    "ht_home_goals",
    "ht_away_goals",
    "home_shots",
    "away_shots",
    "home_shots_on_target",
    "away_shots_on_target",
    "home_fouls",
    "away_fouls",
    "home_corners",
    "away_corners",
    "home_yellows",
    "away_yellows",
    "home_reds",
    "away_reds",
    "home_elo",
    "away_elo",
    "home_form3",
    "home_form5",
    "away_form3",
    "away_form5",
]

# Tables with rows per `matches.match_id`, removed together with the match.
MATCH_KEYED_TABLES = ("odds", "weather", "features")

# StatsBomb open competitions with full event data, small enough to ingest end-to-end.
DEFAULT_STATSBOMB_COMPETITIONS: list[tuple[int, int]] = [
    (11, 90),  # La Liga 2020/21
    (11, 42),  # La Liga 2019/20
    (2, 27),  # Premier League 2015/16
    (43, 106),  # FIFA World Cup 2022
    (55, 43),  # UEFA Euro 2020
    (9, 281),  # Bundesliga 2023/24
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


def _log_progress(source: str, fetched: int, planned: int, skipped: int = 0) -> None:
    """One line per source: how much this run actually fetched out of what was planned. `skipped`
    is how many of `planned` were already in the warehouse and never even attempted — the whole
    point of tracking this separately from `_run`'s "N new rows" line, which only reports what
    got written, not what was skipped or attempted."""
    extra = f" ({skipped} already stored, skipped)" if skipped else ""
    logger.info("%s: fetching %d/%d planned%s", source, fetched, planned, extra)


def _pending_seasons(seasons: list[int], stored_labels: set[str], label_fn: Callable[[int], str]) -> list[int]:
    """Drop seasons already fully stored, except the most recent one — it's still being played, so
    it always needs a fresh fetch even if some of its matches are already in the warehouse."""
    latest = max(seasons)
    return [y for y in seasons if y == latest or label_fn(y) not in stored_labels]


def prepare_match_tables(wh: Warehouse, df: pd.DataFrame) -> None:
    """Create `matches` and `odds`, or add the columns `df` would add, so a later `store_matches(wh, df)` alters
    no table. Callers that store inside a transaction that also deletes from these tables call this first:
    DuckDB cannot commit a transaction that both modifies and alters one table."""
    core = df[[c for c in MATCH_CORE_COLUMNS if c in df.columns]]
    for table, frame in (("matches", core), ("odds", odds_wide_to_long(df))):
        if not frame.empty:
            wh._ensure_table(table, _clean_frame(frame))  # noqa: SLF001


def store_matches(wh: Warehouse, df: pd.DataFrame) -> int:
    """Persist matches + long odds. football-data.co.uk rows are the preferred spine: if the same
    fixture (home, away, date) already exists from another source (e.g. openfootball), the old row and
    its odds are replaced, so ingestion order never matters."""
    if df.empty:
        return 0
    df = df.copy()
    if df["source"].iloc[0] == "football_data_co_uk" and wh.table_exists("matches"):
        _replace_secondary_duplicates(wh, df)
    core = df[[c for c in MATCH_CORE_COLUMNS if c in df.columns]]
    inserted = wh.upsert("matches", core)
    odds = odds_wide_to_long(df)
    if not odds.empty:
        wh.upsert("odds", odds)
    return inserted


def _replace_secondary_duplicates(wh: Warehouse, df: pd.DataFrame) -> None:
    """Delete stored non-football-data rows (and their odds) whose exact (home, away, date) is in `df`."""
    wh._conn.register(
        "incoming_keys", df[["home_team", "away_team", "date"]].assign(d=df["date"].dt.strftime("%Y-%m-%d"))
    )
    dupes = wh.query(
        """
        SELECT m.match_id
        FROM matches m JOIN incoming_keys k
          ON m.home_team = k.home_team AND m.away_team = k.away_team AND strftime(m.date, '%Y-%m-%d') = k.d
        WHERE m.source <> 'football_data_co_uk'
        """
    )
    wh._conn.unregister("incoming_keys")
    if dupes.empty:
        return
    ids = dupes["match_id"].tolist()
    wh._conn.execute("DELETE FROM matches WHERE match_id IN (SELECT UNNEST(?::VARCHAR[]))", [ids])
    if wh.table_exists("odds"):
        wh._conn.execute("DELETE FROM odds WHERE match_id IN (SELECT UNNEST(?::VARCHAR[]))", [ids])
    logger.info("replaced %d secondary-source rows with football-data.co.uk rows", len(ids))


def purge_duplicate_fixtures(wh: Warehouse) -> int:
    """Delete every non-football-data.co.uk `matches` row that `find_duplicate_fixtures` pairs with a
    football-data.co.uk row (the same fixture under another spelling), with its rows in the match-keyed
    tables (`odds`, `weather`, `features`). Unpaired rows are kept. Returns the number of matches removed;
    with no such pairs nothing is touched, so running it repeatedly is harmless."""
    pairs = find_duplicate_fixtures(wh)
    if pairs.empty:
        return 0
    spine = "football_data_co_uk"
    ids: set[str] = set()
    for p in pairs.itertuples(index=False):
        if p.source_a == spine and p.source_b != spine:
            ids.add(p.match_id_b)
        elif p.source_b == spine and p.source_a != spine:
            ids.add(p.match_id_a)
    if not ids:
        return 0
    ordered = sorted(ids)
    for table in ("matches", *MATCH_KEYED_TABLES):
        if wh.table_exists(table):
            wh._conn.execute(f"DELETE FROM {table} WHERE match_id IN (SELECT UNNEST(?::VARCHAR[]))", [ordered])
    logger.info("purged %d matches that duplicate a football-data.co.uk fixture", len(ordered))
    return len(ordered)


# ---------------------------------------------------------------------- sources
def build_football_data_spine(wh: Warehouse, force: bool = False) -> int:
    """Rebuild the football-data `matches` and `odds` rows from the `lake_fd_matches` mirror (health-logged).

    Takes the warehouse and `force` (rebuild every window, not only those whose mirror rows changed). Returns
    the matches written; 0 when the mirror has never been synced (run `pitch-edge lake-sync --only fd_matches`).
    Reads the mirror only: it neither calls the lake nor fetches anything, so it is as fresh as the last sync."""
    from pitch_edge.lake.spine import MIRROR_TABLE, build_fd_spine

    def go() -> int:
        if not wh.table_exists(MIRROR_TABLE):
            logger.warning("%s is missing; run `pitch-edge lake-sync --only fd_matches` first", MIRROR_TABLE)
            return 0
        return build_fd_spine(wh, force=force)["matches"]

    return _run(wh, "football_data_spine", go)


def ingest_club_elo(wh: Warehouse, dates: list[str] | None = None) -> int:
    from pitch_edge.data.http import HostCircuitOpenError

    src = ClubEloSource()
    # Every date below hits the same host — one 502 already means the whole API is down, not just
    # that date, so trip the circuit fast instead of waiting for the client's general-purpose
    # default (6 consecutive failures, tuned for sources with occasional per-item blips).
    src.http.max_consecutive_failures = 2
    today = datetime.now(UTC).date()
    dates = dates or [f"{y}-{m:02d}-01" for y in range(2010, today.year + 1) for m in (1, 7) if date(y, m, 1) <= today]

    def go() -> int:
        total = 0
        teams = (
            wh.query("SELECT DISTINCT home_team AS t FROM matches")["t"].tolist() if wh.table_exists("matches") else []
        )
        resolver = TeamNameResolver(teams)
        already = (
            set(
                wh.query("SELECT DISTINCT CAST(date AS DATE) AS d FROM team_ratings WHERE source = 'club_elo'")[
                    "d"
                ].astype(str)
            )
            if wh.table_exists("team_ratings")
            else set()
        )
        pending = [d for d in dates if d not in already]
        _log_progress("club_elo", len(pending), len(dates), len(dates) - len(pending))
        for i, d in enumerate(pending):
            try:
                df = src.fetch_ratings_by_date(d)
            except HostCircuitOpenError:
                logger.warning(
                    "club_elo: host circuit open (repeated failures) — stopping early, %d date(s) left untried",
                    len(pending) - i,
                )
                break
            except Exception as exc:
                logger.warning("club elo %s: %s", d, exc)
                continue
            df["team_raw"] = df["team"]
            df["team"] = df["team_raw"].map(lambda n: resolver.resolve(n) or n)
            df["date"] = df["as_of"]
            df["source"] = "club_elo"
            total += wh.upsert(
                "team_ratings",
                df[["team", "team_raw", "date", "elo", "rank", "country", "level", "source"]].rename(
                    columns={"rank": "elo_rank"}
                ),
            )
        return total

    return _run(wh, "club_elo", go)


def ingest_openfootball(wh: Warehouse, leagues: list[str] | None = None, seasons: list[int] | None = None) -> int:
    src = OpenFootballSource()
    leagues = leagues or [
        "en.1",
        "de.1",
        "es.1",
        "it.1",
        "fr.1",
        "at.1",
        "ch.1",
        "hu.1",
        "cz.1",
        "mx.1",
        "br.1",
        "jp.1",
    ]
    seasons = seasons or list(range(2015, latest_season_start() + 1))

    def go() -> int:
        total = 0
        for lg in leagues:
            stored = (
                set(wh.query("SELECT DISTINCT season FROM openfootball_matches WHERE league_code = ?", [lg])["season"])
                if wh.table_exists("openfootball_matches")
                else set()
            )
            pending = _pending_seasons(seasons, stored, lambda y: f"{y}/{(y + 1) % 100:02d}")
            _log_progress(f"openfootball:{lg}", len(pending), len(seasons), len(seasons) - len(pending))
            if not pending:
                continue
            df = src.fetch_matches(league=lg, seasons=pending)
            if df.empty:
                continue
            total += wh.upsert("openfootball_matches", df, keys=["match_id"])
        return total

    return _run(wh, "openfootball", go)


def ingest_statsbomb(
    wh: Warehouse,
    competitions: list[tuple[int, int]] | None = None,
    with_events: bool = True,
    max_matches_per_competition: int | None = None,
) -> int:
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
            stored_ids = (
                set(wh.query("SELECT DISTINCT statsbomb_match_id FROM statsbomb_events")["statsbomb_match_id"])
                if wh.table_exists("statsbomb_events")
                else set()
            )
            pending_ids = [mid for mid in ids if mid not in stored_ids]
            _log_progress(
                f"statsbomb_open_data:{comp_id}/{season_id} events",
                len(pending_ids),
                len(ids),
                len(ids) - len(pending_ids),
            )
            for mid in pending_ids:
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
        teams = (
            wh.query("SELECT DISTINCT home_team AS t FROM matches")["t"].tolist() if wh.table_exists("matches") else []
        )
        scored = scanner.score(items, teams)
        return wh.upsert("news_items", scored)

    return _run(wh, "news_rss", go)


def pmxt_scanned_days(wh: Warehouse) -> set[str]:
    if not wh.table_exists("pmxt_days_scanned"):
        return set()
    return set(wh.read("pmxt_days_scanned")["day"].astype(str))


def ingest_pmxt_soccer_markets(
    wh: Warehouse, earliest_day: date = date(2026, 4, 13), future_buffer_days: int = 14
) -> int:
    """Resolve every soccer market whose real match day falls in
    `[earliest_day, today + future_buffer_days]` into `dim_soccer_markets` — the `condition_id`
    lookup `ingest_pmxt_orderbook_hour` filters the archive against.

    Walks backward from `today + future_buffer_days` (the buffer catches matches whose game is
    just past the archive's newest hour but whose pre-match ticks fall inside it — see
    `pmxt_archive.py`'s module docstring) one calendar day at a time, stopping the instant it
    reaches a day already recorded in `pmxt_days_scanned` or passes `earliest_day`. A day's set
    of matches never changes once recorded, so this makes re-running genuinely incremental: the
    first run walks the whole range, every run after that only touches new day(s) since the
    last one — never re-scanning history that's already known."""
    from pitch_edge.data.alt.pmxt_archive import PMXTArchiveSource

    src = PMXTArchiveSource()
    scanned = pmxt_scanned_days(wh)

    def go() -> int:
        total_new = 0
        day = date.today() + timedelta(days=future_buffer_days)
        while day >= earliest_day:
            day_str = day.isoformat()
            if day_str in scanned:
                logger.info("pmxt_soccer_markets: %s already scanned — stopping walk-back", day_str)
                break
            df = src.discover_soccer_markets_for_day(day)
            total_new += wh.upsert("dim_soccer_markets", df)
            wh.upsert("pmxt_days_scanned", pd.DataFrame([{"day": day_str, "n_markets": len(df)}]))
            day -= timedelta(days=1)
        return total_new

    return _run(wh, "pmxt_soccer_markets", go)


def pmxt_hour_already_ingested(wh: Warehouse, date: str, hour: int) -> bool:
    """True if this hour's file has already been successfully processed — a re-run must never
    re-download a ~100MB archive file it already has rows from."""
    if not wh.table_exists("pipeline_runs"):
        return False
    rows = wh.query(
        "SELECT 1 FROM pipeline_runs WHERE source = ? AND status = 'ok' LIMIT 1",
        [f"pmxt_orderbook:{date}T{hour:02d}"],
    )
    return not rows.empty


def ingest_pmxt_orderbook_hour(wh: Warehouse, date: str, hour: int, force: bool = False) -> int:
    """One hour of the PMXT Polymarket orderbook archive, filtered to soccer markets already
    resolved in `dim_soccer_markets` (run `ingest_pmxt_soccer_markets` first).

    Skips the fetch entirely (no network call) if this hour already has a logged successful
    run, unless `force=True` — makes a multi-year backfill restartable for free: killing and
    re-running `ingest_pmxt_orderbook_range` never re-downloads an hour it already has."""
    from pitch_edge.data.alt.pmxt_archive import PMXTArchiveSource

    if not force and pmxt_hour_already_ingested(wh, date, hour):
        return 0

    src = PMXTArchiveSource()

    def go() -> int:
        if not wh.table_exists("dim_soccer_markets"):
            logger.warning("pmxt_orderbook_hour: dim_soccer_markets is empty — run ingest_pmxt_soccer_markets first")
            return 0
        condition_ids = wh.read("dim_soccer_markets")["condition_id"].tolist()
        df = src.fetch_orderbook_hour(date, hour, condition_ids)
        return wh.upsert("pmxt_orderbook", df)

    return _run(wh, f"pmxt_orderbook:{date}T{hour:02d}", go)


def ingest_pmxt_orderbook_range(wh: Warehouse, start: datetime, end: datetime) -> dict[str, int]:
    """Every hour in `[start, end]` inclusive, **newest first** — matches
    `ingest_pmxt_soccer_markets`'s walk-backward order, and for the same reason: whatever hours
    we don't get to are the least valuable ones to have missed (the oldest, furthest from
    "now"), not the most recent.

    Interruptible and resumable: a killed run just leaves earlier hours unfetched, and calling
    this again picks up exactly where it left off (see `pmxt_hour_already_ingested`).

    Logs one line per hour at INFO — the archive is genuinely gappy and a full backfill can run
    for a long time, so silent multi-hour stretches with no sign of life aren't acceptable; a
    caller redirecting stdout to a log file should run with `PYTHONUNBUFFERED=1` (or Python's
    `-u`) so these lines actually land in the file as they happen instead of sitting in a
    fully-buffered pipe until the whole range finishes."""
    total_hours = int((end - start).total_seconds() // 3600) + 1
    report: dict[str, int] = {}
    cursor = end.replace(minute=0, second=0, microsecond=0)
    start = start.replace(minute=0, second=0, microsecond=0)
    i = 0
    while cursor >= start:
        i += 1
        date_str = f"{cursor:%Y-%m-%d}"
        key = f"{date_str}T{cursor.hour:02d}"
        already_done = pmxt_hour_already_ingested(wh, date_str, cursor.hour)
        n = ingest_pmxt_orderbook_hour(wh, date_str, cursor.hour)
        report[key] = n
        status = "skip(already done)" if already_done else (f"+{n} rows" if n else "gap (no file)")
        logger.info("pmxt_orderbook_range [%d/%d] %s: %s", i, total_hours, key, status)
        cursor -= timedelta(hours=1)
    return report


def build_pmxt_match_map(wh: Warehouse) -> int:
    """Recompute `pmxt_match_map` (condition_id -> match_id, outcome_side) from whatever
    `dim_soccer_markets`/`matches` currently hold. Cheap, idempotent, no network calls — safe to
    re-run any time either table grows (new markets discovered, new matches ingested)."""
    from pitch_edge.data.alt.pmxt_match_map import build_pmxt_match_map as _build

    def go() -> int:
        return len(_build(wh))

    return _run(wh, "pmxt_match_map", go)


def ingest_sweden(wh: Warehouse, seasons: tuple[int, ...] = (2023, 2024, 2025, 2026)) -> int:
    """Swedish Allsvenskan / Superettan / Ettan results from openfootball/europe (text schedules)."""
    src = OpenFootballSource()

    def go() -> int:
        df = src.fetch_sweden(seasons)
        if df.empty:
            return 0
        # de-dup against spine rows (football-data.co.uk SWE file, when reachable) by fixture key
        if wh.table_exists("matches"):
            existing = wh.query(
                "SELECT home_team, away_team, CAST(date AS DATE) AS d FROM matches WHERE league_code IN ('SWE','SWE1','SWE2','SWE3')"
            )
            if not existing.empty:
                resolver = TeamNameResolver(sorted(set(existing["home_team"]) | set(existing["away_team"])))
                key_existing = set(
                    zip(existing["home_team"], existing["away_team"], existing["d"].astype(str), strict=True)
                )
                keys = [
                    (resolver.resolve(h) or h, resolver.resolve(a) or a, d)
                    for h, a, d in zip(
                        df["home_team"], df["away_team"], df["date"].dt.strftime("%Y-%m-%d"), strict=True
                    )
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
        teams = (
            wh.query("SELECT DISTINCT home_team AS t FROM matches WHERE league_code LIKE 'SWE%'")["t"].tolist()
            if wh.table_exists("matches")
            else []
        )
        for team in teams[:40]:
            info = src.team_lookup(team)
            if not info.empty:
                wh.upsert("tsdb_teams", info)
            players = src.team_players(team)
            if not players.empty:
                total += wh.upsert("tsdb_players", players)
        return total

    return _run(wh, "thesportsdb:sweden", go)


def ingest_transfermarkt_open(
    wh: Warehouse,
    tables: tuple[str, ...] = (
        "competitions",
        "clubs",
        "players",
        "games",
        "game_events",
        "appearances",
        "game_lineups",
        "player_valuations",
    ),
) -> int:
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
        teams = (
            wh.query("SELECT DISTINCT home_team AS t FROM matches")["t"].tolist() if wh.table_exists("matches") else []
        )
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
    include_statsbomb: bool = True,
    include_weather: bool = True,
    statsbomb_max_matches: int | None = None,
    include_players: bool = True,
) -> dict[str, int]:
    report: dict[str, int] = {}
    # Matches come from football-data (the spine, built from the `lake_fd_matches` mirror that `lake-sync`
    # keeps current) and openfootball's Swedish schedules. Stored rows of any other source that repeat a
    # football-data fixture are purged first.
    purged = purge_duplicate_fixtures(wh)
    if purged:
        report["purged_matches"] = purged
    report["football_data_spine"] = build_football_data_spine(wh)
    # `ingest_sweden` is a real primary source for Swedish leagues; its rows yield to football-data's.
    if include_statsbomb:
        report["statsbomb"] = ingest_statsbomb(wh, max_matches_per_competition=statsbomb_max_matches)
    if include_weather:
        report["venues_weather"] = ingest_venues_and_weather(wh)
    report["sweden_openfootball"] = ingest_sweden(wh)
    report["news"] = ingest_news(wh)
    # Market-probability data is Polymarket-only now, via the PMXT archive (see
    # `pmxt_archive.py`): the old live-snapshot connectors (`polymarket.py`, `kalshi.py`,
    # `odds_api.py`, all deleted) and football-data.co.uk's own bookmaker-odds columns (no longer
    # parsed at all) were removed as redundant with it / replaced by it.
    # Discovery is cheap and safe to run on every ingest; the actual hour-by-hour tick backfill is
    # deliberately NOT run here (a 1-2 year backfill is thousands of ~100MB files) — see
    # `ingest_pmxt_orderbook_range` / the `pitch-edge ingest-pmxt START END` CLI command for that.
    report["pmxt_soccer_markets"] = ingest_pmxt_soccer_markets(wh)
    if include_players:
        report["transfermarkt_open"] = ingest_transfermarkt_open(wh)
        report["thesportsdb_sweden"] = ingest_thesportsdb_sweden(wh)
    report["wikipedia_pageviews"] = ingest_wikipedia_attention(wh)
    report["referee_announcements"] = ingest_referee_announcements(wh)
    # club_elo and api_football are intentionally NOT called here: api.clubelo.com has been down
    # for the life of this project (every run returns 0 rows, see `ingest_club_elo`) and
    # API-Football's free tier (100 requests/day) never returned usable data either. Both
    # functions are still defined and callable directly if a working replacement is found later.
    report["espn_soccer_data"] = ingest_espn_soccer_data(wh)
    return report


def ingest_espn_soccer_data(wh: Warehouse, archive_dir: str | Path | None = None) -> int:
    """Load the locally-downloaded `espn-soccer-data` archive into its own `espn_*` DuckDB tables.
    Soft no-op (logged, not an error) if the archive directory isn't present — this is manually
    downloaded, not fetched over the network like every other source here."""
    from pitch_edge.data.sources.espn_soccer_data import build_espn_mapping, load_espn_soccer_data

    base = Path(archive_dir) if archive_dir else get_settings().data_dir / "espn-soccer-data"

    def go() -> int:
        if not base.exists():
            logger.info("espn_soccer_data: %s not found — skipping", base)
            return 0
        counts = load_espn_soccer_data(wh, base)
        for table, n in counts.items():
            logger.info("espn_soccer_data: %s -> %d rows", table, n)
        mapping = build_espn_mapping(wh)
        for table, n in mapping.items():
            logger.info("espn_soccer_data: %s -> %d", table, n)
        return sum(counts.values())

    return _run(wh, "espn_soccer_data", go)
