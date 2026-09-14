"""ESPN soccer data — a large local archive, not a live API.

Unlike every other source in `data/sources/`, this one is never fetched over the network: the
project owner downloaded the `espn-soccer-data` GitHub scrape (base match/team/player metadata
plus per-competition play-by-play, key events, lineups, player stats and text commentary) to
`data/espn-soccer-data/` ahead of time. This module's only job is loading those ~1,400 CSV files
into DuckDB as their own `espn_*` tables, reachable by plain SQL.

Coverage is broad but shallow in time: ~224 competitions worldwide, 2024-2026, including genuine
*scheduled* (not yet played) fixtures — much wider league coverage than StatsBomb's 597 matches,
though StatsBomb's per-touch x/y event data is more granular where it exists at all. ESPN's
`eventId`/`teamId` namespace does not line up with pitch-edge's `match_id`/team-name keys, so
these tables are kept separate rather than merged into `matches` — joining them in is a follow-up
feature-engineering step (team-name resolution + eventId->match_id mapping), not part of loading.

Uses DuckDB's native `read_csv_auto` (with `union_by_name=true` since per-competition files don't
all share identical columns) instead of a pandas round-trip — orders of magnitude faster for this
volume of data, and it's the same warehouse the rest of the project already queries.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from pitch_edge.data.storage import Warehouse

logger = logging.getLogger(__name__)

# base_data/*.csv: one file each, small reference/dimension tables.
BASE_DATA_FILES: dict[str, str] = {
    "espn_fixtures": "fixtures.csv",
    "espn_key_event_types": "keyEventDescription.csv",
    "espn_leagues": "leagues.csv",
    "espn_players": "players.csv",
    "espn_standings": "standings.csv",
    "espn_status": "status.csv",
    "espn_team_roster": "teamRoster.csv",
    "espn_team_stats": "teamStats.csv",
    "espn_teams": "teams.csv",
    "espn_venues": "venues.csv",
}

# <dir>/*.csv: one file per competition-season, unioned into a single table.
GLOB_DATA_DIRS: dict[str, str] = {
    "espn_player_stats": "playerStats_data",
    "espn_lineups": "lineup_data",
    "espn_key_events": "keyEvents_data",
    "espn_plays": "plays_data",
    "espn_commentary": "commentary_data",
}


def _quote(path: str) -> str:
    return "'" + path.replace("'", "''") + "'"


# The subset actually kept: `espn_lineups` (formations/subs) and `espn_team_stats`
# (possession%/tackle%/cross% — a materially richer stat line than football-data.co.uk's
# shots/corners/fouls alone) have no equivalent elsewhere. `espn_key_events` is kept too — it
# feeds `rag/documents.py::espn_documents`'s match-report narration (goal/card key events),
# confirmed by grep before cutting anything, not assumed. `espn_fixtures`/`espn_teams`/
# `espn_leagues` are kept purely as the join keys `build_espn_mapping` needs to resolve
# `eventId`/`teamId` onto real matches/teams — without them none of the above can be joined to
# anything. Dropped as confirmed-unused: `espn_commentary`/`espn_plays` (2.4M/2.8M rows —
# `rag/documents.py` explicitly says it never embeds these), `espn_player_stats`, `espn_standings`,
# `espn_status`, `espn_team_roster`, `espn_venues`, `espn_key_event_types`, `espn_players`.
DEFAULT_TABLES = (
    "espn_fixtures",
    "espn_teams",
    "espn_leagues",
    "espn_team_stats",
    "espn_lineups",
    "espn_key_events",
)


def load_espn_soccer_data(
    wh: Warehouse, archive_dir: str | Path, tables: tuple[str, ...] | None = DEFAULT_TABLES
) -> dict[str, int]:
    """Load `espn-soccer-data` CSV files into their own DuckDB tables. `tables=None` loads every
    table (~1,400 files); the default loads only `DEFAULT_TABLES` (lineups/team-stats plus the
    join keys they need). Raises if `archive_dir` doesn't exist — callers that want a soft no-op
    when the archive isn't present should check first (see `ingest_espn_soccer_data`)."""
    base = Path(archive_dir)
    counts: dict[str, int] = {}
    wanted = set(tables) if tables is not None else None

    for table, fname in BASE_DATA_FILES.items():
        if wanted is not None and table not in wanted:
            continue
        path = base / "base_data" / fname
        if not path.exists():
            logger.warning("espn_soccer_data: missing %s, skipping table %s", path, table)
            continue
        wh._conn.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM read_csv_auto({_quote(str(path))}, union_by_name=true)")
        counts[table] = wh.count(table)

    for table, dirname in GLOB_DATA_DIRS.items():
        if wanted is not None and table not in wanted:
            continue
        dir_path = base / dirname
        if not dir_path.exists():
            logger.warning("espn_soccer_data: missing dir %s, skipping table %s", dir_path, table)
            continue
        pattern = _quote(str(dir_path / "*.csv"))
        wh._conn.execute(
            f"CREATE OR REPLACE TABLE {table} AS "
            f"SELECT *, regexp_extract(filename, '([^/]+)\\.csv$', 1) AS source_file "
            f"FROM read_csv_auto({pattern}, union_by_name=true, filename=true)"
        )
        counts[table] = wh.count(table)

    return counts


# league_code (matches LEAGUE_CODES in football_data_co_uk.py) -> the exact ESPN leagueName.
# Only the leagues football-data.co.uk already covers are mapped; ESPN's other ~200 competitions
# stay unmapped (NULL league_code in espn_fixtures_mapped) until there's a reason to widen this.
ESPN_LEAGUE_NAME_TO_CODE: dict[str, str] = {
    "English Premier League": "E0",
    "English League Championship": "E1",
    "English League One": "E2",
    "English League Two": "E3",
    "English National League": "EC",
    "Scottish Premiership": "SC0",
    "Scottish Championship": "SC1",
    "Scottish League One": "SC2",
    "Scottish League Two": "SC3",
    "German Bundesliga": "D1",
    "German 2. Bundesliga": "D2",
    "Spanish LALIGA": "SP1",
    "Spanish LALIGA 2": "SP2",
    "Italian Serie A": "I1",
    "Italian Serie B": "I2",
    "French Ligue 1": "F1",
    "French Ligue 2": "F2",
    "Dutch Eredivisie": "N1",
    "Belgian Pro League": "B1",
    "Portuguese Primeira Liga": "P1",
    "Turkish Super Lig": "T1",
    "Greek Super League": "G1",
}


def build_espn_mapping(wh: Warehouse) -> dict[str, int]:
    """Resolve ESPN's teamId/leagueId namespace onto pitch-edge's canonical team names and
    league_code, so espn_* tables become joinable against `matches`/the feature store by plain
    SQL. Writes two small lookup tables — `espn_team_map` (teamId -> canonical_team, via the same
    `TeamNameResolver` every other source already uses to agree on team spelling) and
    `espn_league_map` (leagueId -> league_code, only for leagues football-data.co.uk covers) —
    then a `espn_fixtures_mapped` view carrying both plus a best-effort `matched_match_id` (joined
    by team names + date, +/-1 day, to absorb timezone/kickoff-time differences between sources)
    back onto the existing `matches` table. Both lookup tables are fully rebuilt each call
    (`replace`, not `upsert`) since they're derived, not accumulated, data."""
    from pitch_edge.data.teams import TeamNameResolver

    if not wh.table_exists("matches") or not wh.table_exists("espn_teams"):
        logger.info("espn_mapping: matches or espn_teams not loaded yet — skipping")
        return {}

    canonical = set(wh.query("SELECT home_team AS t FROM matches UNION SELECT away_team AS t FROM matches")["t"])
    resolver = TeamNameResolver(canonical)
    teams = wh.query("SELECT teamId, displayName, location, name FROM espn_teams")

    def _resolve(row: pd.Series) -> str | None:
        for candidate in (row["displayName"], f"{row['location']} {row['name']}".strip()):
            if candidate:
                hit = resolver.resolve(str(candidate))
                if hit:
                    return hit
        return None

    teams["canonical_team"] = teams.apply(_resolve, axis=1)
    wh.replace("espn_team_map", teams[["teamId", "displayName", "canonical_team"]])

    league_rows: list[dict] = []
    for name, code in ESPN_LEAGUE_NAME_TO_CODE.items():
        ids = wh.query("SELECT DISTINCT leagueId FROM espn_leagues WHERE leagueName = ?", [name])["leagueId"]
        league_rows.extend({"leagueId": lid, "leagueName": name, "league_code": code} for lid in ids)
    wh.replace("espn_league_map", pd.DataFrame(league_rows))

    wh._conn.execute(
        """
        CREATE OR REPLACE VIEW espn_fixtures_mapped AS
        SELECT
            f.*,
            htm.canonical_team AS home_team,
            atm.canonical_team AS away_team,
            lm.league_code,
            m.match_id AS matched_match_id
        FROM espn_fixtures f
        LEFT JOIN espn_team_map htm ON htm.teamId = f.homeTeamId
        LEFT JOIN espn_team_map atm ON atm.teamId = f.awayTeamId
        LEFT JOIN espn_league_map lm ON lm.leagueId = f.leagueId
        LEFT JOIN matches m
            ON m.home_team = htm.canonical_team AND m.away_team = atm.canonical_team
           AND ABS(DATE_DIFF('day', CAST(m.date AS DATE), CAST(f.date AS DATE))) <= 1
        """
    )
    n_teams_resolved = int(teams["canonical_team"].notna().sum())
    n_matched = int(
        wh.query("SELECT COUNT(*) n FROM espn_fixtures_mapped WHERE matched_match_id IS NOT NULL")["n"].iloc[0]
    )
    return {
        "espn_team_map": n_teams_resolved,
        "espn_league_map": len(league_rows),
        "espn_fixtures_matched": n_matched,
    }
