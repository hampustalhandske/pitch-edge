"""Per-match context from the open Transfermarkt extract: rotation load and referee names.

Two things the aggregate spine cannot see, both strictly pre-match:

* **Rotation / fatigue load** — the spine only knows league fixtures, so a midweek cup tie or a
  European away trip is invisible to `rest_and_congestion`. Transfermarkt's `games` table carries
  every first-team fixture (league, domestic cup, Europe, super cups) and `appearances` carries
  minutes per player per game, so for each side we compute: squad minutes in the trailing 7 days
  (per starter-equivalent), days since the club's last match in *any* competition, and whether a
  non-league fixture was played in the 2-4 days before kickoff. Reserve/U21 fixtures are **not**
  in any free feed we use; that gap is stated in the README rather than papered over.
* **Referee** — the fallback spine has no referee column, so the referee feature group tested as a
  no-op in the first case study. Transfermarkt games carry the referee for ~70k matches since
  2015; joined by (date, home club, away club) it re-activates `referee_features` (cards per game,
  home card share) for the European divisions.

Everything runs as DuckDB SQL inside the warehouse (range joins on 1.9 M appearance rows).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from pitch_edge.data.storage import Warehouse
from pitch_edge.data.teams import TeamNameResolver

logger = logging.getLogger(__name__)

ROTATION_FEATURES = [
    "rot_home_minutes_7d",
    "rot_away_minutes_7d",
    "rot_home_days_since_any",
    "rot_away_days_since_any",
    "rot_home_midweek_cup",
    "rot_away_midweek_cup",
    "rot_load_diff",
]
CONTEXT_COLUMNS = [*ROTATION_FEATURES, "tm_referee", "tm_game_id"]


def club_id_map(wh: Warehouse, teams: list[str]) -> dict[str, int]:
    """Spine team name -> Transfermarkt club_id (resolver over club names; latest-season club wins ties)."""
    if not wh.table_exists("tm_clubs"):
        return {}
    clubs = wh.query("SELECT club_id, name, last_season FROM tm_clubs")
    if clubs.empty:
        return {}
    resolver = TeamNameResolver(teams)
    clubs["canonical"] = clubs["name"].map(lambda n: resolver.resolve(str(n)))
    clubs = clubs.dropna(subset=["canonical"])
    clubs["last_season"] = pd.to_numeric(clubs["last_season"], errors="coerce").fillna(0)
    clubs = clubs.sort_values("last_season", ascending=False).drop_duplicates("canonical")
    return {str(c): int(i) for c, i in zip(clubs["canonical"], clubs["club_id"], strict=True)}


def transfermarkt_context(wh: Warehouse, matches: pd.DataFrame) -> pd.DataFrame:
    """One row per match_id with ROTATION_FEATURES + tm_referee; empty frame if the tables are absent."""
    if matches.empty or not (wh.table_exists("tm_games") and wh.table_exists("tm_appearances")):
        return pd.DataFrame(columns=["match_id", *CONTEXT_COLUMNS])
    teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
    cmap = club_id_map(wh, teams)
    if not cmap:
        return pd.DataFrame(columns=["match_id", *CONTEXT_COLUMNS])
    ctx = pd.DataFrame(
        {
            "match_id": matches["match_id"].astype(str),
            "d": pd.to_datetime(matches["date"]).dt.normalize(),
            "home_club_id": matches["home_team"].map(cmap),
            "away_club_id": matches["away_team"].map(cmap),
        }
    )
    ctx = ctx.dropna(subset=["home_club_id", "away_club_id"]).astype({"home_club_id": int, "away_club_id": int})
    if ctx.empty:
        return pd.DataFrame(columns=["match_id", *CONTEXT_COLUMNS])
    conn = wh._conn
    conn.register("ctx_matches", ctx)
    try:
        conn.execute(
            """
            CREATE OR REPLACE TEMP TABLE ctx_games AS
            SELECT CAST(date AS DATE) AS d, game_id, CAST(home_club_id AS BIGINT) AS home_club_id,
                   CAST(away_club_id AS BIGINT) AS away_club_id, referee,
                   coalesce(competition_type, 'other') AS competition_type
            FROM tm_games
            WHERE date >= (SELECT min(d) FROM ctx_matches) - INTERVAL 40 DAY
              AND date <= (SELECT max(d) FROM ctx_matches)
            """
        )
        conn.execute(
            """
            CREATE OR REPLACE TEMP TABLE ctx_games_long AS
            SELECT d, home_club_id AS club_id, competition_type FROM ctx_games
            UNION ALL SELECT d, away_club_id, competition_type FROM ctx_games
            """
        )
        conn.execute(
            """
            CREATE OR REPLACE TEMP TABLE ctx_app AS
            SELECT CAST(player_club_id AS BIGINT) AS club_id, CAST(date AS DATE) AS d,
                   coalesce(minutes_played, 0) AS minutes_played
            FROM tm_appearances
            WHERE date >= (SELECT min(d) FROM ctx_matches) - INTERVAL 10 DAY
              AND date <= (SELECT max(d) FROM ctx_matches)
            """
        )
        side_sql = []
        for side in ("home", "away"):
            side_sql.append(
                f"""
                {side}_min AS (
                    SELECT m.match_id, sum(a.minutes_played) / 11.0 AS rot_{side}_minutes_7d
                    FROM ctx_matches m JOIN ctx_app a
                      ON a.club_id = m.{side}_club_id AND a.d BETWEEN m.d - INTERVAL 7 DAY AND m.d - INTERVAL 1 DAY
                    GROUP BY 1
                ),
                {side}_last AS (
                    SELECT m.match_id,
                           date_diff('day', max(g.d), m.d) AS rot_{side}_days_since_any,
                           count(*) FILTER (WHERE g.competition_type <> 'domestic_league'
                                              AND g.d BETWEEN m.d - INTERVAL 4 DAY AND m.d - INTERVAL 2 DAY)
                               AS rot_{side}_midweek_cup
                    FROM ctx_matches m JOIN ctx_games_long g
                      ON g.club_id = m.{side}_club_id AND g.d BETWEEN m.d - INTERVAL 30 DAY AND m.d - INTERVAL 1 DAY
                    GROUP BY m.match_id, m.d
                )"""
            )
        out = conn.execute(
            f"""
            WITH {", ".join(side_sql)},
            ref AS (
                SELECT m.match_id, any_value(g.referee) AS tm_referee, any_value(g.game_id) AS tm_game_id
                FROM ctx_matches m JOIN ctx_games g
                  ON g.home_club_id = m.home_club_id AND g.away_club_id = m.away_club_id AND g.d = m.d
                GROUP BY 1
            )
            SELECT m.match_id,
                   hm.rot_home_minutes_7d, am.rot_away_minutes_7d,
                   hl.rot_home_days_since_any, al.rot_away_days_since_any,
                   hl.rot_home_midweek_cup, al.rot_away_midweek_cup,
                   ref.tm_referee, ref.tm_game_id
            FROM ctx_matches m
            LEFT JOIN home_min hm USING (match_id) LEFT JOIN away_min am USING (match_id)
            LEFT JOIN home_last hl USING (match_id) LEFT JOIN away_last al USING (match_id)
            LEFT JOIN ref USING (match_id)
            """
        ).df()
    finally:
        conn.unregister("ctx_matches")
    for c in ("rot_home_minutes_7d", "rot_away_minutes_7d"):
        out[c] = out[c].fillna(0.0).astype(float)
    for c in ("rot_home_midweek_cup", "rot_away_midweek_cup"):
        out[c] = out[c].fillna(0).astype(float)
    out["rot_load_diff"] = out["rot_home_minutes_7d"] - out["rot_away_minutes_7d"]
    out["tm_referee"] = out["tm_referee"].where(out["tm_referee"].notna() & (out["tm_referee"] != ""), np.nan)
    logger.info(
        "transfermarkt context: %d/%d matches mapped, %d with referee",
        len(out),
        len(matches),
        int(out["tm_referee"].notna().sum()),
    )
    return out[["match_id", *CONTEXT_COLUMNS]]


def load_context(wh: Warehouse, matches: pd.DataFrame) -> pd.DataFrame | None:
    """Everything `FeatureBuilder.build(context=...)` accepts: Transfermarkt rotation/referee context
    plus Wikipedia attention anomalies when the `wiki_pageviews` table exists. None if nothing applies."""
    from pitch_edge.data.alt.wikipedia_attention import attention_features

    frames = []
    tm = transfermarkt_context(wh, matches)
    if not tm.empty:
        frames.append(tm)
    if wh.table_exists("wiki_pageviews"):
        pv = wh.read("wiki_pageviews")
        if not pv.empty:
            att = attention_features(matches, pv)
            if not att.empty:
                frames.append(att)
    if not frames:
        return None
    ctx = frames[0]
    for f in frames[1:]:
        ctx = ctx.merge(f, on="match_id", how="outer")
    return ctx
