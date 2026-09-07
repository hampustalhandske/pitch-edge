"""Confirmed-lineup squad-value features — Phase 6.

CASE_STUDY.md's "what changes the answer" section names player-level information (confirmed lineups,
injuries) as the single biggest thing the closing price has that the aggregate spine does not. The
Phase-5 `rotation_load` group tested *minutes played* (fatigue) and was rejected; it never looked at
*who* was missing or how good the confirmed XI actually is. This module builds two genuinely different
signals from the same open Transfermarkt extract used by `features/context.py`:

* **Starting-XI value** (`sv_home_xi_value`, `sv_away_xi_value`, `sv_xi_value_diff`) — log1p of the
  summed market value (as-of the match date, via a DuckDB `ASOF JOIN` on `tm_player_valuations`) of the
  eleven players Transfermarkt lists as `starting_lineup` for that game. This is the confirmed XI's raw
  quality, not just how many minutes the squad has played.
* **Missing-star value** (`sv_home_missing_pct`, `sv_away_missing_pct`, `sv_missing_pct_diff`) — for each
  side, the "usual XI" is the eleven players with the most total minutes for that club in the trailing
  365 days before kickoff; `missing_pct` is the value-weighted share of that usual XI who are absent from
  *today's* confirmed lineup (neither starting nor on the bench) — a direct proxy for injuries/
  suspensions/bans the aggregate spine cannot see at all, independent of rotation/fatigue.

Both features are strictly pre-match: they use the *confirmed* lineup Transfermarkt recorded for that
fixture (announced pre-kickoff, same timing a bookmaker's closing price reflects) and valuations dated
on or before the match date. Coverage is bounded by the Transfermarkt <-> spine club mapping and by
lineup availability (`tm_game_lineups`), reported by `squad_value_context` via logging, the same way
`transfermarkt_context` reports its own coverage.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from pitch_edge.data.storage import Warehouse

logger = logging.getLogger(__name__)

SQUAD_VALUE_FEATURES = [
    "sv_home_xi_value",
    "sv_away_xi_value",
    "sv_xi_value_diff",
    "sv_home_missing_pct",
    "sv_away_missing_pct",
    "sv_missing_pct_diff",
]


def squad_value_context(wh: Warehouse, tm_context: pd.DataFrame) -> pd.DataFrame:
    """`tm_context`: output of `features.context.transfermarkt_context` (needs match_id + tm_game_id).

    Returns one row per match_id with `SQUAD_VALUE_FEATURES`; empty frame if lineups/valuations are
    absent or nothing maps.
    """
    if (
        tm_context.empty
        or "tm_game_id" not in tm_context
        or not wh.table_exists("tm_game_lineups")
        or not wh.table_exists("tm_player_valuations")
    ):
        return pd.DataFrame(columns=["match_id", *SQUAD_VALUE_FEATURES])
    linked = tm_context.dropna(subset=["tm_game_id"])[["match_id", "tm_game_id"]].copy()
    linked["tm_game_id"] = linked["tm_game_id"].astype("int64")
    if linked.empty:
        return pd.DataFrame(columns=["match_id", *SQUAD_VALUE_FEATURES])

    conn = wh._conn
    conn.register("sv_linked", linked)
    try:
        conn.execute(
            """
            CREATE OR REPLACE TEMP TABLE sv_games AS
            SELECT l.match_id, CAST(g.game_id AS BIGINT) AS game_id, CAST(g.date AS DATE) AS d,
                   CAST(g.home_club_id AS BIGINT) AS home_club_id,
                   CAST(g.away_club_id AS BIGINT) AS away_club_id
            FROM sv_linked l
            JOIN tm_games g ON CAST(g.game_id AS BIGINT) = l.tm_game_id
            """
        )
        conn.execute(
            """
            CREATE OR REPLACE TEMP TABLE sv_lineups AS
            SELECT CAST(lu.game_id AS BIGINT) AS game_id, CAST(lu.club_id AS BIGINT) AS club_id,
                   CAST(lu.player_id AS BIGINT) AS player_id, lu.type
            FROM tm_game_lineups lu
            WHERE lu.game_id IN (SELECT game_id FROM sv_games)
            """
        )
        conn.execute(
            """
            CREATE OR REPLACE TEMP TABLE sv_val AS
            SELECT CAST(player_id AS BIGINT) AS player_id, CAST(date AS DATE) AS d,
                   market_value_in_eur
            FROM tm_player_valuations
            WHERE market_value_in_eur IS NOT NULL
            """
        )
        # ---- starting-XI value: as-of valuation (latest value on/before kickoff) per starter ----
        conn.execute(
            """
            CREATE OR REPLACE TEMP TABLE sv_xi_val AS
            SELECT sg.match_id, sl.club_id,
                   sum(coalesce(v.market_value_in_eur, 0)) AS xi_value,
                   count(*) AS n_starters
            FROM sv_games sg
            JOIN sv_lineups sl ON sl.game_id = sg.game_id AND sl.type = 'starting_lineup'
            ASOF LEFT JOIN sv_val v ON v.player_id = sl.player_id AND v.d <= sg.d
            GROUP BY 1, 2
            """
        )
        # ---- usual XI (trailing-365d minutes leaders) per club/game, from tm_appearances ----
        conn.execute(
            """
            CREATE OR REPLACE TEMP TABLE sv_app AS
            SELECT CAST(player_club_id AS BIGINT) AS club_id, CAST(player_id AS BIGINT) AS player_id,
                   CAST(date AS DATE) AS d, coalesce(minutes_played, 0) AS minutes_played
            FROM tm_appearances
            WHERE date >= (SELECT min(d) FROM sv_games) - INTERVAL 400 DAY
              AND date <= (SELECT max(d) FROM sv_games)
            """
        )
        usual_sql = []
        for side in ("home", "away"):
            usual_sql.append(
                f"""
                {side}_minutes AS (
                    SELECT sg.match_id, a.player_id, sum(a.minutes_played) AS mins
                    FROM sv_games sg
                    JOIN sv_app a ON a.club_id = sg.{side}_club_id
                                  AND a.d BETWEEN sg.d - INTERVAL 365 DAY AND sg.d - INTERVAL 1 DAY
                    GROUP BY 1, 2
                ),
                {side}_ranked AS (
                    SELECT match_id, player_id, mins,
                           row_number() OVER (PARTITION BY match_id ORDER BY mins DESC) AS rk
                    FROM {side}_minutes
                ),
                {side}_usual AS (
                    SELECT match_id, player_id FROM {side}_ranked WHERE rk <= 11
                ),
                {side}_usual_val AS (
                    SELECT u.match_id, u.player_id, v.market_value_in_eur AS val
                    FROM {side}_usual u
                    JOIN sv_games sg ON sg.match_id = u.match_id
                    ASOF LEFT JOIN sv_val v ON v.player_id = u.player_id AND v.d <= sg.d
                ),
                {side}_present AS (
                    SELECT sg.match_id, sl.player_id
                    FROM sv_games sg
                    JOIN sv_lineups sl ON sl.game_id = sg.game_id AND sl.club_id = sg.{side}_club_id
                ),
                {side}_missing AS (
                    SELECT uv.match_id,
                           sum(coalesce(uv.val, 0)) FILTER (WHERE p.player_id IS NULL) AS missing_value,
                           sum(coalesce(uv.val, 0)) AS usual_value
                    FROM {side}_usual_val uv
                    LEFT JOIN {side}_present p ON p.match_id = uv.match_id AND p.player_id = uv.player_id
                    GROUP BY 1
                )
                """
            )
        out = conn.execute(
            f"""
            WITH {", ".join(usual_sql)}
            SELECT sg.match_id,
                   hxi.xi_value AS sv_home_xi_value, axi.xi_value AS sv_away_xi_value,
                   hm.missing_value AS h_missing_value, hm.usual_value AS h_usual_value,
                   am.missing_value AS a_missing_value, am.usual_value AS a_usual_value
            FROM sv_games sg
            LEFT JOIN sv_xi_val hxi ON hxi.match_id = sg.match_id AND hxi.club_id = sg.home_club_id
            LEFT JOIN sv_xi_val axi ON axi.match_id = sg.match_id AND axi.club_id = sg.away_club_id
            LEFT JOIN home_missing hm ON hm.match_id = sg.match_id
            LEFT JOIN away_missing am ON am.match_id = sg.match_id
            """
        ).df()
    finally:
        conn.unregister("sv_linked")

    out = out.drop_duplicates("match_id")
    out["sv_home_xi_value"] = np.log1p(out["sv_home_xi_value"].fillna(0.0))
    out["sv_away_xi_value"] = np.log1p(out["sv_away_xi_value"].fillna(0.0))
    out["sv_xi_value_diff"] = out["sv_home_xi_value"] - out["sv_away_xi_value"]
    out["sv_home_missing_pct"] = (out["h_missing_value"] / out["h_usual_value"].replace(0, np.nan)).clip(0, 1)
    out["sv_away_missing_pct"] = (out["a_missing_value"] / out["a_usual_value"].replace(0, np.nan)).clip(0, 1)
    out["sv_missing_pct_diff"] = out["sv_away_missing_pct"].fillna(0.0) - out["sv_home_missing_pct"].fillna(0.0)
    logger.info(
        "squad value context: %d/%d matches with a starting XI, %d with a usual-XI baseline",
        len(out),
        len(tm_context),
        int(out["h_usual_value"].notna().sum()),
    )
    return out[["match_id", *SQUAD_VALUE_FEATURES]]
