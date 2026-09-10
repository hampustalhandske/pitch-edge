"""Confirmed-lineup features for *upcoming* fixtures (live signals path).

`features/squad_value.py` computes starting-XI value and missing-star percentage for
**historical** matches from Transfermarkt's after-the-fact confirmed lineups. For a fixture that
hasn't been played yet there is no Transfermarkt record — the only source of a pre-kickoff
confirmed XI is `APIFootballSource.lineups()`, and it only returns real data shortly before
kickoff (an unpublished lineup is simply an empty response, not an error).

This module resolves a fixture to an API-Football `fixture_id` (team name + date, via the same
`TeamNameResolver` used everywhere else), then:

* if API-Football has a published starting XI for it — value the confirmed starters against
  Transfermarkt's `tm_player_valuations` (fuzzy name match), compute the same value-weighted
  missing-percentage against the club's trailing-365-day "usual XI" as `squad_value.py`, and tag
  the row `lineup_source="confirmed"`;
* otherwise return nothing for that fixture — the caller (`pipeline.upcoming_fixture_frame`)
  keeps its existing last-used-XI carry-forward and tags it `lineup_source="provisional"`.

No `API_FOOTBALL_KEY` set = no-op, exactly like every other optional source in this project.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from pitch_edge.data.alt.api_football import APIFootballSource
from pitch_edge.data.storage import Warehouse
from pitch_edge.data.teams import TeamNameResolver, normalise
from pitch_edge.features.context import club_id_map

logger = logging.getLogger(__name__)

LIVE_LINEUP_FEATURES = [
    "sv_home_xi_value",
    "sv_away_xi_value",
    "sv_xi_value_diff",
    "sv_home_missing_pct",
    "sv_away_missing_pct",
    "sv_missing_pct_diff",
]


def resolve_fixture_ids(
    src: APIFootballSource, fixtures: pd.DataFrame, league_code: str, season: int
) -> dict[str, int]:
    """match_id -> API-Football fixture_id, matched by (home_team, away_team, date)."""
    if fixtures.empty or not src.enabled:
        return {}
    live_fx = src.fixtures(league_code, season)
    if live_fx.empty:
        return {}
    teams = sorted(set(fixtures["home_team"]) | set(fixtures["away_team"]))
    resolver = TeamNameResolver(teams)
    live_fx = live_fx.assign(
        home_team=live_fx["home_team"].map(lambda n: resolver.resolve(str(n)) or n),
        away_team=live_fx["away_team"].map(lambda n: resolver.resolve(str(n)) or n),
        date_only=pd.to_datetime(live_fx["date"], utc=True, errors="coerce").dt.date,
    )
    out: dict[str, int] = {}
    for _, fx in fixtures.iterrows():
        fx_date = pd.Timestamp(fx["date"]).date()
        rows = live_fx[
            (live_fx["home_team"] == fx["home_team"])
            & (live_fx["away_team"] == fx["away_team"])
            & (live_fx["date_only"] == fx_date)
        ]
        if not rows.empty and pd.notna(rows["fixture_id"].iloc[0]):
            out[fx["match_id"]] = int(rows["fixture_id"].iloc[0])
    return out


def _usual_xi_values(wh: Warehouse, club_id: int, asof_date, lookback_days: int = 365) -> pd.DataFrame:
    """Trailing-`lookback_days` minutes leaders (top 11) for a club, with their latest known value."""
    if not (wh.table_exists("tm_appearances") and wh.table_exists("tm_player_valuations")):
        return pd.DataFrame(columns=["player_id", "player_name", "value"])
    return wh.query(
        """
        WITH app AS (
            SELECT CAST(player_id AS BIGINT) AS player_id, sum(coalesce(minutes_played, 0)) AS mins
            FROM tm_appearances
            WHERE CAST(player_club_id AS BIGINT) = ?
              AND date BETWEEN CAST(? AS DATE) - INTERVAL '365' DAY AND CAST(? AS DATE) - INTERVAL '1' DAY
            GROUP BY 1
        ),
        ranked AS (
            SELECT player_id, mins, row_number() OVER (ORDER BY mins DESC) AS rk FROM app
        ),
        val AS (
            SELECT CAST(player_id AS BIGINT) AS player_id, market_value_in_eur, date
            FROM tm_player_valuations
            WHERE market_value_in_eur IS NOT NULL AND date <= CAST(? AS DATE)
        )
        SELECT r.player_id, p.name AS player_name,
               (SELECT v.market_value_in_eur FROM val v WHERE v.player_id = r.player_id
                ORDER BY v.date DESC LIMIT 1) AS value
        FROM ranked r
        LEFT JOIN tm_players p ON CAST(p.player_id AS BIGINT) = r.player_id
        WHERE r.rk <= 11
        """,
        params=[club_id, str(asof_date), str(asof_date), str(asof_date)],
    )


def _latest_value(wh: Warehouse, player_name: str, asof_date) -> float:
    if not (wh.table_exists("tm_players") and wh.table_exists("tm_player_valuations")):
        return 0.0
    rows = wh.query(
        """
        SELECT p.player_id, p.name FROM tm_players p WHERE lower(p.name) = lower(?)
        """,
        params=[player_name],
    )
    if rows.empty:
        return 0.0
    player_id = int(rows["player_id"].iloc[0])
    val = wh.query(
        """
        SELECT market_value_in_eur FROM tm_player_valuations
        WHERE CAST(player_id AS BIGINT) = ? AND date <= CAST(? AS DATE) AND market_value_in_eur IS NOT NULL
        ORDER BY date DESC LIMIT 1
        """,
        params=[player_id, str(asof_date)],
    )
    return float(val["market_value_in_eur"].iloc[0]) if not val.empty else 0.0


def confirmed_lineup_features(
    wh: Warehouse, fixtures: pd.DataFrame, api_src: APIFootballSource | None = None
) -> pd.DataFrame:
    """One row per match_id with `LIVE_LINEUP_FEATURES` + `lineup_source="confirmed"` — only for
    fixtures where API-Football has already published a starting XI. Empty frame if the key is
    unset, nothing resolves, or no fixture has a published lineup yet (the common case)."""
    src = api_src or APIFootballSource()
    if fixtures.empty or not src.enabled or "league_code" not in fixtures.columns:
        return pd.DataFrame(columns=["match_id", "lineup_source", *LIVE_LINEUP_FEATURES])

    season = datetime.now(UTC).year - (1 if datetime.now(UTC).month < 7 else 0)
    rows_out = []
    for league_code, grp in fixtures.groupby("league_code"):
        fixture_ids = resolve_fixture_ids(src, grp, str(league_code), season)
        if not fixture_ids:
            continue
        teams = sorted(set(grp["home_team"]) | set(grp["away_team"]))
        cmap = club_id_map(wh, teams)
        for _, fx in grp.iterrows():
            fixture_id = fixture_ids.get(fx["match_id"])
            if fixture_id is None:
                continue
            lineup = src.lineups(fixture_id)
            starters = lineup[lineup["slot"] == "start"] if not lineup.empty else lineup
            if starters.empty:
                continue  # not published yet — caller keeps the provisional carry-forward
            values: dict[str, float] = {}
            missing_pct: dict[str, float] = {}
            for side, team_name in (("home", fx["home_team"]), ("away", fx["away_team"])):
                team_starters = starters[starters["team"].map(lambda t: normalise(str(t))) == normalise(team_name)]
                if team_starters.empty:
                    values[side] = 0.0
                    missing_pct[side] = float("nan")
                    continue
                xi_value = sum(_latest_value(wh, str(p), fx["date"]) for p in team_starters["player"])
                values[side] = xi_value
                club_id = cmap.get(team_name)
                if club_id is None:
                    missing_pct[side] = float("nan")
                    continue
                usual = _usual_xi_values(wh, club_id, fx["date"])
                if usual.empty:
                    missing_pct[side] = float("nan")
                    continue
                present_names = {normalise(str(p)) for p in team_starters["player"]}
                usual = usual.assign(value=usual["value"].fillna(0.0))
                usual_total = float(usual["value"].sum())
                missing_value = float(
                    usual.loc[~usual["player_name"].fillna("").map(normalise).isin(present_names), "value"].sum()
                )
                pct = missing_value / usual_total if usual_total > 0 else float("nan")
                missing_pct[side] = pct if np.isnan(pct) else float(np.clip(pct, 0.0, 1.0))
            home_missing = missing_pct.get("home")
            away_missing = missing_pct.get("away")
            rows_out.append(
                {
                    "match_id": fx["match_id"],
                    "lineup_source": "confirmed",
                    "sv_home_xi_value": np.log1p(values.get("home", 0.0)),
                    "sv_away_xi_value": np.log1p(values.get("away", 0.0)),
                    "sv_xi_value_diff": np.log1p(values.get("home", 0.0)) - np.log1p(values.get("away", 0.0)),
                    "sv_home_missing_pct": home_missing,
                    "sv_away_missing_pct": away_missing,
                    # NaN (unresolved club/usual-XI) treated as 0 for the diff, matching
                    # squad_value.py's .fillna(0.0) — `x or 0.0` would NOT do this, since NaN is truthy.
                    "sv_missing_pct_diff": (0.0 if away_missing is None or np.isnan(away_missing) else away_missing)
                    - (0.0 if home_missing is None or np.isnan(home_missing) else home_missing),
                }
            )
    if not rows_out:
        return pd.DataFrame(columns=["match_id", "lineup_source", *LIVE_LINEUP_FEATURES])
    logger.info("confirmed lineups: %d/%d fixtures have a published starting XI", len(rows_out), len(fixtures))
    return pd.DataFrame(rows_out)
