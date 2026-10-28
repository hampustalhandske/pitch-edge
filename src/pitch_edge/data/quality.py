"""Warehouse data-quality checks that read `matches` and report problems without changing anything."""

from __future__ import annotations

import itertools

import pandas as pd

from pitch_edge.data.storage import Warehouse
from pitch_edge.paper.fixtures import ambiguous_tokens, canonical_team, fixture_score

DUPLICATE_COLUMNS = [
    "date",
    "league_code",
    "match_id_a",
    "source_a",
    "home_a",
    "away_a",
    "match_id_b",
    "source_b",
    "home_b",
    "away_b",
]


def find_duplicate_fixtures(wh: Warehouse) -> pd.DataFrame:
    """Pairs of `matches` rows that are the same fixture stored twice.

    Two rows are a pair when they share the calendar date and `league_code` and their home and away teams
    are the same clubs under different spellings (`paper.fixtures.fixture_score`: team names are canonicalised,
    then both clubs must be similar and at least one close). Returns one row per pair with the columns in
    `DUPLICATE_COLUMNS`, empty when the warehouse has none or no `matches` table.
    """
    if not wh.table_exists("matches"):
        return pd.DataFrame(columns=DUPLICATE_COLUMNS)
    matches = wh.query(
        "SELECT match_id, CAST(CAST(date AS DATE) AS VARCHAR) AS day, league_code, source, home_team, away_team"
        " FROM matches"
    )
    if matches.empty:
        return pd.DataFrame(columns=DUPLICATE_COLUMNS)
    matches["home_key"] = matches["home_team"].map(canonical_team)
    matches["away_key"] = matches["away_team"].map(canonical_team)
    rows: list[dict] = []
    for league_code, league in matches.groupby("league_code", dropna=False):
        ambiguous = ambiguous_tokens(set(league["home_key"]) | set(league["away_key"]))
        for day, group in league.groupby("day"):
            if len(group) < 2:
                continue
            for a, b in itertools.combinations(group.itertuples(index=False), 2):
                if fixture_score(a.home_key, a.away_key, b.home_key, b.away_key, ambiguous) is None:
                    continue
                rows.append(
                    {
                        "date": day,
                        "league_code": league_code,
                        "match_id_a": a.match_id,
                        "source_a": a.source,
                        "home_a": a.home_team,
                        "away_a": a.away_team,
                        "match_id_b": b.match_id,
                        "source_b": b.source,
                        "home_b": b.home_team,
                        "away_b": b.away_team,
                    }
                )
    return pd.DataFrame(rows, columns=DUPLICATE_COLUMNS)
