"""Pull the fixed lake snapshot the R3 `closing-line-prediction` studies run on.

Writes to data/research/r3/ (gitignored):
- fd_all.parquet      every fd_matches row (all divisions, all dates): keys, kickoff, goals, result,
                      source_file — the results history behind pi-ratings and the fixture schedule.
- fd_odds.parquet     main-league (mmz4281) rows from 2012-07-01 with the typed open/close 1X2 columns
                      and every `odds_all` key flattened to its own float column (raw football-data names).
- clubelo.parquet     clubelo_ratings (club_slug, date, elo).
- us_team_matches.parquet, tm_manager_tenures.parquet, tm_injuries.parquet,
  tm_injuries_first_seen.parquet (min scraped_at per spell from tm_injuries_log),
  club_source_map.parquet, club_alias.parquet
- tm_games.parquet   warehouse tm_games (transfermarkt-datasets, all competitions), read-only.
- snapshot_at.txt     UTC time of the pull (the data_snapshot label recorded with every trial).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pandas as pd
from lake import REPO, q

OUT = REPO / "data" / "research" / "r3"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    fd_all = q(
        "select division, country, league, season, match_date, kickoff_time, home_team, away_team, "
        "ft_home_goals, ft_away_goals, ft_result, source_file from fd_matches"
    )
    fd_all.to_parquet(OUT / "fd_all.parquet")
    print("fd_all", fd_all.shape)

    typed = ", ".join(
        f"{b}_{o}_{s}"
        for b in ("pinnacle", "bet365", "max", "avg", "betfair_ex")
        for o in ("home", "draw", "away")
        for s in ("open", "close")
    )
    d = q(
        f"select division, country, league, season, match_date, kickoff_time, home_team, away_team, "
        f"ft_home_goals, ft_away_goals, ft_result, source_file, scraped_at, {typed}, odds_all::VARCHAR as odds_all "
        f"from fd_matches where source_file like 'mmz4281/%' and match_date >= DATE '2012-07-01'"
    )
    odds = pd.DataFrame([json.loads(s) if s else {} for s in d["odds_all"]])
    odds = odds.apply(pd.to_numeric, errors="coerce").add_prefix("oa_")
    fd_odds = pd.concat([d.drop(columns="odds_all").reset_index(drop=True), odds], axis=1)
    fd_odds.to_parquet(OUT / "fd_odds.parquet")
    print("fd_odds", fd_odds.shape)

    for name, sql in {
        "clubelo": "select club_slug, date, elo, scraped_at from clubelo_ratings",
        "us_team_matches": 'select * exclude (dt, "table") from us_team_matches',
        "tm_manager_tenures": 'select * exclude (dt, "table") from tm_manager_tenures',
        "tm_managers": "select coach_id, name from tm_managers",
        "tm_injuries": 'select * exclude (dt, "table") from tm_injuries',
        "tm_injuries_first_seen": (
            "select player_id, injury, from_date, min(scraped_at) as first_seen_at, count(*) as n_log_rows "
            "from tm_injuries_log group by 1, 2, 3"
        ),
        "club_source_map": 'select * exclude (dt, "table") from club_source_map',
        "club_alias": 'select * exclude (dt, "table") from club_alias',
        "tm_clubs": 'select * exclude (dt, "table") from tm_clubs',
    }.items():
        t = q(sql)
        t.to_parquet(OUT / f"{name}.parquet")
        print(name, t.shape)
    (OUT / "snapshot_at.txt").write_text(at + "\n")
    print("snapshot", at)


def pull_tm_games() -> None:
    """Copy the warehouse's transfermarkt-datasets `tm_games` (all competitions, 2012+) to
    data/research/r3/tm_games.parquet, opening the warehouse read-only."""
    import duckdb

    con = duckdb.connect(str(REPO / "data" / "pitch_edge.duckdb"), read_only=True)
    con.execute("SET enable_progress_bar=false")
    g = con.execute(
        "select game_id, competition_id, competition_type, season, date::varchar as date, home_club_id, "
        "away_club_id, home_club_goals, away_club_goals, home_club_name, away_club_name from tm_games"
    ).df()
    con.close()
    g.to_parquet(OUT / "tm_games.parquet")
    print("tm_games", g.shape)


if __name__ == "__main__":
    main()
    pull_tm_games()
