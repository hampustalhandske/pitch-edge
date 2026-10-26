"""Stage-1 pull for R4 (crowd attention): metadata only, no prices and no comment bodies, into data/research/r4/.

Writes:
  pm_1x2.parquet          every Polymarket win_home/draw/win_away map row (all dates) joined to its market, with the
                          club-registry ids of both teams (`home_club_id`, `away_club_id`)
  reddit_posts.parquet    every r/soccer thread in the lake (title, created_utc, num_comments as retrieved); `score`,
                          `upvote_ratio` and `num_comments` are as-retrieved and are never used as features
  bluesky_queries.parquet the Bluesky search queries (club / competition) and the club ids they stand for
  bluesky_windows.parquet the weekly search windows per query (hits, posts read, completeness status)
  club_alias.parquet      known names per club_id (the lake's club registry), used to link Reddit titles to clubs
  dim_club.parquet        canonical club names
  snapshot_meta.json      pull time and row counts

Polymarket `winning_outcome_label`/`resolved_at` are not pulled: R4 never needs the result.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "r1-common"))

from lake import REPO, sql  # noqa: E402

OUT = REPO / "data" / "research" / "r4"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    snap: dict = {"pulled_at": datetime.now(UTC).isoformat()}
    pulls = {
        "pm_1x2": """select m.condition_id, m.event_slug, m.league_code, m.match_date, m.home_name, m.away_name,
                            m.home_club_id, m.away_club_id, m.match_id, m.orientation, m.outcome_side, m.market_type,
                            m.match_method, m.fd_match_key, k.question, k.neg_risk, k.opens_at, k.close_at,
                            k.source as pm_source, k.event_id, k.event_title
                     from pm_soccer_match_map m join pm_soccer_markets k using (condition_id)
                     where m.market_type in ('win_home', 'draw', 'win_away')""",
        "reddit_posts": """select post_id, subreddit, created_utc, retrieved_on, title, selftext, num_comments, url, meta,
                                  author_hash from reddit_posts""",
        "bluesky_queries": 'select * exclude (dt, "table") from bluesky_queries',
        "bluesky_windows": 'select * exclude (dt, "table") from bluesky_query_windows',
        "club_alias": "select * from club_alias",
        "dim_club": "select * from dim_club",
    }
    for name, q in pulls.items():
        df = sql(q)
        df.to_parquet(OUT / f"{name}.parquet")
        snap[name] = len(df)
        print(name, len(df), flush=True)
    (OUT / "snapshot_meta.json").write_text(json.dumps(snap, indent=2))
    print(snap)


if __name__ == "__main__":
    main()
