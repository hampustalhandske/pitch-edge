"""Stage-1 pull for R5: market metadata only (no prices), into data/research/r5/.

Writes:
  kalshi_markets.parquet  every Kalshi soccer market whose series ticker ends in GAME (1X2 match markets)
  pm_1x2.parquet          every Polymarket win_home/draw/win_away map row from 2025-04-01, joined to its market
  fd.parquet              football-data matches from 2025-04-01 (odds open/close, kickoff, result; no odds_all)
  kalshi_series_fees.json fee_type / fee_multiplier per Kalshi GAME series from Kalshi's public series API
  snapshot_meta.json      pull time and row counts

Kalshi `result` and Polymarket `winning_outcome_label`/`resolved_at` are pulled but are hindsight: the
studies use them only to settle paper positions, never to select or map markets.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "r1-common"))

from lake import REPO, sql  # noqa: E402

from pitch_edge.data.http import CachedHttpClient  # noqa: E402

OUT = REPO / "data" / "research" / "r5"
SINCE = "2025-04-01"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    snap: dict = {"pulled_at": datetime.now(UTC).isoformat()}
    km = sql(
        """select ticker, series_ticker, event_ticker, title, yes_sub_title, no_sub_title, rules_primary, open_time,
                  close_time, occurrence_datetime, expiration_time, status, result, volume, open_interest,
                  last_price, custom_strike, source, scraped_at
           from kalshi_markets where category = 'soccer' and series_ticker like '%GAME'"""
    )
    km.to_parquet(OUT / "kalshi_markets.parquet")
    snap["kalshi_markets"] = len(km)
    pm = sql(
        f"""select m.condition_id, m.event_slug, m.league_code, m.match_date, m.home_code, m.away_code, m.home_name,
                   m.away_name, m.match_id, m.orientation, m.outcome_side, m.market_type, m.match_method,
                   m.fd_match_key, k.question, k.outcomes, k.neg_risk, k.opens_at, k.close_at, k.resolved_at,
                   k.winning_outcome_label, k.source as pm_source, k.event_id, k.event_title
            from pm_soccer_match_map m join pm_soccer_markets k using (condition_id)
            where m.market_type in ('win_home', 'draw', 'win_away') and m.match_date >= '{SINCE}'"""
    )
    pm.to_parquet(OUT / "pm_1x2.parquet")
    snap["pm_1x2"] = len(pm)
    fd = sql(
        f"""select f.* exclude (odds_all), f.division || '|' || strftime(f.match_date, '%Y-%m-%d') || '|' || f.home_team
                   || '|' || f.away_team as fd_match_key
            from fd_matches f where f.match_date >= '{SINCE}'"""
    )
    fd.to_parquet(OUT / "fd.parquet")
    snap["fd"] = len(fd)
    client = CachedHttpClient(OUT / "http_cache", min_interval_s=0.5)
    fees = {}
    for s in sorted(km.series_ticker.unique()):
        d = client.get_json(f"https://api.elections.kalshi.com/trade-api/v2/series/{s}")
        ser = d.get("series", {})
        fees[s] = {"fee_type": ser.get("fee_type"), "fee_multiplier": ser.get("fee_multiplier")}
    (OUT / "kalshi_series_fees.json").write_text(json.dumps(fees, indent=2, sort_keys=True))
    snap["kalshi_series"] = len(fees)
    (OUT / "snapshot_meta.json").write_text(json.dumps(snap, indent=2))
    print(snap)


if __name__ == "__main__":
    main()
