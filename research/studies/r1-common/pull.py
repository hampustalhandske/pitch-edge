"""Pull the R1 slices from the lakehouse once, into data/research/r1/ as parquet.

Writes:
  map_markets.parquet  every pm_soccer_match_map 1X2 row with an fd_match_key, joined to pm_soccer_markets
  fd.parquet           the fd_matches rows those keys point to (odds open/close, result, kickoff_time)
  bars.parquet         every pm_soccer_bars_1h row for those condition_ids (both Yes and No tokens)
  snapshot.json        pull time and row counts (the data snapshot the study is scored against)
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pandas as pd
from lake import REPO, sql

CHUNK = 400

OUT = REPO / "data" / "research" / "r1"
OUT.mkdir(parents=True, exist_ok=True)

KEYED = """
  select m.*, k.question, k.outcomes, k.neg_risk, k.neg_risk_market_id, k.opens_at, k.close_at, k.resolved_at,
         k.resolution_status, k.winning_outcome_label, k.source as pm_source, k.market_detail, k.event_id,
         k.event_title, k.gamma_tagged, k.soccer_rule
  from pm_soccer_match_map m join pm_soccer_markets k using (condition_id)
  where m.fd_match_key is not null and m.market_type in ('win_home', 'draw', 'win_away')
"""


def main() -> None:
    snap = {"pulled_at": datetime.now(UTC).isoformat()}
    mm = sql(KEYED)
    mm.to_parquet(OUT / "map_markets.parquet")
    snap["map_markets"] = len(mm)
    fd = sql(f"""
      select f.* exclude (odds_all), f.division || '|' || strftime(f.match_date, '%Y-%m-%d') || '|' || f.home_team
             || '|' || f.away_team as fd_match_key
      from fd_matches f
      where f.division || '|' || strftime(f.match_date, '%Y-%m-%d') || '|' || f.home_team || '|' || f.away_team
            in (select distinct fd_match_key from ({KEYED}))
    """)
    fd.to_parquet(OUT / "fd.parquet")
    snap["fd"] = len(fd)
    ids = sorted(mm.condition_id.unique())
    parts = []
    for i in range(0, len(ids), CHUNK):
        chunk = ids[i : i + CHUNK]
        marks = ",".join("?" * len(chunk))
        parts.append(
            sql(
                f"""select asset_id, hour_start, condition_id, outcome_seq, outcome_label, open, high, low, close,
                       volume_usdc, buy_volume_usdc, n_trades, source
                from pm_soccer_bars_1h where condition_id in ({marks})""",
                chunk,
            )
        )
        print(f"bars chunk {i // CHUNK + 1}: {len(parts[-1])} rows", flush=True)
    bars = pd.concat(parts, ignore_index=True)
    bars.to_parquet(OUT / "bars.parquet")
    snap["bars"] = len(bars)
    (OUT / "snapshot.json").write_text(json.dumps(snap, indent=2))
    print(snap)


if __name__ == "__main__":
    main()
