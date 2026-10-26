"""Pull the raw football-data rows (main mmz4281 files, 2012-07 onward) with their `odds_all` JSON from the lake.

Writes data/research/r2/audit_q3_raw.parquet with one row per match: keys, result, kickoff, scraped_at, and
every odds_all key flattened into its own float column (raw football-data column names, e.g. PSH, MaxH, BbMxH).
"""

import json
from datetime import UTC, datetime

import pandas as pd
from lake import REPO, q

OUT = REPO / "data" / "research" / "r2" / "audit_q3_raw.parquet"

sql = """
select division, season, match_date, kickoff_time, home_team, away_team, ft_result, source_file, scraped_at,
       odds_all::VARCHAR as odds_all
from fd_matches
where source_file like 'mmz4281/%' and match_date >= DATE '2012-07-01'
"""
pulled_at = datetime.now(UTC).isoformat()
d = q(sql)
odds = pd.DataFrame([json.loads(s) if s else {} for s in d["odds_all"]])
odds = odds.apply(pd.to_numeric, errors="coerce")
out = pd.concat([d.drop(columns="odds_all").reset_index(drop=True), odds], axis=1)
out["pulled_at"] = pulled_at
out.to_parquet(OUT)
print(out.shape, pulled_at)
print(sorted(odds.columns))
