"""Pull the H2 slices of fd_matches into data/research/soft-open-vs-betfair-late-season-stakes/ (git-ignored).

Writes `train.parquet` (2012-07-01..2024-06-30: ids, goals, Pinnacle and max opens, Pinnacle close) and `eval.parquet`
(2025-07-01..2026-10-19: the same with Betfair Exchange) plus `snapshot.json` with the lake build (`Lake.version`).
The bet count reads `train.parquet` and only its id, goal and open columns.
Run: arch -arm64 uv run python research/studies/soft-open-vs-betfair-late-season-stakes/pull.py
"""

from __future__ import annotations

import late_stakes as H
import pandas as pd
import sobf

if __name__ == "__main__":
    meta = sobf.pull_parquets(H.pull_queries(), H.OUT)
    for stem, ref in (("train", "pinnacle"), ("eval", "betfair_ex")):
        have = set(pd.read_parquet(H.OUT / f"{stem}.parquet").columns)
        missing = sorted(set(H.pull_columns(ref)) - have)
        print(stem, "columns missing:", missing or "none")
        assert not missing, missing
    print(sobf.snapshot_label(H.OUT))
