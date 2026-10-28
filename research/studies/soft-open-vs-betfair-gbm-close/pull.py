"""Pull the H3 slices of fd_matches into data/research/soft-open-vs-betfair-gbm-close/ (git-ignored).

Writes `train.parquet` (2012-07-01..2024-06-30: soft opens and the Pinnacle close, the train target) and
`eval.parquet` (2025-07-01..2026-10-19: soft opens, Betfair close, ft_result) plus `snapshot.json` with the lake build
(`Lake.version`). The two files share no sharp-book column.
Run: arch -arm64 uv run python research/studies/soft-open-vs-betfair-gbm-close/pull.py
"""

from __future__ import annotations

import gbm_close as H
import sobf

if __name__ == "__main__":
    meta = sobf.pull_parquets(H.pull_queries(), H.OUT)
    print(sobf.snapshot_label(H.OUT))
