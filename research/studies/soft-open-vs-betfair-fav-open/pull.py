"""Pull the H1 slices of fd_matches into data/research/soft-open-vs-betfair-fav-open/ (git-ignored).

Writes `train.parquet` (exploratory window) and `eval.parquet` (evaluation window) plus `snapshot.json`, which
records the lake build (`Lake.version`) the pull was served from. Selection reads `train.parquet` only.
Run: arch -arm64 uv run python research/studies/soft-open-vs-betfair-fav-open/pull.py
"""

from __future__ import annotations

import fav_open as H
import sobf

if __name__ == "__main__":
    meta = sobf.pull_parquets(H.pull_queries(), H.OUT)
    print(sobf.snapshot_label(H.OUT))
