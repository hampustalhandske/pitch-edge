"""Snapshot of the feature frame the walk-forward backtester consumes, read from the warehouse read-only.

Writes data/research/r7/features.parquet (one row per match: spine, early/closing Pinnacle odds, features) and prints
only coverage counts (no outcome-bearing metric).
"""

from __future__ import annotations

import time
from pathlib import Path

from pitch_edge.data.storage import Warehouse
from pitch_edge.pipeline import load_feature_frame

OUT = Path(__file__).resolve().parents[3] / "data" / "research" / "r7"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    wh = Warehouse(read_only=True)
    f = load_feature_frame(wh, leagues=None, min_date="2015-07-01")
    f.to_parquet(OUT / "features.parquet")
    print("rows", len(f), "cols", f.shape[1], "secs", round(time.time() - t0))


if __name__ == "__main__":
    main()
