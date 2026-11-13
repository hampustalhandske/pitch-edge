"""r4-bsky-mention-skew-v1: Bluesky abnormal-mention skew vs the direction of the Polymarket move. Scores every registered grid point once through the research harness.

Usage: arch -arm64 uv run python research/studies/r4-bsky-mention-skew-v1/run.py   (R5_DRY=1 for a dry run that writes nothing)
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
COMMON = HERE.parent / "r4-common"
sys.path.insert(0, str(COMMON))

from runner import run  # noqa: E402

CODE = [
    HERE / "run.py",
    *(
        COMMON / f
        for f in ("runner.py", "model.py", "panel.py", "linking.py", "lexicon.py", "pull_data.py", "pull_meta.py")
    ),
]

if __name__ == "__main__":
    run("r4-bsky-mention-skew-v1", "mention", CODE)
