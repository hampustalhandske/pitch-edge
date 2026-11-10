"""Driver: edgelab dataset export and submission manifest on fixed inputs.

Inputs: four synthetic binary markets (six hourly Yes-book snapshots before kickoff, a resolution, two model
forecasts each, market types cycling through win_home / draw / win_away / btts), and a research ledger built
beforehand by `drivers/research.py` in the `--inputs` folder (copied into this run's folder; its registration files
stay where they are, so every run exports the same paths). The run writes
`export_prediction_market_dataset` into `submission/data`, a fixed `submission/strategy/strategy.yaml`, then
`write_submission` for the ledger's grid hypothesis up to the ledger's last chain hash.

Output: every file of the submission folder by relative path with its sha256 and size (the byte-identity check),
plus the parquet files' rows and the text files' contents to make a difference readable, and the export report.
No field is volatile: every input, the ledger copy and the git sha in the spec are fixed.

Defect: `unseeded` adds an unseeded random jitter to the model forecasts.
"""

from __future__ import annotations

import hashlib
import shutil
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import duckdb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from canon import driver_args, finish, frame_records, write_canonical  # noqa: E402

from pitch_edge.edgelab.export import export_prediction_market_dataset  # noqa: E402
from pitch_edge.edgelab.submission import SubmissionSpec, write_submission  # noqa: E402
from pitch_edge.research.ledger import Ledger  # noqa: E402

FAMILY = "determinism-family"
HYPOTHESIS = "determinism-grid-v1"
FIRST_KICKOFF = datetime(2026, 3, 1, 15, 0, tzinfo=UTC)
MARKET_TYPES = ("win_home", "draw", "win_away", "btts")


def frames(unseeded: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    snaps, res, feats = [], [], []
    for i, mtype in enumerate(MARKET_TYPES):
        market = f"pm-{i}"
        ko = FIRST_KICKOFF + timedelta(days=i)
        for h in range(6, 0, -1):
            mid = 0.30 + 0.05 * i + 0.01 * (6 - h)
            snaps.append({"market": market, "ts": ko - timedelta(hours=h), "bid": round(mid - 0.01, 4),
                          "ask": round(mid + 0.01, 4), "size": 100.0 + 10 * h})  # fmt: skip
        res.append({"market": market, "resolved_at": ko + timedelta(hours=3), "outcome": i % 2})
        for h in (5, 2):
            ts = ko - timedelta(hours=h)
            feats.append({"market": market, "ts": ts, "known_at": ts + timedelta(minutes=10),
                          "model_p": 0.40 + 0.04 * i + 0.01 * h, "kickoff_ts": ko, "market_type": mtype})  # fmt: skip
    features = pd.DataFrame(feats)
    if unseeded:
        features["model_p"] += np.random.default_rng().normal(0, 1e-3, len(features))
    return pd.DataFrame(snaps), pd.DataFrame(res), features


def main() -> None:
    args = driver_args(paths={"inputs": "working folder of a finished drivers/research.py run"})
    work: Path = args.work
    ledger_path = work / "ledger.duckdb"
    shutil.copyfile(args.inputs / "data" / "research" / "ledger.duckdb", ledger_path)
    ledger = Ledger(ledger_path)
    con = duckdb.connect(str(ledger_path), read_only=True)
    try:
        upto = con.execute("SELECT row_hash FROM ledger_chain ORDER BY seq DESC LIMIT 1").fetchone()[0]
    finally:
        con.close()

    folder = work / "submission"
    (folder / "strategy").mkdir(parents=True)
    (folder / "strategy" / "strategy.yaml").write_text(f"name: {HYPOTHESIS}\nentry: strategy.py\n")
    snapshots, resolutions, features = frames(args.defect == "unseeded")
    report = export_prediction_market_dataset(snapshots, resolutions, features, folder / "data")
    spec = SubmissionSpec(
        hypothesis_id=HYPOTHESIS,
        git_sha="0123abc",
        family=FAMILY,
        upto=upto,
        train_end=date(2021, 12, 31),
        test_window=(date(2022, 1, 1), date(2023, 12, 31)),
        profile="fast",
    )
    write_submission(spec, folder, ledger)

    files = {}
    for path in sorted(p for p in folder.rglob("*") if p.is_file()):
        raw = path.read_bytes()
        entry: dict[str, object] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
        if path.suffix == ".parquet":
            entry["rows"] = frame_records(pd.read_parquet(path))
        else:
            entry["text"] = raw.decode("utf-8")
        files[path.relative_to(folder).as_posix()] = entry
    summary = {k: v for k, v in vars(report).items() if k != "path"}
    write_canonical(args.out, {"files": files, "export_report": summary})


if __name__ == "__main__":
    main()
    finish()
