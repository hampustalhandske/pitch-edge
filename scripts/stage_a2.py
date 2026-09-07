"""Follow-up to stage A: artifacts, RAG index, ablation, and a first signal batch to the approval gate."""

import logging
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

from pitch_edge.ablation import run_ablation  # noqa: E402
from pitch_edge.artifacts import build_all_artifacts  # noqa: E402
from pitch_edge.backtest.engine import WalkForwardConfig  # noqa: E402
from pitch_edge.config import get_settings  # noqa: E402
from pitch_edge.data.storage import Warehouse  # noqa: E402
from pitch_edge.models import GBDTMatchModel  # noqa: E402
from pitch_edge.pipeline import build_rag_index, build_signal_pipeline  # noqa: E402

s = get_settings()
s.ensure_dirs()
t0 = time.time()
with Warehouse(s.db_path) as wh:
    f = wh.read("features")
    print("features", f.shape, flush=True)
    print("artifacts", build_all_artifacts(wh, f), round(time.time() - t0), "s", flush=True)
    print("rag docs", build_rag_index(wh).count(), round(time.time() - t0), "s", flush=True)
    table, _ = run_ablation(f, config=WalkForwardConfig(retrain_every_days=90), n_estimators=200)
    table.to_csv(s.reports_dir / "ablation.csv", index=False)
    print(table.round(4).to_string(), flush=True)
    model = GBDTMatchModel().fit(f)
    pipe = build_signal_pipeline(wh, f, model)
    thread_id, state = pipe.run_to_gate()
    import json

    (s.artifacts_dir / "pending_signals.json").write_text(
        json.dumps({"thread_id": thread_id, "proposals": state.get("proposals", [])}, default=str, indent=2)
    )
    print("signals", len(state.get("proposals", [])), "proposals at gate; log:", state.get("log", [])[-3:], flush=True)
print("STAGE A2 COMPLETE", round(time.time() - t0), "s", flush=True)
