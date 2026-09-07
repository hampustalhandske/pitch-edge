import logging
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
from pitch_edge.backtest.engine import WalkForwardConfig
from pitch_edge.config import get_settings
from pitch_edge.data.storage import Warehouse
from pitch_edge.models import DixonColesMatchModel, GBDTMatchModel
from pitch_edge.pipeline import load_feature_frame, persist_features, run_backtests

RUN_ID = time.strftime("%Y%m%dT%H%M%S")
s = get_settings()
s.ensure_dirs()
with Warehouse(s.db_path) as wh:
    for t in ("backtest_bets", "backtest_summaries", "model_predictions"):
        if wh.table_exists(t):
            wh._conn.execute(f"DELETE FROM {t}")  # discard the aborted run (no usable bet price)
    t0 = time.time()
    f = load_feature_frame(wh, min_date="2015-07-01")
    print("features", f.shape, flush=True)
    persist_features(wh, f)
    run_backtests(
        f,
        [DixonColesMatchModel(), GBDTMatchModel(), GBDTMatchModel(include_market=True)],
        WalkForwardConfig(retrain_every_days=45),
        wh=wh,
        label="main",
        run_id=RUN_ID,
    )
    (s.artifacts_dir / "run_id.txt").write_text(RUN_ID)
    print("backtests done in", round(time.time() - t0), "s", flush=True)
print("STAGE A COMPLETE", flush=True)
