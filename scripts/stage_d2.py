"""Stage D2: re-run the Lightning GRU + Transformer walk-forward with the corrected recipe (same run id)."""

import logging
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
from pitch_edge.backtest.engine import WalkForwardConfig
from pitch_edge.config import get_settings
from pitch_edge.data.storage import Warehouse
from pitch_edge.models import GRUSequenceModel, TransformerSequenceModel
from pitch_edge.pipeline import load_feature_frame, run_backtests

s = get_settings()
t0 = time.time()
run_id = (s.artifacts_dir / "run_id.txt").read_text().strip()
with Warehouse(s.db_path) as wh:
    for t in ("backtest_bets", "backtest_summaries", "model_predictions"):
        wh._conn.execute(
            f"DELETE FROM {t} WHERE backtest_id LIKE 'main:gru_sequence:%' OR backtest_id LIKE 'main:transformer_sequence:%'"
            if t != "model_predictions"
            else f"DELETE FROM {t} WHERE model_name IN ('gru_sequence','transformer_sequence')"
        )
    f = load_feature_frame(wh, min_date="2018-07-01")
    print("features", f.shape, flush=True)
    run_backtests(
        f,
        [GRUSequenceModel(), TransformerSequenceModel()],
        WalkForwardConfig(retrain_every_days=90),
        wh=wh,
        label="main",
        run_id=run_id,
    )
print("STAGE D2 COMPLETE", round(time.time() - t0), "s", flush=True)
