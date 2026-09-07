"""Stage C: developing-market backtest (the information-asymmetry test), Swedish slice, and the GRU
on the main slice — all under the run id written by stage A so the dashboard shows them together."""

import logging
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

from pitch_edge.backtest.engine import WalkForwardConfig  # noqa: E402
from pitch_edge.config import get_settings  # noqa: E402
from pitch_edge.data.storage import Warehouse  # noqa: E402
from pitch_edge.models import DixonColesMatchModel, GBDTMatchModel, GRUSequenceModel  # noqa: E402
from pitch_edge.pipeline import load_feature_frame, run_backtests  # noqa: E402

DEVELOPING = [
    "ARG",
    "BRA",
    "MEX",
    "JAP",
    "USA",
    "NOR",
    "SWE",
    "DEN",
    "POL",
    "ROM",
    "RUS",
    "CHN",
    "IRL",
    "FIN",
    "AUT",
    "SUI",
]
s = get_settings()
s.ensure_dirs()
run_id = (s.artifacts_dir / "run_id.txt").read_text().strip()
t0 = time.time()
with Warehouse(s.db_path) as wh:
    dev = load_feature_frame(wh, leagues=DEVELOPING, min_date="2013-01-01")
    print("developing features", dev.shape, flush=True)
    run_backtests(
        dev,
        [DixonColesMatchModel(), GBDTMatchModel(), GBDTMatchModel(include_market=True)],
        WalkForwardConfig(retrain_every_days=60),
        wh=wh,
        label="developing",
        run_id=run_id,
    )
    print("developing done", round(time.time() - t0), "s", flush=True)
    main = load_feature_frame(wh, min_date="2018-07-01")
    print("main-2018 features", main.shape, flush=True)
    run_backtests(
        main,
        [GRUSequenceModel(epochs=12)],
        WalkForwardConfig(retrain_every_days=90),
        wh=wh,
        label="main",
        run_id=run_id,
    )
    print("gru done", round(time.time() - t0), "s", flush=True)
print("STAGE C COMPLETE", round(time.time() - t0), "s", flush=True)
