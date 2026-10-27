"""Walk-forward predictions of one `ask` model with the production backtester.

  generate.py <model> [--cut YYYY-MM-DD]

Reads data/research/r7/features.parquet (the warehouse feature frame snapshot), keeps matches dated <= the end of the
registered evaluation window (or <= --cut for the truncation-invariance test), runs
`pitch_edge.backtest.engine.WalkForwardBacktester` (min_train_matches=400, retrain_every_days=14, isotonic calibration on
realised past folds, no staking strategies) and writes data/research/r7/pred_<model>[_cut].parquet with the raw and
calibrated probabilities, the Pinnacle open/close odds and the walk-forward fold each prediction belongs to
(`fold_start`, `train_max_date`, `train_rows`). Nothing is written to the warehouse.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pitch_edge.backtest.engine import WalkForwardBacktester, WalkForwardConfig
from pitch_edge.models import DixonColesMatchModel, GBDTMatchModel, StochasticStrengthModel

OUT = Path(__file__).resolve().parents[3] / "data" / "research" / "r7"
EVAL_END = pd.Timestamp("2025-06-30")
CFG = {"min_train_matches": 400, "retrain_every_days": 14, "calibrate": True, "strategies": []}

MODELS = {
    "dixon_coles": DixonColesMatchModel,
    "gbdt": lambda: GBDTMatchModel(include_market=False),
    "stochastic_strength": StochasticStrengthModel,
}


def main() -> None:
    name = sys.argv[1]
    cut = pd.Timestamp(sys.argv[sys.argv.index("--cut") + 1]) if "--cut" in sys.argv else None
    end = cut or EVAL_END
    f = pd.read_parquet(OUT / "features.parquet")
    f["date"] = pd.to_datetime(f["date"])
    f = f[f["date"] <= end].reset_index(drop=True)
    model = MODELS[name]()
    assert model.name == name
    cfg = WalkForwardConfig(**CFG)
    t0 = time.time()
    res = WalkForwardBacktester(cfg).run(f, model)
    pred = res.predictions
    # fold bookkeeping, re-derived exactly as the engine does it
    d = f.sort_values(["date", "match_id"]).reset_index(drop=True)
    starts = pd.date_range(d.loc[cfg.min_train_matches, "date"], d["date"].max(), freq=f"{cfg.retrain_every_days}D")
    idx = np.searchsorted(starts.values, pred["date"].values, side="right") - 1
    pred["fold_start"] = starts[idx]
    pred["train_rows"] = [int((d["date"] < s).sum()) for s in pred["fold_start"]]
    tm = {s: d.loc[d["date"] < s, "date"].max() for s in pred["fold_start"].unique()}
    pred["train_max_date"] = pred["fold_start"].map(tm)
    cols_extra = ["PSH", "PSD", "PSA", "PSCH", "PSCD", "PSCA"]
    pred = pred.drop(columns=[c for c in pred.columns if c.startswith(("bet_odds_", "close_odds_", "mkt_", "edge_"))])
    pred = pred.merge(d[["match_id"] + cols_extra], on="match_id", how="left", validate="one_to_one")
    pred["model_name"] = name
    suffix = "_cut" if cut is not None else ""
    pred.to_parquet(OUT / f"pred_{name}{suffix}.parquet")
    print(name, "rows", len(pred), "folds", pred["fold_start"].nunique(), "secs", round(time.time() - t0))


if __name__ == "__main__":
    main()
