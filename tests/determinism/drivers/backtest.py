"""Driver: feature build and walk-forward backtests of the two cheap models on the synthetic league.

Input: `make_synthetic_league(n_teams=8, n_seasons=3, seed=42)` (tests/conftest.py), built into features by
`FeatureBuilder`, then `WalkForwardBacktester` (100 training matches, a refit every 60 days, edge threshold 0,
isotonic calibration from 60 realised rows) for `DixonColesMatchModel` and `GBDTMatchModel` (40 trees).

Output: the feature frame, and per model the predictions, bets, strategy summaries and calibration block. No
field is volatile: nothing here reads a clock or draws an id.

Defect: `unseeded` adds an unseeded random jitter to the Dixon-Coles probabilities (an unseeded Monte Carlo step).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parents[1])]


def _stub_sequence_models() -> None:
    """Register an empty `pitch_edge.models.sequence` (its two class names only) before the models package loads.

    The package imports the sequence models, whose torch / lightning / transformers stack costs about 9 s per
    process and is never used by the two models run here. Neither model's code path touches the stub."""
    import types

    stub = types.ModuleType("pitch_edge.models.sequence")
    stub.GRUSequenceModel = type("GRUSequenceModel", (), {})  # type: ignore[attr-defined]
    stub.TransformerSequenceModel = type("TransformerSequenceModel", (), {})  # type: ignore[attr-defined]
    sys.modules[stub.__name__] = stub


_stub_sequence_models()

import numpy as np  # noqa: E402
from canon import driver_args, finish, frame_records, write_canonical  # noqa: E402
from conftest import make_synthetic_league  # noqa: E402

from pitch_edge.backtest.engine import WalkForwardBacktester, WalkForwardConfig  # noqa: E402
from pitch_edge.features.build import FeatureBuilder  # noqa: E402
from pitch_edge.models import DixonColesMatchModel, GBDTMatchModel  # noqa: E402

CONFIG = WalkForwardConfig(min_train_matches=100, retrain_every_days=60, edge_threshold=0.0, calibration_min_rows=60)


def plant_unseeded() -> None:
    original = DixonColesMatchModel.predict_proba

    def jittered(self: Any, df: Any) -> Any:
        probs = original(self, df)
        noisy = probs.to_numpy() * (1 + np.random.default_rng().normal(0, 1e-6, probs.shape))
        probs.loc[:, :] = noisy / noisy.sum(axis=1, keepdims=True)
        return probs

    DixonColesMatchModel.predict_proba = jittered  # type: ignore[method-assign]


def main() -> None:
    args = driver_args()
    if args.defect == "unseeded":
        plant_unseeded()
    features = FeatureBuilder().build(make_synthetic_league(n_teams=8, n_seasons=3, seed=42))
    results = {}
    for model in (DixonColesMatchModel(), GBDTMatchModel(n_estimators=40)):
        r = WalkForwardBacktester(CONFIG).run(features, model)
        results[r.model_name] = {
            "predictions": frame_records(r.predictions),
            "bets": frame_records(r.bets),
            "summaries": frame_records(r.summaries),
            "calibration": r.calibration,
        }
    write_canonical(args.out, {"features": frame_records(features), "backtests": results})


if __name__ == "__main__":
    main()
    finish()
