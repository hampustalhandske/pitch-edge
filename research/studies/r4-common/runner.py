"""Run one R4 hypothesis (all of its registered grid points) through the research harness.

`run(reg_id, kind, code_files)` loads `panel.parquet`, and for each registered `lookback_h` builds the frame
(`kind` = "h1" attention vs move size, "mention" or "sentiment" skew vs move direction), fits the walk-forward
baseline and full models over the registered evaluation window, turns the OOS errors into per-decision Δ OOS R²
values and records one trial per grid point with `r5-common/harness.record` (which calls `run_trial` and reads the
ledger back). Diagnostics per grid point (pooled OOS R² of both models, per-fold crowd coefficients, thirds,
per-league table, sign of the crowd coefficient) go to data/research/r4/<reg_id>/.

R5_DRY=1 computes and prints without writing to the ledger.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "r5-common"))

from harness import N_BOOT, SEED, record  # noqa: E402  (r5-common)
from model import H1_BASE, H1_CROWD, H2_BASE, H2_CROWD, h1_frame, h2_frame, per_decision, walk_forward  # noqa: E402
from panel import DATA  # noqa: E402

from pitch_edge.research.registry import load  # noqa: E402
from pitch_edge.research.stats import block_bootstrap  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
CEILING = 0.02  # Δ OOS R² above 2 pp of the move variance is a leakage suspect (registration literature_ceiling)
MIN_LEAGUE_N = 100


def _frame(panel: pd.DataFrame, kind: str, L: int) -> tuple[pd.DataFrame, list[str], list[str]]:
    if kind == "h1":
        return h1_frame(panel, L), H1_BASE, H1_CROWD
    return h2_frame(panel, L, kind), H2_BASE, H2_CROWD


def _ci(dec: pd.DataFrame) -> list[float]:
    b = block_bootstrap(dec.value.to_numpy(), dec.block.to_numpy(), n_boot=N_BOOT, seed=SEED)
    return [round(float(b.estimate), 6), round(float(b.lo), 6), round(float(b.hi), 6), int(b.n)]


def run(reg_id: str, kind: str, code_files: list[Path]) -> None:
    reg = load(REPO / "research" / "registry" / f"{reg_id}.yaml")
    ev_from, ev_to = (str(x) for x in reg.window.evaluation)
    panel = pd.read_parquet(DATA / "panel.parquet")
    snap = json.loads((DATA / "snapshot_meta.json").read_text())
    bars_snap = json.loads((DATA / "snapshot_bars.json").read_text())
    bsky_snap = json.loads((DATA / "snapshot_bsky.json").read_text())
    snapshot = (
        f"lake meta {snap['pulled_at']}; bars {bars_snap['finished_at']} rows={bars_snap['rows']}; "
        f"bsky {bsky_snap['finished_at']} rows={bsky_snap['rows']}; panel rows={len(panel)}"
    )
    out_dir = DATA / reg_id
    out_dir.mkdir(parents=True, exist_ok=True)
    for cfg in reg.expand_grid():
        L = int(cfg["lookback_h"])
        df, base, crowd = _frame(panel, kind, L)
        pred, folds = walk_forward(df, base, crowd, ev_from, ev_to)
        dec, summ = per_decision(pred)
        assert dec.block.min() >= ev_from and dec.block.max() <= ev_to
        # stability diagnostics (not gates by themselves; run_trial recomputes thirds and LOLO)
        blocks = sorted(dec.block.unique())
        cuts = [blocks[len(blocks) // 3], blocks[2 * len(blocks) // 3]]
        dec["third"] = np.where(dec.block < cuts[0], 1, np.where(dec.block < cuts[1], 2, 3))
        thirds = {int(t): _ci(s) for t, s in dec.groupby("third")}
        lc = dec.league.value_counts()
        dec["league_pooled"] = np.where(dec.league.map(lc) >= MIN_LEAGUE_N, dec.league, "pooled-small")
        leagues = {k: _ci(s) for k, s in dec.groupby("league_pooled") if s.block.nunique() >= 2}
        coef_sign = {c: [f["crowd_coef"][c] for f in folds] for c in crowd}
        diag = {
            "config": cfg,
            "summary": summ,
            "folds": folds,
            "thirds": thirds,
            "leagues": leagues,
            "crowd_coef_by_fold": coef_sign,
        }
        (out_dir / f"diag_L{L}.json").write_text(json.dumps(diag, indent=2, default=str))
        pred.to_parquet(out_dir / f"pred_L{L}.parquet")
        frame = dec[["decision_id", "block", "value", "league_pooled"]].rename(columns={"league_pooled": "league"})
        est = float(frame.value.mean())
        extra = {
            "clean": True,
            "pit_assertions": "panel.py asserts every bar used as a feature has known_at <= H and every Bluesky bucket "
            "is known by H; target starts at the first bar after H; walk_forward asserts train ko < fold start <= test ko",
            "within_ceiling": bool(abs(est) <= CEILING),
            "ceiling": CEILING,
            "summary": summ,
            "folds": folds,
            "metric_unit": "share of OOS target variance (Δ OOS R²), per-decision mean over hours",
        }
        record(reg_id, cfg, frame, reg.primary_metric, snapshot, extra, code_files)
        print(
            json.dumps(
                {"config": cfg, "summary": summ, "thirds": thirds, "leagues": leagues, "crowd_coef_by_fold": coef_sign},
                indent=1,
                default=str,
            )
        )
