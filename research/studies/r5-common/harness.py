"""R5 glue to the research harness (`pitch_edge.research`): registration, counted trials, slope values.

- `register(reg_id)` records `research/registry/<id>.yaml` in the ledger (idempotent; refuses an edited file).
- `record(reg_id, config, frame, metric, snapshot, extra, code_files)` runs `run_trial` once for one grid
  point and then reads the ledger back to confirm the row landed. The ledger is a DuckDB file that other
  studies write too: on a lock error ("Could not set lock") it waits and retries, and it never swallows any
  other exception. With R5_DRY=1 nothing is written (the numbers are computed and printed only).
- `slope_values(x, y)` returns per-decision influence values whose mean is exactly the OLS slope of y on x,
  so a slope can be scored with `run_trial` (mean of values, match-day block bootstrap). Bootstrapping the
  mean of these values is the linearised (influence-function) bootstrap of the slope; `exact_slope_ci`
  gives the exact block-bootstrap slope CI to report beside it.
- `file_hashes(paths)` gives sha256 per study code file, stored in each trial's `extra` as `code_files`.
"""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from pitch_edge.research.ledger import Ledger
from pitch_edge.research.registry import load
from pitch_edge.research.trial import TrialResult, run_trial

REPO = Path(__file__).resolve().parents[3]
REGISTRY = REPO / "research" / "registry"
N_BOOT = 5000
SEED = 20261020
LOCK_RETRIES = 30


def _is_lock_error(exc: BaseException) -> bool:
    return isinstance(exc, duckdb.IOException) and "lock" in str(exc).lower()


def _retry(fn, *args, **kwargs):  # type: ignore[no-untyped-def]
    delay = 2.0
    for attempt in range(LOCK_RETRIES):
        try:
            return fn(*args, **kwargs)
        except duckdb.IOException as exc:
            if not _is_lock_error(exc) or attempt == LOCK_RETRIES - 1:
                raise
            print(f"ledger locked ({exc}); retry in {delay:.0f}s", flush=True)
            time.sleep(delay)
            delay = min(delay * 1.5, 30.0)
    raise RuntimeError("unreachable")


def ledger() -> Ledger:
    return _retry(Ledger)


def register(reg_id: str) -> None:
    reg = load(REGISTRY / f"{reg_id}.yaml")
    new = _retry(ledger().register, reg)
    print(f"{reg_id}: {'registered' if new else 'already registered (unchanged)'}")


def file_hashes(paths: list[Path]) -> dict[str, str]:
    return {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def record(
    reg_id: str,
    config: dict[str, Any],
    frame: pd.DataFrame,
    metric: str,
    snapshot: str,
    extra: dict[str, Any],
    code_files: list[Path],
) -> TrialResult | None:
    """Score one grid point with `run_trial` and confirm the ledger holds it. Returns None in dry mode."""
    reg = load(REGISTRY / f"{reg_id}.yaml")
    extra = {**extra, "code_files": file_hashes(code_files)}
    if os.environ.get("R5_DRY") == "1":
        from pitch_edge.research.stats import block_bootstrap

        b = block_bootstrap(frame["value"].to_numpy(), frame["block"].to_numpy(), n_boot=N_BOOT, seed=SEED)
        print(f"[dry] {reg_id} {config} {metric}: {b.estimate:.6g} [{b.lo:.6g}, {b.hi:.6g}] n={b.n} days={b.n_blocks}")
        return None
    led = ledger()
    res = _retry(run_trial, reg, config, frame, metric, led, snapshot, extra, n_boot=N_BOOT, seed=SEED)
    rows = _retry(led.trials, hypothesis_id=reg_id)
    if res.trial_id not in set(rows["trial_id"]):
        raise RuntimeError(f"trial {res.trial_id} of {reg_id} is not in the ledger after run_trial")
    print(
        f"{reg_id} {config} {metric}: {res.metric_value:.6g} [{res.ci_lo:.6g}, {res.ci_hi:.6g}] "
        f"p={res.p_value:.4g} n={res.n} days={res.n_blocks} (trial {res.trial_id[:10]} verified)"
    )
    return res


def slope_values(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Influence values psi_i = b + n (x_i - xbar)(e_i) / Sxx, with e the OLS residual; mean(psi) = b exactly."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    xc = x - x.mean()
    sxx = float(np.sum(xc * xc))
    b = float(np.sum(xc * (y - y.mean())) / sxx)
    resid = (y - y.mean()) - b * xc
    return b + len(x) * xc * resid / sxx


def exact_slope_ci(x: np.ndarray, y: np.ndarray, blocks: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """OLS slope with its exact match-day block-bootstrap percentile CI (resampling whole blocks)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    codes, uniq = pd.factorize(pd.Series(blocks))
    nb = len(uniq)
    agg = np.vstack([np.bincount(codes, weights=v, minlength=nb) for v in (np.ones_like(x), x, y, x * x, x * y)])

    def slope(a: np.ndarray) -> np.ndarray:
        n, sx, sy, sxx, sxy = a
        return (n * sxy - sx * sy) / (n * sxx - sx * sx)

    point = float(slope(agg.sum(axis=1)))
    rng = np.random.default_rng(seed)
    w = np.zeros((n_boot, nb))
    for i in range(n_boot):
        w[i] = np.bincount(rng.integers(0, nb, nb), minlength=nb)
    boot = slope(agg @ w.T)
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return {"slope": point, "ci_lo": float(lo), "ci_hi": float(hi), "n": len(x), "n_days": nb}
