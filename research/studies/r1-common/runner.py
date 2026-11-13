"""Shared trial runner for the R1 studies: registration check, filtering, scoring summaries, ledger rows.

A study script calls `start(reg_id, window)` to load its registration (refusing a missing one), the match
frame restricted to the window, and the Polymarket price book; it then builds per-decision values and calls
`trial_row(...)` once per grid point, and `finish(...)` to append the rows to its trials.csv and write the
per-trial detail JSON under data/research/r1/<id>/.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from common import (
    DATA,
    REGISTRY,
    REPO,
    PriceBook,
    append_trials,
    block_bootstrap,
    config_hash,
    file_sha256,
    git_sha,
    load_matches,
    one_sided_p,
    snapshot_tag,
)

N_BOOT = 5000


def start(reg_id: str, window: str) -> tuple[dict, str, pd.DataFrame, PriceBook, dict]:
    reg_path = REGISTRY / f"{reg_id}.yaml"
    if not reg_path.exists():
        raise SystemExit(f"refusing to run: {reg_path} is not registered")
    reg = yaml.safe_load(reg_path.read_text())
    assert reg["id"] == reg_id
    df, drops = load_matches()
    df = df[df.window == window].reset_index(drop=True)
    book = PriceBook(pd.read_parquet(DATA / "bars.parquet"))
    return reg, file_sha256(reg_path), df, book, drops


def summarise(values: np.ndarray, days: np.ndarray, direction: str) -> dict:
    values = np.asarray(values, float)
    if len(values) == 0:
        return {"n": 0, "n_days": 0, "value": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "p": np.nan}
    boot = block_bootstrap(values, days, n_boot=N_BOOT)
    point = float(values.mean())
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "n": len(values),
        "n_days": int(pd.Series(days).nunique()),
        "value": point,
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "p": float(one_sided_p(boot, point, direction)),
    }


def thirds(values: np.ndarray, days: np.ndarray, direction: str) -> list[dict]:
    """Metric per chronological third of match-days."""
    d = pd.Series(days).reset_index(drop=True)
    uniq = np.sort(d.unique())
    cuts = np.array_split(uniq, 3)
    out = []
    for i, c in enumerate(cuts):
        if len(c) == 0:
            continue
        m = d.isin(c).to_numpy()
        s = summarise(np.asarray(values)[m], d[m].to_numpy(), direction)
        out.append({"third": i + 1, "from": str(c[0]), "to": str(c[-1]), **s})
    return out


def by_league(
    values: np.ndarray, days: np.ndarray, leagues: np.ndarray, direction: str, min_n: int = 100
) -> list[dict]:
    """Metric per division, pooling divisions with fewer than `min_n` values into 'pooled_small'."""
    lg = pd.Series(leagues).reset_index(drop=True)
    counts = lg.value_counts()
    lab = lg.where(lg.map(counts) >= min_n, "pooled_small")
    out = []
    for name in sorted(lab.unique()):
        m = (lab == name).to_numpy()
        out.append({"league": name, **summarise(np.asarray(values)[m], np.asarray(days)[m], direction)})
    return out


def trial_row(reg: dict, reg_sha: str, window: str, cfg: dict, metric: str, s: dict, direction: str) -> dict:
    return {
        "id": reg["id"],
        "config_hash": config_hash(reg["id"], cfg),
        "config": json.dumps(cfg, sort_keys=True),
        "registration_sha256": reg_sha,
        "git_sha": git_sha(),
        "data_snapshot": snapshot_tag(),
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "window": window,
        "n": s["n"],
        "n_days": s["n_days"],
        "metric": metric,
        "value": s["value"],
        "ci_lo": s["ci_lo"],
        "ci_hi": s["ci_hi"],
        "p_one_sided": s["p"],
        "direction": direction,
    }


def finish(study_dir: Path, reg_id: str, window: str, rows: list[dict], detail: dict) -> None:
    """Append the trial rows to the ledger and write the detail JSON.

    With R1_REPRO_CHECK=1 nothing is written: the rows are only printed, to check that a rerun reproduces
    the ledger's numbers without adding duplicate trials.
    """
    if os.environ.get("R1_REPRO_CHECK") != "1":
        append_trials(study_dir, rows)
        out = REPO / "data" / "research" / "r1" / reg_id
        out.mkdir(parents=True, exist_ok=True)
        (out / f"detail_{window}.json").write_text(json.dumps(detail, indent=2, default=str))
    print(pd.DataFrame(rows)[["config", "n", "n_days", "metric", "value", "ci_lo", "ci_hi", "p_one_sided"]].to_string())
