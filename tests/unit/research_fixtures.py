"""Builders shared by the `test_research_*` modules: registration files and synthetic per-decision data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def registration_text(
    hypothesis_id: str = "sharp-close-test-v1",
    family: str = "sharp-close",
    *,
    direction: str = "positive",
    primary_metric: str = "edge_bits",
    secondary_metrics: list[str] | None = None,
    grid: dict[str, list[Any]] | None = None,
    min_n: int = 300,
    extra_windows: dict[str, list[str]] | None = None,
    extra_fields: dict[str, Any] | None = None,
) -> str:
    doc = {
        "id": hypothesis_id,
        "family": family,
        "registered_at": "2026-10-20T21:00:00Z",
        "author": "test",
        "question": "Does the test strategy beat the de-vigged close?",
        "hypothesis": {"h1": "edge > 0", "h0": "edge = 0", "direction": direction, "disconfirming": "CI includes 0"},
        "decision_time": "kickoff - 24h",
        "universe": "synthetic",
        "window": {
            "exploratory": ["2020-01-01", "2021-12-31"],
            "evaluation": ["2022-01-01", "2023-12-31"],
            **(extra_windows or {}),
        },
        "benchmark": "synthetic close",
        "primary_metric": primary_metric,
        "secondary_metrics": secondary_metrics if secondary_metrics is not None else ["roi"],
        "costs": "none",
        "grid": grid if grid is not None else {},
        "min_n": min_n,
        "literature_ceiling": "+0.02 bits",
        **(extra_fields or {}),
    }
    return yaml.safe_dump(doc, sort_keys=False)


def write_registration(directory: Path, hypothesis_id: str = "sharp-close-test-v1", **kwargs: Any) -> Path:
    path = directory / f"{hypothesis_id}.yaml"
    path.write_text(registration_text(hypothesis_id, **kwargs))
    return path


def per_decision(
    n_blocks: int,
    per_block: int,
    mu: float,
    sd: float = 0.2,
    seed: int = 0,
    leagues: tuple[str, ...] | None = ("E0", "D1"),
) -> pd.DataFrame:
    """`n_blocks * per_block` decisions with mean `mu`; blocks are ISO match-day strings in order. With
    `leagues`, a `league` column cycles through them decision by decision (so every league plays every day)."""
    rng = np.random.default_rng(seed)
    days = pd.date_range("2022-01-01", periods=n_blocks, freq="D").strftime("%Y-%m-%d")
    block = np.repeat(days.to_numpy(), per_block)
    n = n_blocks * per_block
    df = pd.DataFrame(
        {"decision_id": [f"d{i}" for i in range(n)], "block": block, "value": mu + rng.normal(0, sd, size=n)}
    )
    if leagues:
        df["league"] = [leagues[i % len(leagues)] for i in range(n)]
    return df


def native_record(reg: Any, config: dict[str, Any] | None = None, **over: Any) -> Any:
    """A `TrialRecord` that passes the ledger's validation for a native (non-imported) trial of `reg`:
    a registered grid point, a registered metric, CI around the estimate, p consistent with p_gt0/p_lt0.
    Keyword overrides replace record fields; `extra=` entries are merged into the default extra."""
    import json

    from pitch_edge.research.ledger import TrialRecord
    from pitch_edge.research.registry import config_hash

    cfg = config if config is not None else reg.expand_grid()[0]
    direction = reg.direction_for(cfg)
    extra = {
        "p_gt0": 0.001,
        "p_lt0": 0.999,
        "n_blocks": 100,
        "n_boot": 2000,
        "seed": 0,
        "alpha": 0.05,
        "window": "evaluation",
        "is_primary_metric": True,
        "registration_sha256": reg.source_sha256,
        "clean": True,
    }
    extra.update(over.pop("extra", {}))
    p = {"positive": extra["p_gt0"], "negative": extra["p_lt0"]}.get(
        direction, min(1.0, 2 * min(extra["p_gt0"], extra["p_lt0"]))
    )
    fields = {
        "hypothesis_id": reg.id,
        "family": reg.family,
        "config_json": json.dumps(cfg, sort_keys=True),
        "config_hash": config_hash(cfg),
        "git_sha": "abc123",
        "data_snapshot": "snap",
        "n": 400,
        "metric_name": reg.primary_metric,
        "metric_value": 0.05,
        "ci_lo": 0.03,
        "ci_hi": 0.07,
        "p_value": p,
        "direction": direction,
        "extra_json": json.dumps(extra),
    }
    fields.update(over)
    return TrialRecord(**fields)
