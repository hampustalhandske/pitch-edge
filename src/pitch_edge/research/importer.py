"""Importing study-local `trials.csv` files into the ledger, so every trial a study ran before the
harness existed counts in its family's multiple-testing corrections.

Two formats are recognised by their columns:

- R1 (`research/studies/r1-*/trials.csv`): id, config_hash, config, registration_sha256, git_sha,
  data_snapshot, run_at, window, n, n_days, metric, value, ci_lo, ci_hi, p_one_sided, direction.
- R2 (`research/studies/r2-*/trials.csv`): id, config_hash, git_sha, run_at, n, metric (the value),
  ci_lo, ci_hi, metric_name, window, label, direction, p_value, sr, skew, kurt, registration_sha256,
  data_snapshot, config_json.

Each row becomes one ledger trial marked `imported: true`, keeping its original timestamp, window,
direction, git sha and registration sha256. A row's config is mapped onto a registered grid point when
its values for every grid axis equal that point (the full original config is kept as
`original_config`); otherwise it is stored as its own off-grid config (`off_grid: true`). The trial id is
derived from the row's content, so importing the same file twice adds nothing.

Which rows carry the primary metric: the names in `primary_metrics` when given; otherwise a row whose
metric name equals the registration's `primary_metric` exactly, or is an alias of it — listed in
`METRIC_ALIASES` (CSV name -> the exact registered strings it stands for) or passed as `aliases`. There is
no fuzzy matching: an unlisted name is a secondary metric, however similar it looks. A row matched through
an alias records `metric_alias_of` (the registered string).

Imported rows count in BH m and DSR N; they can never clear L1 on their own (no `clean` flag,
reproduction or archived code), and their per-decision Sharpe moments are kept as `imported_sr` etc.,
not mixed with the per-block ones the gates use.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from pitch_edge.research.errors import ProtocolError
from pitch_edge.research.ledger import DIRECTIONS, Ledger, TrialRecord
from pitch_edge.research.registry import Registration, config_hash, load, relative_registry_path, resolve_registry_path

# CSV metric name -> the exact registered primary-metric strings it is an alias of.
_R2_CLV = (
    "clv_log = log(odds_taken * p_close_devig[o]), mean per bet",
    "clv_log = log(odds_taken * p_close_devig[side]), mean per bet",
)
_R2_Q1 = ("calibration_slope_minus_1 (OLS, home+away rows), draw_bias, paired mean log-loss difference",)
METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "clv_log_mean": ("clv_log", *_R2_CLV),
    "ols_slope_minus_1": _R2_Q1,
    "draw_bias": _R2_Q1,
    "mean_logloss_diff_nats": _R2_Q1,
}

R1_COLUMNS = {"config", "metric", "value", "p_one_sided"}
R2_COLUMNS = {"config_json", "metric_name", "metric", "p_value"}


@dataclass
class ImportReport:
    """Counts from one import: rows written, rows already present, rows for other ids, rows refused (with
    reasons), rows stored off-grid, and the metric names treated as primary."""

    imported: int = 0
    duplicates: int = 0
    other_ids: int = 0
    off_grid: int = 0
    refused: list[str] = field(default_factory=list)
    primary_metrics: list[str] = field(default_factory=list)


def _num(x: Any) -> float | None:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    cols = set(df.columns)
    if R1_COLUMNS <= cols:
        return pd.DataFrame(
            {
                "id": df["id"],
                "config": df["config"],
                "metric_name": df["metric"],
                "value": df["value"],
                "p_value": df["p_one_sided"],
                "n_blocks": df.get("n_days"),
                "label": None,
                "sr": None,
                "skew": None,
                "kurt": None,
            }
        ).join(
            df[
                [
                    "registration_sha256",
                    "git_sha",
                    "data_snapshot",
                    "run_at",
                    "window",
                    "n",
                    "ci_lo",
                    "ci_hi",
                    "direction",
                ]
            ]
        )
    if R2_COLUMNS <= cols:
        return pd.DataFrame(
            {
                "id": df["id"],
                "config": df["config_json"],
                "metric_name": df["metric_name"],
                "value": df["metric"],
                "p_value": df["p_value"],
                "n_blocks": None,
                "label": df.get("label"),
                "sr": df.get("sr"),
                "skew": df.get("skew"),
                "kurt": df.get("kurt"),
            }
        ).join(
            df[
                [
                    "registration_sha256",
                    "git_sha",
                    "data_snapshot",
                    "run_at",
                    "window",
                    "n",
                    "ci_lo",
                    "ci_hi",
                    "direction",
                ]
            ]
        )
    raise ValueError(f"unrecognised trials.csv columns: {sorted(cols)}")


def alias_target(name: str, registration: Registration, aliases: dict[str, str] | None = None) -> str | None:
    """The registered primary metric `name` stands for — itself when equal, else through `aliases` or
    `METRIC_ALIASES` (exact strings only) — or None when it is not the primary metric."""
    primary = registration.primary_metric
    if name == primary:
        return primary
    if aliases and aliases.get(name) == primary:
        return primary
    return primary if primary in METRIC_ALIASES.get(name, ()) else None


def default_primary_metrics(
    registration: Registration, metric_names: Sequence[str], aliases: dict[str, str] | None = None
) -> list[str]:
    """Metric names in `metric_names` that are the registration's primary metric (see module docstring)."""
    return sorted({m for m in metric_names if alias_target(m, registration, aliases) is not None})


def _grid_point(registration: Registration, config: dict[str, Any]) -> dict[str, Any] | None:
    keys = list(registration.grid)
    if not keys or any(k not in config for k in keys):
        return None
    for point in registration.expand_grid():
        if all(point[k] == config[k] for k in keys):
            return point
    return None


def import_trials_csv(
    csv_path: str | Path,
    hypothesis_id: str,
    ledger: Ledger,
    primary_metrics: Sequence[str] | None = None,
    aliases: dict[str, str] | None = None,
) -> ImportReport:
    """Import the rows of `csv_path` belonging to `hypothesis_id` as imported trials.

    `primary_metrics` names the CSV metrics to treat as primary outright; `aliases` adds CSV-name ->
    registered-primary-string pairs to `METRIC_ALIASES` for this import. Raises ValueError when the
    hypothesis is not registered in `ledger` or the CSV format is unknown.
    Rows that the ledger refuses (e.g. an unknown direction or ci_lo > ci_hi) are skipped and listed in
    the report; nothing else is skipped silently.
    """
    registered = ledger.registration(hypothesis_id)
    if registered is None:
        raise ValueError(f"{hypothesis_id} is not registered; register it before importing its trials")
    registration = load(resolve_registry_path(registered["path"]))
    raw = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    rows = _normalise(raw)
    mine = rows[rows["id"] == hypothesis_id]
    report = ImportReport(other_ids=int((rows["id"] != hypothesis_id).sum()))
    names = sorted(set(mine["metric_name"]))
    report.primary_metrics = (
        sorted(set(primary_metrics)) if primary_metrics else default_primary_metrics(registration, names, aliases)
    )
    existing = set(ledger.trials(hypothesis_id=hypothesis_id)["trial_id"])
    source = relative_registry_path(csv_path)

    for i, (_, row) in enumerate(mine.iterrows()):
        original = raw.loc[row.name].to_dict()
        row_sha = hashlib.sha256(json.dumps(original, sort_keys=True).encode()).hexdigest()
        trial_id = f"imp-{row_sha[:32]}"
        if trial_id in existing:
            report.duplicates += 1
            continue
        try:
            config = json.loads(row["config"]) if row["config"] else {}
        except json.JSONDecodeError:
            report.refused.append(f"row {i}: config is not JSON")
            continue
        point = _grid_point(registration, config)
        stored_config = point if point is not None else config
        direction = str(row["direction"]).strip()
        if direction not in DIRECTIONS:
            report.refused.append(f"row {i}: direction {direction!r} is not one of {DIRECTIONS}")
            continue
        value, lo, hi, p = _num(row["value"]), _num(row["ci_lo"]), _num(row["ci_hi"]), _num(row["p_value"])
        warnings = []
        if value is not None and lo is not None and hi is not None and not (lo <= value <= hi):
            warnings.append("estimate outside its CI")
        extra: dict[str, Any] = {
            "imported": True,
            "source_csv": source,
            "source_row_sha256": row_sha,
            "window": str(row["window"]) or "evaluation",
            "is_primary_metric": row["metric_name"] in report.primary_metrics,
            "metric_alias_of": (
                alias_target(str(row["metric_name"]), registration, aliases)
                if row["metric_name"] in report.primary_metrics and row["metric_name"] != registration.primary_metric
                else None
            ),
            "original_config": config,
            "off_grid": point is None,
            "registration_sha256": row["registration_sha256"] or None,
        }
        if _num(row["n_blocks"]) is not None:
            extra["n_blocks"] = int(float(row["n_blocks"]))
        for key in ("sr", "skew", "kurt"):
            if _num(row[key]) is not None:
                extra[f"imported_{key}"] = _num(row[key])
        if row["label"]:
            extra["label"] = row["label"]
        if warnings:
            extra["validation_warnings"] = warnings
        run_at = pd.Timestamp(row["run_at"])
        run_at = run_at.tz_convert("UTC").tz_localize(None) if run_at.tzinfo else run_at
        record = TrialRecord(
            hypothesis_id=hypothesis_id,
            family=registration.family,
            config_json=json.dumps(stored_config, sort_keys=True, default=str),
            config_hash=config_hash(stored_config),
            git_sha=row["git_sha"] or "unknown",
            data_snapshot=row["data_snapshot"] or "unknown",
            n=int(float(row["n"])) if _num(row["n"]) is not None else 0,
            metric_name=str(row["metric_name"]),
            metric_value=value,
            ci_lo=lo,
            ci_hi=hi,
            p_value=p,
            direction=direction,
            extra_json=json.dumps(extra, sort_keys=True, default=str),
            trial_id=trial_id,
            run_at=run_at.to_pydatetime(),
        )
        try:
            ledger.record_trial(record)
        except ProtocolError as exc:
            report.refused.append(f"row {i}: {exc}")
            continue
        existing.add(trial_id)
        report.imported += 1
        report.off_grid += point is None
    return report
