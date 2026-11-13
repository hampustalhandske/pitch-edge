"""Running one counted trial: score a registered grid point with a block bootstrap and append it to the
ledger with its provenance (git sha, code hash, config hash, data snapshot, row counts, seed).

`run_trial` is the only way a native trial row is meant to enter the ledger. It refuses
(ProtocolError) to run an unregistered hypothesis, a registration whose file changed since it was
registered, a config that is not a point of the registered grid, a metric the registration does not
name, or an unknown window; the ledger re-validates the row it writes.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pitch_edge.research.errors import ProtocolError
from pitch_edge.research.ledger import Ledger, TrialRecord
from pitch_edge.research.registry import REPO_ROOT, Registration, config_hash, sha256
from pitch_edge.research.stats import block_bootstrap, directional_p_value

REQUIRED_COLUMNS = ("decision_id", "block", "value")
RESEARCH_CODE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class TrialResult:
    """What `run_trial` recorded: the headline (`metric_value` with its block-bootstrap CI), the p-value
    in the registered direction, both one-sided p-values, provenance and the stored `extra` dict."""

    trial_id: str
    hypothesis_id: str
    family: str
    config: dict[str, Any]
    config_hash: str
    metric_name: str
    n: int
    n_blocks: int
    metric_value: float
    ci_lo: float
    ci_hi: float
    p_value: float
    p_gt0: float
    p_lt0: float
    direction: str
    git_sha: str
    data_snapshot: str
    run_at: datetime
    extra: dict[str, Any] = field(default_factory=dict)


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=RESEARCH_CODE_DIR, capture_output=True, text=True, timeout=10, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def git_state() -> tuple[str, bool | None]:
    """(HEAD commit sha of the repo holding this code or 'unknown', whether the working tree has
    uncommitted changes or None when git is unavailable)."""
    sha = _git("rev-parse", "HEAD")
    if not sha:
        return "unknown", None
    status = _git("status", "--porcelain")
    return sha, (bool(status) if status is not None else None)


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.name


def code_hash(code_files: Sequence[str | Path] = ()) -> tuple[str, list[Path]]:
    """sha256 over (path, sha256 of bytes) of the caller's study code files plus every file of this
    package (`src/pitch_edge/research/**/*.py`), sorted by path. Returns the hash and the files hashed."""
    files = sorted({Path(f).resolve() for f in code_files} | set(RESEARCH_CODE_DIR.rglob("*.py")))
    h = hashlib.sha256()
    for f in files:
        h.update(f"{_display_path(f)}\0{hashlib.sha256(f.read_bytes()).hexdigest()}\n".encode())
    return h.hexdigest(), files


def _archive_code(files: list[Path], destination: Path) -> None:
    """Copy the hashed files under `destination`, keeping repository-relative paths (files outside the
    repository by name). Idempotent: an existing archive for the same hash is left as it is."""
    if destination.exists():
        return
    tmp = destination.with_name(destination.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    for f in files:
        target = tmp / _display_path(f)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, target)
    tmp.rename(destination)


def _validate_frame(per_decision: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in per_decision.columns]
    if missing:
        raise ValueError(f"per_decision is missing columns {missing}; needs {list(REQUIRED_COLUMNS)}")
    if per_decision.empty:
        raise ValueError("per_decision is empty")
    if per_decision["decision_id"].duplicated().any():
        raise ValueError("per_decision has duplicate decision_id values (each decision may be scored once)")
    if per_decision["block"].isna().any():
        raise ValueError("per_decision has missing block labels")
    if "league" in per_decision.columns and per_decision["league"].isna().any():
        raise ValueError("per_decision has missing league labels")
    values = pd.to_numeric(per_decision["value"], errors="raise").astype(float)
    if not np.all(np.isfinite(values)):
        raise ValueError("per_decision has NaN or infinite values")
    return per_decision.assign(value=values)


def _computed_statistics(df: pd.DataFrame) -> tuple[dict[str, Any], list[list[Any]]]:
    """Sharpe-type moments of the per-block sums, chronological-thirds estimates, and the per-block series.

    The moments are taken over per-block *sums* — for a unit-stake strategy scored per bet, each
    match-day's total, i.e. the strategy's per-period P&L. That is the unit a Sharpe ratio is defined on,
    it absorbs the within-day dependence that makes per-decision moments overstate the effective sample
    size, and its mean is proportional to the decision-weighted headline. (Per-block *means* would weight
    a one-bet day like a forty-bet day.) `sr_n` is therefore the number of blocks.
    """
    by_block = df.groupby("block", sort=True)["value"].agg(["sum", "count"])
    sums = by_block["sum"]
    sd = float(sums.std(ddof=1)) if len(sums) > 1 else float("nan")
    ok = sd > 0
    stats: dict[str, Any] = {
        "sr": float(sums.mean() / sd) if ok else None,
        "skew": float(sums.skew()) if len(sums) > 2 and ok else None,
        "kurt": float(sums.kurt() + 3.0) if len(sums) > 3 and ok else None,  # non-excess (normal = 3)
        "sr_n": len(sums),
        "sr_unit": "per-block sum",
    }
    thirds = np.array_split(np.arange(len(by_block)), 3)
    stats["subperiod_estimates"] = [
        float(by_block["sum"].iloc[t].sum() / by_block["count"].iloc[t].sum()) for t in thirds if len(t)
    ]
    series = [[str(label), float(row["sum"]), int(row["count"])] for label, row in by_block.iterrows()]
    return stats, series


def _leave_one_league_out(df: pd.DataFrame, n_boot: int, seed: int, alpha: float) -> dict[str, dict[str, Any]]:
    """For each league, the headline recomputed without that league: {league: {est, lo, hi, n}} with the
    same block bootstrap (seed, draws, alpha) as the headline. Empty when the data has a single league
    (removing it leaves nothing). A remainder too small to bootstrap gets lo = hi = None."""
    leagues = sorted(df["league"].astype(str).unique())
    if len(leagues) < 2:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for league in leagues:
        rest = df[df["league"].astype(str) != league]
        try:
            b = block_bootstrap(
                rest["value"].to_numpy(), rest["block"].to_numpy(), n_boot=n_boot, seed=seed, alpha=alpha
            )
            out[league] = {"est": b.estimate, "lo": b.lo, "hi": b.hi, "n": len(rest)}
        except ValueError:
            out[league] = {"est": float(rest["value"].mean()), "lo": None, "hi": None, "n": len(rest)}
    return out


def _allowed_windows(registration: Registration) -> set[str]:
    return {"exploratory", "evaluation", *registration.window.extensions}


def run_trial(
    registration: Registration,
    config: dict[str, Any],
    per_decision: pd.DataFrame,
    metric_name: str,
    ledger: Ledger,
    data_snapshot: str,
    extra: dict[str, Any] | None = None,
    *,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
    window: str = "evaluation",
    code_files: Sequence[str | Path] = (),
) -> TrialResult:
    """Score one registered grid point and append it to `ledger`.

    Takes the loaded `registration`, one `config` from `registration.expand_grid()`, a `per_decision`
    frame with one row per decision (`decision_id` unique, `block` = resampling unit such as the
    match-day, `value` = the metric for that decision, and optionally `league`, e.g. log(taken odds / de-vigged close) for a CLV
    metric), the metric's name (the registration's primary or a secondary metric), the ledger, a label of
    the data the frame was built from (`data_snapshot`), an optional `extra` dict of the study's own
    analysis outputs the gates read (`clean`, `within_ceiling`, `leave_one_league_out`, ...), the
    bootstrap settings, which registered `window` the data comes from, and the study's code files.

    Computes the mean of `value` with a block-bootstrap percentile CI (`n_boot` draws, `seed`, level
    `1 - alpha`) and the one-sided p-values, then recomputes the bootstrap from scratch and requires the
    identical result (`reproduced: true`). Records the git sha and whether the tree is dirty, and a
    `code_hash` over `code_files` plus this package; when the tree is dirty or git is unavailable the
    hashed files are copied to `<ledger dir>/<id>/code/<code_hash>/` so the run stays reproducible.

    Stored in `extra_json`: the caller's keys; then, overriding any caller key of the same name (the
    caller's value is kept as `caller_<key>`): the bootstrap settings and p-values, `n_blocks`, `alpha`,
    `window`, `is_primary_metric`, the registration sha256, provenance, the per-block series
    `[label, sum, count]`, `sr`/`skew`/`kurt` (non-excess)/`sr_n` of the per-block sums, the
    chronological-thirds `subperiod_estimates` (blocks are assumed to sort chronologically), and — when
    `per_decision` has a `league` column — `leave_one_league_out` {league: {est, lo, hi, n}}, the
    headline and its block-bootstrap CI with each league removed (None without a league column).

    Returns the `TrialResult` that was written. Raises ProtocolError for a protocol violation and
    ValueError for an invalid frame (including fewer than two blocks) or an empty `data_snapshot`;
    nothing is written in either case.
    """
    path = registration.source_path
    if path is None:
        raise ProtocolError("run_trial needs a registration loaded from its file (registry.load)")
    registered = ledger.registration(registration.id)
    if registered is None:
        raise ProtocolError(f"{registration.id} is not registered; run `pitch-edge research register` first")
    current = sha256(path)
    if current != registered["sha256"] or registration.source_sha256 != registered["sha256"]:
        raise ProtocolError(
            f"{path} was edited after {registration.id} was registered (sha256 {current[:12]}… vs registered "
            f"{registered['sha256'][:12]}…); bump the version and register it as a new hypothesis"
        )
    cfg_hash = config_hash(config)
    if cfg_hash not in {config_hash(c) for c in registration.expand_grid()}:
        raise ProtocolError(f"config {config} is not a point of the registered grid {registration.grid}")
    allowed_metrics = [registration.primary_metric, *registration.secondary_metrics]
    if metric_name not in allowed_metrics:
        raise ProtocolError(f"metric {metric_name!r} is not registered for {registration.id}: {allowed_metrics}")
    if window not in _allowed_windows(registration):
        raise ProtocolError(f"window {window!r} is not a registered window: {sorted(_allowed_windows(registration))}")
    if not data_snapshot:
        raise ValueError("data_snapshot must name the data the trial was computed from")
    df = _validate_frame(per_decision)
    values, blocks = df["value"].to_numpy(), df["block"].to_numpy()

    boot = block_bootstrap(values, blocks, n_boot=n_boot, seed=seed, alpha=alpha)
    again = block_bootstrap(values.copy(), blocks.copy(), n_boot=n_boot, seed=seed, alpha=alpha)
    if again != boot:
        raise RuntimeError("the block bootstrap did not reproduce with the same data and seed")
    direction = registration.direction_for(config)
    p_value = directional_p_value(boot.p_gt0, boot.p_lt0, direction)
    git_sha, git_dirty = git_state()
    c_hash, files = code_hash(code_files)
    if git_dirty is not False:
        _archive_code(files, ledger.path.parent / registration.id / "code" / c_hash)
    statistics, series = _computed_statistics(df)
    has_league = "league" in df.columns
    loo = _leave_one_league_out(df, n_boot, seed, alpha) if has_league else None

    computed: dict[str, Any] = {
        **statistics,
        "is_primary_metric": metric_name == registration.primary_metric,
        "window": window,
        "n_blocks": boot.n_blocks,
        "n_boot": n_boot,
        "seed": seed,
        "alpha": alpha,
        "p_gt0": boot.p_gt0,
        "p_lt0": boot.p_lt0,
        "reproduced": True,
        "git_dirty": git_dirty,
        "code_hash": c_hash,
        "code_files": [_display_path(Path(f)) for f in code_files],
        "registration_sha256": current,
        "block_series": series,
        "leave_one_league_out": loo,
        "n_leagues": int(df["league"].nunique()) if has_league else None,
    }
    caller = dict(extra or {})
    stored = {k: v for k, v in caller.items() if k not in computed}
    stored.update({f"caller_{k}": v for k, v in caller.items() if k in computed})
    stored.update(computed)
    record = TrialRecord(
        hypothesis_id=registration.id,
        family=registration.family,
        config_json=json.dumps(config, sort_keys=True, default=str),
        config_hash=cfg_hash,
        git_sha=git_sha,
        data_snapshot=data_snapshot,
        n=boot.n,
        metric_name=metric_name,
        metric_value=boot.estimate,
        ci_lo=boot.lo,
        ci_hi=boot.hi,
        p_value=p_value,
        direction=direction,
        extra_json=json.dumps(stored, sort_keys=True, default=str),
    )
    ledger.record_trial(record)
    return TrialResult(
        trial_id=record.trial_id,
        hypothesis_id=registration.id,
        family=registration.family,
        config=dict(config),
        config_hash=cfg_hash,
        metric_name=metric_name,
        n=boot.n,
        n_blocks=boot.n_blocks,
        metric_value=boot.estimate,
        ci_lo=boot.lo,
        ci_hi=boot.hi,
        p_value=p_value,
        p_gt0=boot.p_gt0,
        p_lt0=boot.p_lt0,
        direction=direction,
        git_sha=git_sha,
        data_snapshot=data_snapshot,
        run_at=record.run_at,
        extra=json.loads(record.extra_json),
    )
