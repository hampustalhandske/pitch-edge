"""Bridge from the R2 strategy scripts to the research harness (`pitch_edge.research.run_trial`).

`decisions` turns a bets frame (one row per paper bet, indexed like the eligible-match frame it came from) into the
`per_decision` frame `run_trial` takes: decision_id = division|date|home|away, block = match-day (ISO date),
value = log(odds * p_close) (CLV), league = division with divisions of fewer than 100 bets pooled as OTHER.

`check_against_ledger` compares a freshly computed grid point with the trial the study imported into the ledger from
`trials.csv` (matched on the original config), before anything is recorded: the bet count must be equal and the mean
must agree to the 6 significant figures `trials.csv` keeps. The study's own bootstrap (multinomial match-day weights)
is not the harness's (integer block draws), so the CI bounds are compared on a tolerance of a fraction of the
half-width and reported, not required to be equal.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pitch_edge.research.ledger import Ledger
from pitch_edge.research.registry import Registration, config_hash, load
from pitch_edge.research.stats import block_bootstrap
from pitch_edge.research.trial import TrialResult, run_trial

MIN_LEAGUE = 100
MEAN_RTOL = 1e-5  # trials.csv keeps 6 significant figures
CI_TOL_HALF_WIDTHS = 0.15  # Monte-Carlo difference between the two bootstrap engines, in CI half-widths


def decisions(bets: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    """per_decision for run_trial from `bets` (columns match_date, division, odds, p_close; index into `frame`)."""
    meta = frame.loc[bets.index, ["division", "home_team", "away_team", "match_date"]]
    ids = (
        meta["division"].astype(str)
        + "|"
        + meta["match_date"].dt.strftime("%Y-%m-%d")
        + "|"
        + meta["home_team"].astype(str)
        + "|"
        + meta["away_team"].astype(str)
    )
    counts = bets["division"].value_counts()
    league = np.where(bets["division"].map(counts) >= MIN_LEAGUE, bets["division"], "OTHER")
    return (
        pd.DataFrame(
            {
                "decision_id": ids.to_numpy(),
                "block": bets["match_date"].dt.strftime("%Y-%m-%d").to_numpy(),
                "value": np.log(bets["odds"].to_numpy() * bets["p_close"].to_numpy()),
                "league": league,
            }
        )
        .sort_values(["block", "decision_id"])
        .reset_index(drop=True)
    )


def imported_row(ledger: Ledger, reg_id: str, match: Callable[[dict[str, Any]], bool], window: str) -> pd.Series | None:
    """The imported ledger trial in `window` whose original (study-side) config satisfies `match`."""
    t = ledger.trials(hypothesis_id=reg_id)
    for _, r in t.iterrows():
        e = json.loads(r["extra_json"])
        if e.get("imported") is True and e.get("window") == window and match(e.get("original_config") or {}):
            return r
    return None


def check_against_ledger(pdf: pd.DataFrame, ref: pd.Series | None, seed: int = 0) -> dict[str, Any]:
    """Compare per_decision `pdf` with the imported trial `ref`; returns n/mean/CI differences and `ok`."""
    b = block_bootstrap(pdf["value"].to_numpy(), pdf["block"].to_numpy(), n_boot=2000, seed=seed)
    out: dict[str, Any] = {"n": b.n, "mean": b.estimate, "lo": b.lo, "hi": b.hi, "n_blocks": b.n_blocks}
    if ref is None:
        out.update(ok=False, why="no imported trial found for this grid point")
        return out
    half = (float(ref["ci_hi"]) - float(ref["ci_lo"])) / 2
    d_mean = abs(b.estimate - float(ref["metric_value"]))
    d_lo, d_hi = abs(b.lo - float(ref["ci_lo"])), abs(b.hi - float(ref["ci_hi"]))
    ok_n = int(ref["n"]) == b.n
    ok_mean = d_mean <= MEAN_RTOL * abs(float(ref["metric_value"])) + 1e-12
    ok_ci = max(d_lo, d_hi) <= CI_TOL_HALF_WIDTHS * max(half, 1e-12)
    out.update(
        ref_n=int(ref["n"]),
        ref_mean=float(ref["metric_value"]),
        ref_lo=float(ref["ci_lo"]),
        ref_hi=float(ref["ci_hi"]),
        d_mean=d_mean,
        d_lo=d_lo,
        d_hi=d_hi,
        ci_diff_half_widths=max(d_lo, d_hi) / max(half, 1e-12),
        ok_n=ok_n,
        ok_mean=ok_mean,
        ok_ci=ok_ci,
        ok=ok_n and ok_mean and ok_ci,
    )
    return out


def record(
    reg: Registration,
    config: dict[str, Any],
    pdf: pd.DataFrame,
    ledger: Ledger,
    extra: dict[str, Any],
    code_files: list[Path],
    snapshot: str,
) -> TrialResult:
    """run_trial on the registration's primary metric, evaluation window, harness defaults (2000 draws, seed 0)."""
    return run_trial(
        reg, config, pdf, reg.primary_metric, ledger, snapshot, extra, window="evaluation", code_files=code_files
    )


def load_registration(repo: Path, reg_id: str) -> Registration:
    return load(repo / "research" / "registry" / f"{reg_id}.yaml")


def run_grid(
    reg_id: str,
    repo: Path,
    make_bets: Callable[[dict[str, Any]], tuple[pd.DataFrame, pd.DataFrame]],
    original_config: Callable[[dict[str, Any]], Callable[[dict[str, Any]], bool]],
    code_files: list[Path],
    snapshot: str,
    do_record: bool,
    window: str = "evaluation",
) -> list[dict[str, Any]]:
    """Score every registered grid point of `reg_id` in `window` through run_trial.

    `make_bets(config)` returns (bets, frame) for one grid point (twice-built to check determinism);
    `original_config(config)` returns the predicate that finds the study's imported trial for that point. All
    points are computed and compared with their imported trials first. Nothing is recorded unless `do_record` and
    every scoreable point reproduced (n equal, mean to 6 s.f., CI within tolerance) and was deterministic. A point
    with fewer than two match-days cannot be bootstrapped and is reported as not scoreable. Returns one summary
    dict per grid point.
    """
    reg = load_registration(repo, reg_id)
    ledger = Ledger()
    plan = []
    for cfg in reg.expand_grid():
        bets, frame = make_bets(cfg)
        bets2, _ = make_bets(cfg)
        pdf = decisions(bets, frame)
        deterministic = bool(np.array_equal(pdf["value"].to_numpy(), decisions(bets2, frame)["value"].to_numpy()))
        scoreable = pdf["block"].nunique() >= 2
        chk = check_against_ledger(pdf, imported_row(ledger, reg_id, original_config(cfg), window)) if scoreable else {}
        plan.append({"config": cfg, "pdf": pdf, "deterministic": deterministic, "scoreable": scoreable, "check": chk})
        shown = dict(chk.items())
        print(
            json.dumps(
                {"config": cfg, "n_bets": len(pdf), "deterministic": deterministic, "scoreable": scoreable, **shown},
                default=float,
            )
        )
    bad = [p["config"] for p in plan if p["scoreable"] and not (p["check"].get("ok") and p["deterministic"])]
    if bad:
        print("MISMATCH - nothing recorded:", bad)
        return plan
    if not do_record:
        print("dry run: every scoreable grid point reproduces its imported trial; nothing recorded")
        return plan
    for p in plan:
        if not p["scoreable"]:
            print("not scoreable (< 2 match-days), not recorded:", p["config"])
            continue
        chk = p["check"]
        extra = {
            "clean": p["deterministic"],
            "pit_assertion": "decisions built from *_open columns only (r2lib.assert_open_only in the study code)",
            "within_ceiling": bool(chk["mean"] < 0.02),
            "reproduces_imported_trial": {"ref_n": chk["ref_n"], "ref_mean": chk["ref_mean"]},
            "n_bets": len(p["pdf"]),
        }
        r = record(reg, p["config"], p["pdf"], ledger, extra, code_files, snapshot)
        p["result"] = r
        print(
            f"recorded {p['config']}: {r.metric_value:+.6f} [{r.ci_lo:+.6f}, {r.ci_hi:+.6f}] n={r.n} "
            f"blocks={r.n_blocks} p={r.p_value:.4g}"
        )
    return plan


__all__ = [
    "check_against_ledger",
    "config_hash",
    "decisions",
    "imported_row",
    "load_registration",
    "record",
    "run_grid",
]
