"""H1 runner. `--select-only` chooses (theta, p_min) on the exploratory window; `--score` is the single counted run.

  arch -arm64 uv run python research/studies/soft-open-vs-betfair-fav-open/run.py --select-only [--force]
  arch -arm64 uv run python research/studies/soft-open-vs-betfair-fav-open/run.py --score

`--select-only` reads `train.parquet` only, appends the six candidates to `trials.csv` (exploratory, not ledger
trials) and writes `selection.json`. `--score` refuses without `selection.json` made on the same pull, refuses a
second counted run, then scores the selected configuration once on the evaluation window through `run_trial` with
config {} and the selected values in `extra`.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import fav_open as H
import numpy as np
import pandas as pd
import sobf

from pitch_edge.research.ledger import Ledger
from pitch_edge.research.registry import load
from pitch_edge.research.trial import run_trial

SELECTION = H.OUT / "selection.json"
TRIALS = H.HERE / "trials.csv"
CEILING = 0.02
MIN_N = 300


def do_select(force: bool) -> None:
    if SELECTION.exists() and not force:
        print(f"{SELECTION} exists; selection is deterministic, not re-run (use --force to append a rerun)")
        print(SELECTION.read_text())
        return
    label = sobf.snapshot_label(H.OUT)
    train = pd.read_parquet(H.OUT / "train.parquet", columns=H.PULL_COLS)
    sel, rows = H.select(train)
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    sha = sobf.git_sha()
    chosen = (sel["theta"], sel["p_min"])
    sobf.append_exploratory(
        TRIALS,
        [
            {
                "id": H.REG_ID,
                "kind": "exploratory",
                "run_at": now,
                "git_sha": sha,
                "data_snapshot": label,
                "window": "exploratory " + "..".join(H.EXPLORATORY),
                "candidate": json.dumps({"theta": r["theta"], "p_min": r["p_min"]}),
                "n_bets": r["n_bets"],
                "eligible": r["eligible"],
                "mean_clv": r["mean_clv"],
                "total_clv": r["total_clv"],
                "ci_lo": r["ci_lo"],
                "ci_hi": r["ci_hi"],
                "selected": (r["theta"], r["p_min"]) == chosen,
                "note": "fallback" if sel["fallback"] else "",
            }
            for r in rows
        ],
    )
    SELECTION.write_text(json.dumps({**sel, "data_snapshot": label, "selected_at": now, "git_sha": sha}, indent=1))
    print(json.dumps({k: v for k, v in sel.items() if k != "candidates"}))
    for r in rows:
        print(r)


def do_score() -> None:
    if not SELECTION.exists():
        sys.exit("refusing --score: run --select-only first (no selection.json)")
    sel = json.loads(SELECTION.read_text())
    label = sobf.snapshot_label(H.OUT)
    if sel["data_snapshot"] != label:
        sys.exit("refusing --score: the pull changed since selection; rerun pull and --select-only --force")
    reg = load(sobf.REPO / "research" / "registry" / f"{H.REG_ID}.yaml")
    ledger = Ledger()
    if len(ledger.trials(hypothesis_id=H.REG_ID)):
        sys.exit("refusing --score: the ledger already holds a counted trial for this hypothesis")
    theta, p_min = float(sel["theta"]), float(sel["p_min"])

    raw = pd.read_parquet(H.OUT / "eval.parquet", columns=H.PULL_COLS)
    n_raw_window = len(sobf.in_window(raw, *sobf.EVAL_WINDOW))
    ev = sobf.evaluation_frame(raw)
    n_excluded = n_raw_window - len(ev)
    f, dec, pc = H.universe(ev)
    bets = H.bets_for(f, dec, pc, theta, p_min)
    if len(bets) < 2 or bets["match_date"].nunique() < 2:
        sys.exit(f"only {len(bets)} evaluation bets; nothing to score")
    n_elig = int(dec["eligible"].sum())
    won = (f.loc[bets.index, "ft_result"].to_numpy() == bets["outcome"].map(H.CODE).to_numpy()).astype(float)
    roi = bets["odds"].to_numpy() * won - 1.0
    stab = sobf.stability(bets, *sobf.EVAL_WINDOW)
    mean = float(bets["clv"].mean())
    extra = {
        "selected_theta": theta,
        "selected_p_min": p_min,
        "selection_fallback": bool(sel["fallback"]),
        "selection_snapshot": sel["data_snapshot"],
        "selection_git_sha": sel["git_sha"],
        "pit_assertion": "decide() takes only the whitelisted betfair_ex_*_open and max_*_open columns (sobf.assert_whitelist)",
        "clean": True,
        "within_ceiling": bool(mean < CEILING),
        "underpowered": bool(len(bets) < MIN_N),
        "n_bets": len(bets),
        "n_eligible_matches": n_elig,
        "share_eligible_matches_with_bet": float(len(bets) / n_elig),
        "n_window_rows": n_raw_window,
        "n_excluded_by_evaluation_exclusion": n_excluded,
        "n_dropped_missing_quotes": int(len(ev) - len(f)),
        "n_dropped_data_error": int((~dec["eligible"]).sum()),
        "roi_flat_stake_secondary": float(np.mean(roi)),
        "hit_rate": float(won.mean()),
        "stability": stab,
    }
    res = run_trial(
        reg,
        {},
        sobf.per_decision(bets),
        reg.primary_metric,
        ledger,
        label,
        extra,
        window="evaluation",
        code_files=[Path(__file__), H.HERE / "fav_open.py", sobf.__file__],
    )
    out = H.OUT / "score"
    out.mkdir(exist_ok=True)
    bets.assign(roi=roi).to_parquet(out / "bets.parquet")
    (out / "summary.json").write_text(json.dumps({"trial_id": res.trial_id, **extra}, indent=1, default=float))
    print(
        f"{res.metric_value:+.6f} [{res.ci_lo:+.6f}, {res.ci_hi:+.6f}] n={res.n} blocks={res.n_blocks} p={res.p_value:.4g}"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--select-only", action="store_true")
    g.add_argument("--score", action="store_true")
    ap.add_argument("--force", action="store_true", help="with --select-only: redo and append a rerun")
    a = ap.parse_args()
    do_score() if a.score else do_select(a.force)
