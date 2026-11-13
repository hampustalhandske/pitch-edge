"""H3 runner. `--select-only` chooses theta on out-of-fold exploratory predictions; `--score` is the counted run.

  arch -arm64 uv run python research/studies/soft-open-vs-betfair-gbm-close/run.py --select-only [--force]
  arch -arm64 uv run python research/studies/soft-open-vs-betfair-gbm-close/run.py --score

`--select-only` reads `train.parquet` only (walk-forward by season, Pinnacle close as target and benchmark), appends
the three candidates to `trials.csv` (exploratory, not ledger trials) and writes `selection.json`. `--score` refuses
without a `selection.json` made on the same pull, refuses a second counted run, fits the final model once on the
whole exploratory window, and scores the selected theta once on the evaluation window through `run_trial` with
config {}. It also writes the max-minus-soft-average placebo and the log-loss edge as descriptive outputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import gbm_close as H
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
    cols = H.ID_COLS + H.SOURCE_COLS + H.TRAIN_TARGET_COLS
    train = pd.read_parquet(H.OUT / "train.parquet", columns=cols)
    sel, rows = H.select(train)
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    sha = sobf.git_sha()
    sobf.append_exploratory(
        TRIALS,
        [
            {
                "id": H.REG_ID,
                "kind": "exploratory",
                "run_at": now,
                "git_sha": sha,
                "data_snapshot": label,
                "window": "exploratory out-of-fold " + "..".join(H.TRAIN_WINDOW),
                "candidate": json.dumps({"theta": r["theta"]}),
                "n_bets": r["n_bets"],
                "eligible": r["eligible"],
                "mean_clv": r["mean_clv"],
                "total_clv": r["total_clv"],
                "ci_lo": r["ci_lo"],
                "ci_hi": r["ci_hi"],
                "selected": r["theta"] == sel["theta"],
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
    theta = float(sel["theta"])

    train = pd.read_parquet(H.OUT / "train.parquet", columns=H.ID_COLS + H.SOURCE_COLS + H.TRAIN_TARGET_COLS)
    model = H.freeze(train)
    raw = pd.read_parquet(H.OUT / "eval.parquet", columns=H.ID_COLS + ["ft_result"] + H.SOURCE_COLS + H.EVAL_BENCH_COLS)
    n_raw_window = len(sobf.in_window(raw, *sobf.EVAL_WINDOW))
    ev = sobf.evaluation_frame(raw)
    f, x, p_close = H.eval_universe(ev)
    q_hat = model.predict(x)
    bets = H.bets_at(f, q_hat, p_close, theta)
    if len(bets) < 2 or bets["match_date"].nunique() < 2:
        sys.exit(f"only {len(bets)} evaluation bets; nothing to score")

    # Descriptive: the same rule with the de-vigged soft average in place of q_hat (max-odds placebo), and the
    # log-loss edge of q_hat against the Betfair close relative to that soft-average baseline, in bits.
    p_avg = sobf.devig_power_rows(sobf.triple(f, "avg", "open"))
    placebo = H.bets_at(f, p_avg, p_close, theta)
    pl = sobf.boot(placebo["clv"].to_numpy(), placebo["match_date"].dt.strftime("%Y-%m-%d"))
    dkl = H.kl_bits(p_close, p_avg) - H.kl_bits(p_close, q_hat)
    won = (f.loc[bets.index, "ft_result"].to_numpy() == bets["outcome"].map(H.CODE).to_numpy()).astype(float)
    roi = bets["odds"].to_numpy() * won - 1.0
    stab = sobf.stability(bets, *sobf.EVAL_WINDOW)
    mean = float(bets["clv"].mean())
    extra = {
        "selected_theta": theta,
        "selection_fallback": bool(sel["fallback"]),
        "selection_snapshot": sel["data_snapshot"],
        "selection_git_sha": sel["git_sha"],
        "pit_assertion": "build_features() accepts exactly the nine soft open columns and asserts the 28 registered "
        "feature names with no pinnacle/betfair/close/ft_ substring (gbm_close.assert_features)",
        "clean": True,
        "within_ceiling": bool(mean < CEILING),
        "underpowered": bool(len(bets) < MIN_N),
        "n_bets": len(bets),
        "n_train_rows_final_model": model.n_train,
        "n_eval_matches": len(f),
        "share_matches_with_bet": float(len(bets) / len(f)),
        "n_window_rows": n_raw_window,
        "n_excluded_by_evaluation_exclusion": n_raw_window - len(ev),
        "n_dropped_missing_quotes": int(len(ev) - len(f)),
        "roi_flat_stake_secondary": float(np.mean(roi)),
        "hit_rate": float(won.mean()),
        "placebo_max_minus_soft_avg": pl,
        "logloss_edge_bits_vs_soft_avg_devig": float(dkl.mean()),
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
        code_files=[Path(__file__), H.HERE / "gbm_close.py", sobf.__file__],
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
