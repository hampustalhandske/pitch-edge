"""H2 runner. `--count-only` is the outcome-blind step 1; `--select-only` and `--score` follow the registration.

  arch -arm64 uv run python research/studies/soft-open-vs-betfair-late-season-stakes/run.py --count-only [--force]
  arch -arm64 uv run python research/studies/soft-open-vs-betfair-late-season-stakes/run.py --select-only [--force]
  arch -arm64 uv run python research/studies/soft-open-vs-betfair-late-season-stakes/run.py --score

`--count-only` reads only the id, past-result goal and open-price columns of `train.parquet` (no close, no
`ft_result`), counts the bets of each theta and appends them to `trials.csv` (kind `count`). `--select-only` refuses
until a count exists for the same pull, then scores the three thetas against the Pinnacle close on the exploratory
window and writes `selection.json`. `--score` refuses without a selection made on the same pull, refuses a second
counted run, then scores the selected theta once on the evaluation window through `run_trial` with config {}.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import late_stakes as H
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


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _count_rows(label: str) -> list[dict]:
    if not TRIALS.exists():
        return []
    t = pd.read_csv(TRIALS)
    return t[(t["kind"] == "count") & (t["data_snapshot"] == label)].to_dict("records")


def do_count(force: bool, note: str = "") -> None:
    label = sobf.snapshot_label(H.OUT)
    if _count_rows(label) and not force:
        sys.exit("count already recorded for this pull (use --force to append a rerun)")
    f = pd.read_parquet(H.OUT / "train.parquet", columns=H.count_columns("pinnacle"))
    f = sobf.in_window(f, *H.EXPLORATORY)
    c = H.count_bets(f)
    now, sha = _now(), sobf.git_sha()
    lit = c["literal_no_completeness_requirement"]
    rows = []
    for theta in H.THETAS:
        n = c["bets"][str(theta)]
        rows.append(
            {
                "id": H.REG_ID,
                "kind": "count",
                "run_at": now,
                "git_sha": sha,
                "data_snapshot": label,
                "window": "exploratory " + "..".join(H.EXPLORATORY),
                "candidate": json.dumps({"theta": theta}),
                "n_bets": n,
                "eligible": bool(n >= H.MIN_TRAIN_BETS),
                "note": (
                    (f"{note}; " if note else "")
                    + f"outcome-blind; last5={c['n_last5']}; last5_valid_opens={c['n_last5_valid_opens']}; "
                    f"asymmetric={c['n_asymmetric']}; asymmetric_valid_opens={c['n_asymmetric_valid_opens']}; "
                    f"eligible_matches={c['n_eligible']}; incomplete_division_seasons="
                    f"{c['n_division_seasons_incomplete']}/{c['n_division_seasons']}; "
                    f"literal_reading_bets={lit['bets'][str(theta)]}"
                ),
            }
        )
    sobf.append_exploratory(TRIALS, rows)
    print(json.dumps(c, indent=1))
    for r in rows:
        print(r["run_at"], r["candidate"], "n_bets", r["n_bets"], "eligible", r["eligible"])


def do_select(force: bool) -> None:
    label = sobf.snapshot_label(H.OUT)
    if not _count_rows(label):
        sys.exit("refusing --select-only: run --count-only on this pull first (registration step 1)")
    if SELECTION.exists() and not force:
        print(f"{SELECTION} exists; selection is deterministic, not re-run (use --force to append a rerun)")
        print(SELECTION.read_text())
        return
    train = pd.read_parquet(H.OUT / "train.parquet", columns=H.pull_columns("pinnacle"))
    sel, rows = H.select(train)
    now, sha = _now(), sobf.git_sha()
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
        sys.exit("refusing --score: the pull changed since selection; rerun pull, --count-only and --select-only")
    reg = load(sobf.REPO / "research" / "registry" / f"{H.REG_ID}.yaml")
    ledger = Ledger()
    if len(ledger.trials(hypothesis_id=H.REG_ID)):
        sys.exit("refusing --score: the ledger already holds a counted trial for this hypothesis")
    theta = float(sel["theta"])

    raw = pd.read_parquet(H.OUT / "eval.parquet", columns=H.pull_columns("betfair_ex"))
    n_raw_window = len(sobf.in_window(raw, *sobf.EVAL_WINDOW))
    f, dec, pc = H.universe(raw, "betfair_ex")  # stakes need the whole season, so the exclusion is applied after
    in_eval = sobf.evaluation_frame(f).index
    keep = f.index.isin(in_eval)
    f, dec, pc = f[keep], dec[keep], pc[keep]
    n_excluded = n_raw_window - len(sobf.evaluation_frame(raw))
    bets = H.bets_for(f, dec, pc, theta)
    if len(bets) < 2 or bets["match_date"].nunique() < 2:
        sys.exit(f"only {len(bets)} evaluation bets; nothing to score")
    n_last5 = int(dec["last5"].sum())
    n_asym = int(dec["asym"].sum())
    won = (f.loc[bets.index, "ft_result"].to_numpy() == bets["outcome"].map(H.CODE).to_numpy()).astype(float)
    roi = bets["odds"].to_numpy() * won - 1.0
    stab = sobf.stability(bets, *sobf.EVAL_WINDOW)
    mean = float(bets["clv"].mean())
    extra = {
        "selected_theta": theta,
        "selection_fallback": bool(sel["fallback"]),
        "selection_snapshot": sel["data_snapshot"],
        "selection_git_sha": sel["git_sha"],
        "pit_assertion": "stakes use results dated strictly before the match date; decide() takes only whitelisted "
        "open columns (sobf.assert_whitelist)",
        "clean": True,
        "within_ceiling": bool(mean < CEILING),
        "underpowered": bool(len(bets) < MIN_N),
        "n_bets": len(bets),
        "n_last5_matches": n_last5,
        "n_asymmetric_matches": n_asym,
        "share_last5_with_one_side_at_stake": float(n_asym / n_last5) if n_last5 else float("nan"),
        "n_window_rows": n_raw_window,
        "n_excluded_by_evaluation_exclusion": n_excluded,
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
        code_files=[Path(__file__), H.HERE / "late_stakes.py", sobf.__file__],
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
    g.add_argument("--count-only", action="store_true")
    g.add_argument("--select-only", action="store_true")
    g.add_argument("--score", action="store_true")
    ap.add_argument("--force", action="store_true", help="redo and append a rerun")
    ap.add_argument("--note", default="", help="with --count-only: text prefixed to the trials.csv note")
    a = ap.parse_args()
    if a.score:
        do_score()
    elif a.select_only:
        do_select(a.force)
    else:
        do_count(a.force, a.note)
