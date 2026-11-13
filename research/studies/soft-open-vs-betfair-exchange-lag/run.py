"""H4 runner: `--count-test`, then `--select-only`, then `--score`, each refusing until the step before it has run on
the same pull.

  arch -arm64 uv run python research/studies/soft-open-vs-betfair-exchange-lag/run.py --count-test [--force]
  arch -arm64 uv run python research/studies/soft-open-vs-betfair-exchange-lag/run.py --select-only [--force]
  arch -arm64 uv run python research/studies/soft-open-vs-betfair-exchange-lag/run.py --score

Every command first verifies the pull (`H.load_snapshot`: complete, one versioned lake build, every fixture read by
the lake, every file matching its recorded sha) and aborts when a path row's kickoff_utc differs from the fixture's.

`--count-test` is outcome-blind: it reads only path rows stamped before kickoff, through the decision functions (no
close screen, no result; OddsPapi holds no results), and counts the bets of each (s, b) on the evaluation window (and,
for the record, the exploratory window). It appends `kind=count` rows to trials.csv and writes count.json.
`--select-only` refuses until a count exists for the same pull and the ledger holds every amendment. It then reads only
the exploratory-window chunks, scores the six candidates against the Pinnacle close, applies eligibility, tie-break
and fallback, appends `kind=exploratory` rows to trials.csv and writes selection.json. `--score` refuses without a
selection made on the same pull and refuses a second counted run. It then scores the selected (s, b) once on the
evaluation window through `run_trial` with config {} and league = raw tournament id. Alongside it reports thirds,
per-tournament and leave-one-tournament-out stability, the commission 0.05 and gross secondaries, close-screen drops
by window, invalid-price rows and the outage list.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import exchange_lag as H
import numpy as np
import pandas as pd
import sobf

from pitch_edge.research.ledger import Ledger

TRIALS = H.HERE / "trials.csv"
# 1 path reading / liveness / size / close; 2 invalid quotes (price <= 1.0); 3 readings fixed with the code;
# 4 tournament = OddsPapi tournament_id, leave-one-tournament-out on raw ids
REQUIRED_AMENDMENTS = 4


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _cand(s: float, b: float) -> str:
    return json.dumps({"s": s, "b": b})


def _wlabel(window: str) -> str:
    return f"{window} " + "..".join(H.WINDOWS[window])


# ------------------------------------------------------------------------------------------- scanning


def scan(out: Path, windows: tuple[str, ...]) -> dict[str, Any]:
    """One pass over the chunks of `windows` only, reading decision-time rows (observed_at < kickoff).

    Returns per window: the universe funnel (finished fixtures in the window; with pinnacle rows; with betfair-ex
    rows; with both), rows and invalid-price rows by book, and the decision points of the universe fixtures; plus the
    finished fixtures and their kickoff ns. Exits when any path row's kickoff_utc differs from its fixture's.
    """
    fx = H.read_fixtures(out)
    fx = fx[fx["status_name"].astype(str) == "Finished"]
    kickoff = pd.Series(fx["kickoff_ns"].to_numpy(dtype=np.int64), index=fx["fixture_id"].to_numpy())
    win_of = pd.Series(fx["window"].to_numpy(), index=fx["fixture_id"].to_numpy())
    res: dict[str, Any] = {
        w: {
            "funnel": {"fixtures": int((fx["window"] == w).sum()), "with_pinnacle": 0, "with_betfair": 0, "both": 0},
            "invalid_price_rows": dict.fromkeys(H.BOOKS, 0),
            "rows": dict.fromkeys(H.BOOKS, 0),
            "points": [],
        }
        for w in windows
    }
    for path in [p for w in windows for p in H.chunk_files(out, w)]:
        df = H.read_paths(path, H.PATH_COLS)
        ko_fx = df["fixture_id"].map(kickoff).astype("Int64")
        bad = int((ko_fx.notna() & (ko_fx != df["kickoff_ns"].astype("Int64"))).fillna(False).sum())
        if bad:
            sys.exit(f"{bad} rows in {path.name} carry a kickoff_utc that differs from oddspapi_fixtures: aborting")
        for fid, ko, g in H.iter_fixture_groups(df, kickoff):
            w = win_of[fid]
            if w not in res:
                continue
            g = g[g["observed_ns"].to_numpy(dtype=np.int64) < ko]  # decision-time rows only
            r = res[w]
            books = set(g["bookmaker"])
            for b in H.BOOKS:
                gb = g[g["bookmaker"] == b]
                r["rows"][b] += len(gb)
                r["invalid_price_rows"][b] += int((~H.valid_price(gb["price"].to_numpy(dtype=float))).sum())
            r["funnel"]["with_pinnacle"] += H.PIN in books
            r["funnel"]["with_betfair"] += H.BF in books
            if {H.PIN, H.BF} <= books:
                r["funnel"]["both"] += 1
                r["points"].append(H.decision_points(fid, ko, H.build_paths(g)))
    for w in windows:
        pts = [p for p in res[w]["points"] if len(p)]
        res[w]["points"] = pd.concat(pts, ignore_index=True) if pts else H.empty_points()
        H.assert_pit(res[w]["points"])
    return {"windows": res, "kickoff_row_mismatch": 0, "fixtures": fx, "kickoff": kickoff}


def close_groups(out: Path, frozen: H.FrozenDecisions, kickoff: pd.Series) -> Iterator[tuple[str, int, pd.DataFrame]]:
    """Pinnacle-only rows (stamped <= kickoff) of the fixtures holding a frozen bet, for `H.close_table`. Reads only
    the chunks of the windows those fixtures kick off in."""
    if not isinstance(frozen, H.FrozenDecisions):
        raise AssertionError("closes are read only for frozen decisions")
    want = set(frozen.frame["fixture_id"])
    windows = sorted(set(H.window_of(frozen.frame["kickoff_ns"].to_numpy(dtype=np.int64))) - {""})
    for path in [p for w in windows for p in H.chunk_files(out, w)]:
        df = H.read_paths(path, H.CLOSE_READ_COLS, books=[H.PIN])
        df = df[df["fixture_id"].isin(want)]
        yield from H.iter_fixture_groups(df, kickoff, books=[H.PIN])


def outages(out: Path, fixtures: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    """For each window, the days on which a book has no rows across all fixtures (reads observed_at and bookmaker of
    every chunk, nothing else)."""
    daily = []
    for path in H.chunk_files(out):
        df = H.read_paths(path, ["bookmaker", "observed_ns"])
        daily.append(H.daily_counts(df["observed_ns"].to_numpy(), df["bookmaker"].to_numpy()))
    counts = pd.concat(daily, ignore_index=True).groupby(["day", "bookmaker"], as_index=False)["n"].sum()
    kd = pd.Series(pd.to_datetime(fixtures["kickoff_ns"].to_numpy(dtype=np.int64), unit="ns").strftime("%Y-%m-%d"))
    return {w: H.outage_days(counts, kd, lo, hi) for w, (lo, hi) in H.WINDOWS.items()}


def score_candidate(
    out: Path, points: pd.DataFrame, s: float, b: float, kickoff: pd.Series, tournaments: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Decide, freeze, then read the close and score one candidate. Returns (scored bets, summary)."""
    bets, funnel = H.decide(points, s, b)
    frozen = H.freeze(bets)
    closes = H.close_table(frozen, close_groups(out, frozen, kickoff))
    kept, drops = H.attach_close(frozen, closes)
    scored = H.scored_frame(kept, tournaments)
    day = scored["match_date"].dt.strftime("%Y-%m-%d")
    net = sobf.boot(scored["clv"].to_numpy(), day)
    summary = {
        "s": s,
        "b": b,
        "funnel": funnel,
        "n_decision_bets": funnel["bets"],
        "close_screen_drops": drops,
        "n_bets": int(net["n"]),
        "eligible": bool(net["n"] >= H.MIN_TRAIN_BETS),
        "mean_clv": net["mean"],
        "ci_lo": net["lo"],
        "ci_hi": net["hi"],
        "blocks": net["blocks"],
        "total_clv": float(scored["clv"].sum()),
        "mean_clv_gross": float(scored["clv_gross"].mean()) if len(scored) else float("nan"),
        "mean_clv_c05": float(scored["clv_c05"].mean()) if len(scored) else float("nan"),
    }
    return scored, summary


# ------------------------------------------------------------------------------------------- commands


def _count_rows(label: str, trials: Path) -> list[dict[str, Any]]:
    if not trials.exists():
        return []
    t = pd.read_csv(trials)
    m = (t["kind"] == "count") & (t["data_snapshot"] == label) & t["window"].astype(str).str.startswith("evaluation")
    return t[m].to_dict("records")


def do_count(out: Path = H.OUT, trials: Path = TRIALS, force: bool = False, note: str = "") -> dict[str, Any]:
    label = H.snapshot_label(out)
    if _count_rows(label, trials) and not force:
        sys.exit("count already recorded for this pull (use --force to append a rerun)")
    sc = scan(out, ("evaluation", "exploratory"))
    now, sha = _now(), sobf.git_sha()
    report: dict[str, Any] = {"data_snapshot": label, "counted_at": now, "git_sha": sha, "windows": {}}
    rows = []
    for w in ("evaluation", "exploratory"):
        r = sc["windows"][w]
        per = {}
        for s, b in H.CANDIDATES:
            _, funnel = H.decide(r["points"], s, b)
            per[_cand(s, b)] = funnel
            rows.append(
                {
                    "id": H.REG_ID,
                    "kind": "count",
                    "run_at": now,
                    "git_sha": sha,
                    "data_snapshot": label,
                    "window": _wlabel(w),
                    "candidate": _cand(s, b),
                    "n_bets": funnel["bets"],
                    "eligible": bool(funnel["bets"] >= (H.MIN_N if w == "evaluation" else H.MIN_TRAIN_BETS)),
                    "note": (f"{note}; " if note else "")
                    + "outcome-blind, before the close screen; "
                    + f"universe={r['funnel']['both']}/{r['funnel']['fixtures']} fixtures with both books; "
                    + f"outcome_decisions={funnel['outcome_decisions']}; "
                    + f"size_fail={funnel['outcome_decisions_size_fail']}; "
                    + f"invalid_price_rows={r['invalid_price_rows']}",
                }
            )
        report["windows"][w] = {
            "universe": r["funnel"],
            "rows_before_kickoff": r["rows"],
            "invalid_price_rows": r["invalid_price_rows"],
            "candidates": per,
        }
    report["kickoff_row_mismatch"] = sc["kickoff_row_mismatch"]
    report["outages"] = outages(out, sc["fixtures"])
    sobf.append_exploratory(trials, rows)
    (out / "count.json").write_text(json.dumps(report, indent=1))
    for r in rows:
        print(r["window"], r["candidate"], "n_bets", r["n_bets"])
    return report


def _amendments(ledger: Ledger) -> int:
    rv = ledger.reviews(hypothesis_id=H.REG_ID)
    return int((rv["kind"] == "amendment").sum()) if len(rv) else 0


def do_select(
    out: Path = H.OUT, trials: Path = TRIALS, force: bool = False, ledger: Ledger | None = None
) -> dict[str, Any]:
    label = H.snapshot_label(out)
    if not _count_rows(label, trials):
        sys.exit("refusing --select-only: run --count-test on this pull first")
    ledger = ledger or Ledger()
    if _amendments(ledger) < REQUIRED_AMENDMENTS:
        sys.exit(f"refusing --select-only: the ledger holds fewer than {REQUIRED_AMENDMENTS} amendments for {H.REG_ID}")
    sel_path = out / "selection.json"
    if sel_path.exists():
        prev = json.loads(sel_path.read_text())
        if prev.get("data_snapshot") != label:
            sys.exit(
                f"refusing --select-only: selection.json was made on '{prev.get('data_snapshot')}' but the pull on "
                f"disk is '{label}'. A finished pull never changes in place, so selection.json does not belong to "
                "this pull: find out where it came from before moving it aside"
            )
        if not force:
            print(f"{sel_path} exists for this pull; selection is deterministic, not re-run (--force appends a rerun)")
            return dict(prev)
    sc = scan(out, ("exploratory",))
    pts = sc["windows"]["exploratory"]["points"]
    tour = H.tournament_table(sc["fixtures"])
    rows = [score_candidate(out, pts, s, b, sc["kickoff"], tour)[1] for s, b in H.CANDIDATES]
    sel = H.select(rows)
    now, sha = _now(), sobf.git_sha()
    sobf.append_exploratory(
        trials,
        [
            {
                "id": H.REG_ID,
                "kind": "exploratory",
                "run_at": now,
                "git_sha": sha,
                "data_snapshot": label,
                "window": _wlabel("exploratory"),
                "candidate": _cand(r["s"], r["b"]),
                "n_bets": r["n_bets"],
                "eligible": r["eligible"],
                "mean_clv": r["mean_clv"],
                "total_clv": r["total_clv"],
                "ci_lo": r["ci_lo"],
                "ci_hi": r["ci_hi"],
                "selected": (r["s"], r["b"]) == (sel["s"], sel["b"]),
                "note": ("fallback; " if sel["fallback"] else "")
                + f"net of c={H.COMMISSION}; decision_bets={r['n_decision_bets']}; "
                + f"close_drops={r['close_screen_drops']}; gross={r['mean_clv_gross']:.6f}; c05={r['mean_clv_c05']:.6f}",
            }
            for r in rows
        ],
    )
    record = {
        **sel,
        "candidates": rows,
        "train_universe": sc["windows"]["exploratory"]["funnel"],
        "train_invalid_price_rows": sc["windows"]["exploratory"]["invalid_price_rows"],
        "data_snapshot": label,
        "selected_at": now,
        "git_sha": sha,
    }
    sel_path.write_text(json.dumps(record, indent=1, default=float))
    print(json.dumps({k: v for k, v in record.items() if k != "candidates"}, default=float))
    return record


def do_score(out: Path = H.OUT, ledger: Ledger | None = None) -> None:
    from pitch_edge.research.registry import load
    from pitch_edge.research.trial import run_trial

    sel_path = out / "selection.json"
    if not sel_path.exists():
        sys.exit("refusing --score: run --select-only first (no selection.json)")
    sel = json.loads(sel_path.read_text())
    label = H.snapshot_label(out)
    snap = H.load_snapshot(out)
    if sel["data_snapshot"] != label:
        sys.exit("refusing --score: selection.json was made on a different pull than the one on disk")
    reg = load(sobf.REPO / "research" / "registry" / f"{H.REG_ID}.yaml")
    ledger = ledger or Ledger()
    if len(ledger.trials(hypothesis_id=H.REG_ID)):
        sys.exit("refusing --score: the ledger already holds a counted trial for this hypothesis")
    s, b = float(sel["s"]), float(sel["b"])

    sc = scan(out, ("evaluation",))
    ev = sc["windows"]["evaluation"]
    tour = H.tournament_table(sc["fixtures"])
    scored, summ = score_candidate(out, ev["points"], s, b, sc["kickoff"], tour)
    if len(scored) < 2 or scored["match_date"].nunique() < 2:
        sys.exit(f"only {len(scored)} evaluation bets; nothing to score")
    H.assert_pit(ev["points"])
    H.assert_pit(scored)
    checks = {
        "snapshot_verified_against_file_shas": True,  # H.load_snapshot exits otherwise
        "one_versioned_lake_build": bool(snap["consistent_lake_version"]) and snap["lake_version"] != "unversioned",
        "kickoff_row_mismatch_zero": sc["kickoff_row_mismatch"] == 0,
        "pit_asserted_on_points_and_bets": True,  # H.assert_pit raises otherwise
        "decisions_frozen_before_close": True,  # H.close_table / attach_close raise otherwise
    }
    day = scored["match_date"].dt.strftime("%Y-%m-%d")
    c05 = sobf.boot(scored["clv_c05"].to_numpy(), day)
    gross = sobf.boot(scored["clv_gross"].to_numpy(), day)
    age_min = (scored["t_ns"] - scored["bf_ts_t"]).to_numpy(dtype=float) / H.MINUTE_NS
    train_drops = {f"{c['s']},{c['b']}": c["close_screen_drops"] for c in sel["candidates"]}
    mean = float(scored["clv"].mean())
    extra = {
        "selected_s": s,
        "selected_b": b,
        "selection_fallback": bool(sel["fallback"]),
        "selection_snapshot": sel["data_snapshot"],
        "selection_git_sha": sel["git_sha"],
        "commission": H.COMMISSION,
        "stake_size_check": H.STAKE_S,
        "min_valid_price": H.MIN_VALID_PRICE,
        "pit_assertion": "exchange_lag.assert_pit checks every path point a decision used was stamped <= t (references "
        "<= t - 60 min, liveness row in [t - 60 min, t]); the close is computed only from FrozenDecisions after the "
        "decisions are fixed, from pinnacle rows only",
        "clean_checks": checks,
        "clean": all(checks.values()),
        "within_ceiling": bool(mean < H.CEILING),
        "underpowered": bool(len(scored) < H.MIN_N),
        "n_bets": int(len(scored)),
        "league": "OddsPapi tournament_id, raw (amendment 4)",
        "funnel": summ["funnel"],
        "universe": ev["funnel"],
        "invalid_price_rows_evaluation": ev["invalid_price_rows"],
        "invalid_price_rows_exploratory": sel.get("train_invalid_price_rows"),
        "close_screen_drops_evaluation": summ["close_screen_drops"],
        "close_screen_drops_exploratory_by_candidate": train_drops,
        "close_stale_any_outcome_kept": int(scored["stale_any_outcome"].sum()),
        "clv_c05_secondary": c05,
        "clv_gross_secondary": gross,
        "roi_flat_stake_secondary": None,
        "roi_note": "not computable: OddsPapi tables carry no match result",
        "median_offered_size": float(scored["bf_size"].median()),
        "betfair_quote_age_min_median": float(np.median(age_min)),
        "betfair_quote_age_share_over_30min": float((age_min > 30).mean()),
        "pinnacle_inactive_at_t": int((~scored["pin_active_t"].astype(bool)).sum()),
        "kickoff_row_mismatch": sc["kickoff_row_mismatch"],
        "fetch_status_counts": snap.get("fetch_status_counts"),
        "outages": outages(out, sc["fixtures"]),
        "stability": H.stability(scored, *H.EVALUATION),
    }
    res = run_trial(
        reg,
        {},
        H.per_decision(scored),
        reg.primary_metric,
        ledger,
        label,
        extra,
        window="evaluation",
        code_files=[Path(__file__), H.HERE / "exchange_lag.py", sobf.__file__],
    )
    dest = out / "score"
    dest.mkdir(exist_ok=True)
    scored.to_parquet(dest / "bets.parquet")
    (dest / "summary.json").write_text(json.dumps({"trial_id": res.trial_id, **extra}, indent=1, default=float))
    print(
        f"{res.metric_value:+.6f} [{res.ci_lo:+.6f}, {res.ci_hi:+.6f}] n={res.n} blocks={res.n_blocks} p={res.p_value:.4g}"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--count-test", action="store_true")
    g.add_argument("--select-only", action="store_true")
    g.add_argument("--score", action="store_true")
    ap.add_argument("--force", action="store_true", help="redo and append a rerun")
    ap.add_argument("--note", default="", help="with --count-test: text prefixed to the trials.csv note")
    a = ap.parse_args()
    if a.score:
        do_score()
    elif a.select_only:
        do_select(force=a.force)
    else:
        do_count(force=a.force, note=a.note)
