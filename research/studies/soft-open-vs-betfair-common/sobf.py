"""Shared code for the `soft-open-vs-betfair` studies (best soft-book open price against the Betfair Exchange close).

Provides: lake pulls that record the lake build, the evaluation window and its exclusion, power de-vig of a set of
decimal odds, CLV, the per-decision frame `run_trial` takes, stability tables (calendar thirds, per division,
leave-one-division-out), the exploratory `trials.csv` writer and the selection record.

Everything except the pull is a pure function of its inputs. Closing odds and results enter only as benchmarks and
targets, never through a decision function.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pitch_edge.paper.pricing import power_devig_odds
from pitch_edge.research.stats import block_bootstrap

REPO = Path(__file__).resolve().parents[3]
OUTCOMES = ("home", "draw", "away")
EVAL_WINDOW = ("2025-07-01", "2026-10-19")
EVAL_EXCLUSION = ("2026-04-17", "2026-05-04")
MAIN_LEAGUE_SQL = "source_file LIKE 'mmz4281/%'"
N_BOOT = 2000
MIN_LEAGUE = 100


# ------------------------------------------------------------------------------------------- windows


def in_window(df: pd.DataFrame, lo: str, hi: str) -> pd.DataFrame:
    """Rows with `lo <= match_date <= hi` (inclusive ISO dates)."""
    d = pd.to_datetime(df["match_date"])
    return df[(d >= pd.Timestamp(lo)) & (d <= pd.Timestamp(hi))]


def drop_exclusion(df: pd.DataFrame, lo: str = EVAL_EXCLUSION[0], hi: str = EVAL_EXCLUSION[1]) -> pd.DataFrame:
    """Rows outside the inclusive evaluation-exclusion range (football-data has no Betfair close there)."""
    d = pd.to_datetime(df["match_date"])
    return df[~((d >= pd.Timestamp(lo)) & (d <= pd.Timestamp(hi)))]


def evaluation_frame(df: pd.DataFrame) -> pd.DataFrame:
    """The evaluation window with the exclusion range removed."""
    return drop_exclusion(in_window(df, *EVAL_WINDOW))


def season_start(match_date: pd.Series) -> pd.Series:
    """Football season label as its start year: July-December -> that year, January-June -> the year before."""
    d = pd.to_datetime(match_date)
    return pd.Series(np.where(d.dt.month >= 7, d.dt.year, d.dt.year - 1), index=match_date.index)


# --------------------------------------------------------------------------------------- odds, CLV


def odds_ok(odds: np.ndarray) -> np.ndarray:
    """Row mask: every quote in the (n, k) array is finite and a valid decimal price (> 1)."""
    o = np.asarray(odds, dtype=float)
    return np.isfinite(o).all(axis=1) & (o > 1.0).all(axis=1)


def devig_power_rows(odds: np.ndarray) -> np.ndarray:
    """Power de-vig of each row of an (n, k) decimal-odds array; returns PROBABILITIES (rows sum to 1).

    Rows with a missing or invalid quote come back as NaN. Uses `pitch_edge.paper.pricing.power_devig_odds`.
    """
    o = np.asarray(odds, dtype=float)
    out = np.full(o.shape, np.nan)
    ok = odds_ok(o)
    for i in np.flatnonzero(ok):
        out[i] = power_devig_odds(o[i].tolist())
    return out


def clv_log(odds_taken: np.ndarray, p_close: np.ndarray) -> np.ndarray:
    """Per-bet closing-line value log(odds_taken * p_close), p_close the de-vigged close probability of the side."""
    return np.log(np.asarray(odds_taken, dtype=float) * np.asarray(p_close, dtype=float))


def triple(df: pd.DataFrame, book: str, kind: str) -> np.ndarray:
    """(n, 3) float array of `{book}_{home,draw,away}_{kind}` columns."""
    return df[[f"{book}_{o}_{kind}" for o in OUTCOMES]].to_numpy(dtype=float)


def bets_frame(
    f: pd.DataFrame, j: np.ndarray, odds_taken: np.ndarray, p_close_all: np.ndarray, take: np.ndarray
) -> pd.DataFrame:
    """Bets from matches `f`: outcome index `j`, price `odds_taken`, close probabilities `p_close_all` (n, 3), mask
    `take`. Columns: match_date, division, home_team, away_team, outcome, odds, p_close, clv. Index = f's index."""
    rows = np.arange(len(f))
    pc = p_close_all[rows, j]
    b = pd.DataFrame(
        {
            "match_date": pd.to_datetime(f["match_date"]).to_numpy(),
            "division": f["division"].to_numpy(),
            "home_team": f["home_team"].to_numpy(),
            "away_team": f["away_team"].to_numpy(),
            "outcome": np.array(OUTCOMES)[j],
            "odds": odds_taken,
            "p_close": pc,
        },
        index=f.index,
    )
    b["clv"] = clv_log(b["odds"].to_numpy(), b["p_close"].to_numpy())
    return b[np.asarray(take, dtype=bool)]


# ----------------------------------------------------------------------------------- run_trial frame


def pooled_league(division: pd.Series, min_n: int = MIN_LEAGUE) -> pd.Series:
    """Division labels with divisions of fewer than `min_n` bets pooled as OTHER."""
    counts = division.value_counts()
    return pd.Series(np.where(division.map(counts) >= min_n, division, "OTHER"), index=division.index)


def per_decision(bets: pd.DataFrame) -> pd.DataFrame:
    """The `run_trial` frame: decision_id = division|date|home|away, block = match-day, value = clv, league."""
    day = pd.to_datetime(bets["match_date"]).dt.strftime("%Y-%m-%d")
    ids = bets["division"].astype(str) + "|" + day + "|" + bets["home_team"].astype(str) + "|" + bets["away_team"]
    out = pd.DataFrame(
        {
            "decision_id": ids.to_numpy(),
            "block": day.to_numpy(),
            "value": bets["clv"].to_numpy(dtype=float),
            "league": pooled_league(bets["division"]).to_numpy(),
        }
    )
    return out.sort_values(["block", "decision_id"]).reset_index(drop=True)


# --------------------------------------------------------------------------------------- statistics


def boot(values: np.ndarray, blocks: Iterable[Any], seed: int = 0) -> dict[str, float | int]:
    """Mean with a match-day block-bootstrap 95% percentile CI (2,000 resamples). Empty input gives NaNs."""
    v = np.asarray(values, dtype=float)
    b = np.asarray(list(blocks))
    if len(v) == 0:
        return {"n": 0, "mean": np.nan, "lo": np.nan, "hi": np.nan, "blocks": 0}
    if len(np.unique(b)) < 2:
        return {"n": len(v), "mean": float(v.mean()), "lo": np.nan, "hi": np.nan, "blocks": 1}
    r = block_bootstrap(v, b, n_boot=N_BOOT, seed=seed)
    return {"n": len(v), "mean": float(r.estimate), "lo": float(r.lo), "hi": float(r.hi), "blocks": int(r.n_blocks)}


def calendar_thirds(dates: pd.Series, lo: str, hi: str) -> pd.Series:
    """Label each date 1/2/3 by equal-length calendar thirds of the inclusive window [lo, hi]."""
    t0, t1 = pd.Timestamp(lo), pd.Timestamp(hi) + pd.Timedelta(days=1)
    span = (t1 - t0) / 3
    d = pd.to_datetime(dates)
    return pd.Series(np.where(d < t0 + span, 1, np.where(d < t0 + 2 * span, 2, 3)), index=dates.index)


def stability(bets: pd.DataFrame, lo: str, hi: str) -> dict[str, Any]:
    """CLV by calendar third, by division (pooled below 100 bets) and with each division left out, plus the
    registered disconfirming checks: negative sign in >= 2 of 3 thirds, and any leave-one-division-out CI that
    includes 0 (or lies below it)."""
    day = pd.to_datetime(bets["match_date"]).dt.strftime("%Y-%m-%d")
    league = pooled_league(bets["division"])
    clv = bets["clv"].to_numpy(dtype=float)
    th = calendar_thirds(bets["match_date"], lo, hi).to_numpy()
    thirds = {str(k): boot(clv[th == k], day[th == k]) for k in (1, 2, 3)}
    per_div = {
        str(k): {"n": int(m.sum()), "mean": float(clv[m].mean())}
        for k in sorted(league.unique())
        for m in [(league == k).to_numpy()]
    }
    lodo = {}
    for k in sorted(league.unique()):
        keep = (league != k).to_numpy()
        lodo[str(k)] = boot(clv[keep], day[keep])
    neg_thirds = sum(1 for t in thirds.values() if t["n"] and t["mean"] < 0)
    lodo_includes_zero = sorted(k for k, t in lodo.items() if not (t["lo"] > 0))
    return {
        "thirds": thirds,
        "per_division": per_div,
        "leave_one_division_out": lodo,
        "negative_thirds": int(neg_thirds),
        "disconfirm_two_of_three_thirds_negative": bool(neg_thirds >= 2),
        "disconfirm_lodo_ci_includes_zero_for": lodo_includes_zero,
    }


# --------------------------------------------------------------------------- guards, provenance


def assert_whitelist(cols: Iterable[str], whitelist: Iterable[str]) -> None:
    """Raise AssertionError if any column is outside the whitelist or is a close/result column."""
    wl = set(whitelist)
    bad = sorted(c for c in cols if c not in wl or "close" in c or c.startswith("ft_"))
    if bad:
        raise AssertionError(f"decision inputs outside the registered whitelist: {bad}")


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def git_sha() -> str:
    run = lambda *a: subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout.strip()  # noqa: E731
    return run("rev-parse", "HEAD") + ("-dirty" if run("status", "--porcelain") else "")


def load_env() -> None:
    """Load LAKEHOUSE_* from the repo's .env into the process environment (never printed)."""
    env = REPO / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        if line.startswith("LAKEHOUSE_") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def pull_parquets(queries: dict[str, str], out_dir: Path) -> dict[str, Any]:
    """Run each SQL in `queries` ({file stem: SELECT}) through the lake client, write `<stem>.parquet` and a
    `snapshot.json` (lake build `Lake.version`, pull time, rows, parquet sha, SQL) into `out_dir`."""
    from pitch_edge.lake.client import Lake

    load_env()
    out_dir.mkdir(parents=True, exist_ok=True)
    meta: dict[str, Any] = {"pulled_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), "files": {}}
    versions: list[str] = []
    with Lake(timeout=600.0) as lake:
        for stem, sql in queries.items():
            df = lake.sql(sql)
            path = out_dir / f"{stem}.parquet"
            df.to_parquet(path)
            versions.append(lake.version or "unversioned")
            meta["files"][stem] = {"rows": len(df), "sha": file_sha(path), "sql": sql}
            print(stem, df.shape, lake.version)
    meta["lake_version"] = versions[0] if len(set(versions)) == 1 else "+".join(sorted(set(versions)))
    (out_dir / "snapshot.json").write_text(json.dumps(meta, indent=1))
    return meta


def snapshot_label(out_dir: Path) -> str:
    """`data_snapshot` for the ledger: lake build, pull date and the parquet shas (from snapshot.json)."""
    m = json.loads((out_dir / "snapshot.json").read_text())
    shas = "+".join(v["sha"] for v in m["files"].values())
    return f"lake {m['lake_version']} pulled {m['pulled_at'][:10]} (parquet sha {shas})"


# ----------------------------------------------------------------------------- exploratory records

TRIAL_FIELDS = [
    "id", "kind", "run_at", "git_sha", "data_snapshot", "window", "candidate", "n_bets", "eligible",
    "mean_clv", "total_clv", "ci_lo", "ci_hi", "selected", "note",
]  # fmt: skip


def append_exploratory(path: Path, rows: list[dict[str, Any]]) -> None:
    """Append exploratory candidate rows (not ledger trials) to a study's `trials.csv`."""
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TRIAL_FIELDS)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in TRIAL_FIELDS})


def pick(
    results: list[dict[str, Any]], key: Callable[[dict[str, Any]], tuple[Any, ...]], min_bets: int
) -> dict[str, Any] | None:
    """Best eligible candidate: the one with at least `min_bets` bets that minimises `key`; None if no one is."""
    ok = [r for r in results if r["n_bets"] >= min_bets]
    return min(ok, key=key) if ok else None
