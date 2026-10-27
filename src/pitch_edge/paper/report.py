"""Per-strategy summary of the paper ledger, split into the segments the evidence ladder cares about.

Segments per strategy:

- `primary`: the hypothesis is registered in the research ledger, the decision's `decided_at` falls in the
  registration's `window.evaluation` (inclusive dates, UTC) and is after the time the *ledger* recorded the
  registration (`recorded_at`; the YAML's own `registered_at` is a claim, the ledger time is evidence).
- `exploratory`: a registered hypothesis's decisions outside that cut (e.g. the exploratory window).
- `unregistered`: the hypothesis id has no ledger registration (`control`, `pending-r1`).

Within a segment the CLV metric (mean CLV, 95% block-bootstrap CI by match-day via
`research/stats.py::block_bootstrap`, ROI) uses only *final* settlements, which are lake-sourced closes.
Provisional settlements (our own snapshot or no close) are counted and averaged separately and never enter
the metric. Void decisions and fixtures in leagues without a lake close/result source (UCL) are counted
as `n_void` / `n_no_benchmark` and excluded from every metric.

The loop runs one cycle per hour but the machine sleeps, so cycles are missed and market snapshots cannot be
backfilled. `cycle_gaps` and `window_coverage` measure that from the cycle record (`PaperStore.cycle_starts`);
the primary segment carries its `window_coverage` next to the CLV results. Gaps are reported, never filled,
and the CLV computation does not depend on them.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from pitch_edge.paper.fixtures import FD_DIVISIONS
from pitch_edge.paper.store import PaperStore, utc_now
from pitch_edge.research.stats import block_bootstrap

COLUMNS = [
    "strategy_id",
    "hypothesis_id",
    "segment",
    "n_decided",
    "n_void",
    "n_no_benchmark",
    "n_final",
    "n_provisional",
    "n_unsettled",
    "mean_clv",
    "clv_ci_lo",
    "clv_ci_hi",
    "n_match_days",
    "roi",
    "provisional_mean_clv",
    "window_coverage",
    "registration_cut",
]
CYCLE_INTERVAL = timedelta(hours=1)
GAP_FACTOR = 1.5
NO_BENCHMARK_LEAGUES = ("UCL",)


@dataclass(frozen=True)
class CycleGap:
    """Two consecutive loop cycles further apart than `GAP_FACTOR` x the interval; `missed` cycles never ran."""

    start: datetime  # the cycle before the gap
    end: datetime  # the cycle after the gap
    missed: int


def cycle_gaps(starts: list[datetime], interval: timedelta = CYCLE_INTERVAL) -> list[CycleGap]:
    """Gaps in a sorted list of cycle start times. Consecutive starts more than `GAP_FACTOR` x `interval`
    apart are a gap that missed round(gap / interval) - 1 cycles."""
    gaps = []
    for a, b in zip(starts, starts[1:], strict=False):
        if b - a > GAP_FACTOR * interval:
            gaps.append(CycleGap(a, b, round((b - a) / interval) - 1))
    return gaps


def window_coverage(
    starts: list[datetime], window_start: datetime, window_end: datetime, interval: timedelta = CYCLE_INTERVAL
) -> float:
    """Share of the `interval`-long slots from `window_start` to `window_end` that contain at least one
    cycle start. Only whole slots count; NaN when the window holds none. Cycles outside the window are ignored."""
    n_slots = int((window_end - window_start) / interval)
    if n_slots <= 0:
        return float("nan")
    hit = {int((t - window_start) / interval) for t in starts if window_start <= t < window_start + n_slots * interval}
    return len(hit) / n_slots


class LedgerLike(Protocol):
    def registration(self, hypothesis_id: str) -> dict[str, Any] | None: ...


def default_registry_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "research" / "registry"


def _evaluation_window(registry_dir: Path, hypothesis_id: str) -> tuple[Any, Any] | None:
    path = registry_dir / f"{hypothesis_id}.yaml"
    if not path.exists():
        return None
    import yaml

    data = yaml.safe_load(path.read_text()) or {}
    window = (data.get("window") or {}).get("evaluation")
    if not window or len(window) != 2:
        return None
    return pd.Timestamp(window[0]).date(), pd.Timestamp(window[1]).date()


def _league(fixture_key: str) -> str:
    return fixture_key.split("|", 1)[0]


def _segment_row(g: pd.DataFrame, n_boot: int, seed: int) -> dict[str, Any]:
    benchmarked = g[~g["void"] & ~g["no_benchmark"]]
    final = benchmarked[benchmarked["final"]].dropna(subset=["clv_log"])
    with_result = benchmarked[benchmarked["final"]].dropna(subset=["pnl"])
    prov = benchmarked[~benchmarked["final"] & benchmarked["provisional"]]
    lo = hi = np.nan
    days = final["kickoff_at"].dt.date.astype(str)
    n_days = days.nunique()
    if len(final) >= 2 and n_days >= 2:
        boot = block_bootstrap(final["clv_log"].to_numpy(), days.to_numpy(), n_boot=n_boot, seed=seed)
        lo, hi = boot.lo, boot.hi
    staked = with_result["paper_stake_units"].sum()
    return {
        "n_decided": len(g),
        "n_void": int(g["void"].sum()),
        "n_no_benchmark": int((g["no_benchmark"] & ~g["void"]).sum()),
        "n_final": len(benchmarked[benchmarked["final"]]),
        "n_provisional": len(prov),
        "n_unsettled": int((~benchmarked["final"] & ~benchmarked["provisional"]).sum()),
        "mean_clv": float(final["clv_log"].mean()) if len(final) else np.nan,
        "clv_ci_lo": lo,
        "clv_ci_hi": hi,
        "n_match_days": n_days,
        "roi": float(with_result["pnl"].sum() / staked) if staked > 0 else np.nan,
        "provisional_mean_clv": float(prov["prov_clv_log"].mean()) if prov["prov_clv_log"].notna().any() else np.nan,
    }


def summarise(
    store: PaperStore,
    *,
    registry_dir: Path | None = None,
    ledger: LedgerLike | None = None,
    n_boot: int = 2000,
    seed: int = 0,
    now: datetime | None = None,
    interval: timedelta = CYCLE_INTERVAL,
) -> pd.DataFrame:
    """One row per (strategy, segment) with columns `COLUMNS`; empty when nothing has been decided.

    `window_coverage` (primary segments only, else NaN) is the share of hours between the later of the
    evaluation window start and the ledger registration and the earlier of the window end and `now` that
    contain a loop cycle."""
    decisions = store.decisions()
    if decisions.empty:
        return pd.DataFrame(columns=COLUMNS)
    registry_dir = registry_dir or default_registry_dir()
    if ledger is None:
        from pitch_edge.research.ledger import Ledger

        ledger = Ledger()
    final = store.settlements()
    prov = store.provisional_settlements().rename(columns={"clv_log": "prov_clv_log"})[["decision_id", "prov_clv_log"]]
    df = decisions.merge(final[["decision_id", "clv_log", "pnl"]], on="decision_id", how="left")
    df = df.merge(prov, on="decision_id", how="left")
    df["final"] = df["decision_id"].isin(set(final["decision_id"]))
    df["provisional"] = df["decision_id"].isin(set(prov["decision_id"]))
    df["void"] = df["decision_id"].isin(store.voided_decision_ids())
    df["no_benchmark"] = df["fixture_key"].map(_league).isin(NO_BENCHMARK_LEAGUES)

    now = now or utc_now()
    starts = store.cycle_starts()
    cuts: dict[str, str] = {}
    coverage: dict[str, float] = {}
    segments = []
    for hyp in df["hypothesis_id"].unique():
        reg = ledger.registration(str(hyp))
        window = _evaluation_window(registry_dir, str(hyp)) if reg else None
        mask = df["hypothesis_id"] == hyp
        if reg is None or window is None:
            segments.append(pd.Series("unregistered", index=df.index[mask]))
            continue
        recorded = pd.Timestamp(reg["recorded_at"])
        recorded = recorded.tz_localize("UTC") if recorded.tzinfo is None else recorded.tz_convert("UTC")
        cuts[str(hyp)] = f"evaluation {window[0]}..{window[1]}, after ledger registration {recorded.isoformat()}"
        win_start = max(datetime.combine(window[0], time.min, tzinfo=UTC), recorded.to_pydatetime())
        win_end = min(datetime.combine(window[1], time.min, tzinfo=UTC) + timedelta(days=1), now)
        coverage[str(hyp)] = window_coverage(starts, win_start, win_end, interval)
        sub = df[mask]
        day = sub["decided_at"].dt.date
        primary = (day >= window[0]) & (day <= window[1]) & (sub["decided_at"] > recorded)
        segments.append(primary.map({True: "primary", False: "exploratory"}))
    df["segment"] = pd.concat(segments).reindex(df.index)

    rows = []
    for (sid, hyp, seg), g in df.groupby(["strategy_id", "hypothesis_id", "segment"], sort=True):
        rows.append(
            {
                "strategy_id": sid,
                "hypothesis_id": hyp,
                "segment": seg,
                **_segment_row(g, n_boot, seed),
                "window_coverage": coverage.get(str(hyp), np.nan) if seg == "primary" else np.nan,
                "registration_cut": cuts.get(str(hyp), ""),
            }
        )
    return pd.DataFrame(rows, columns=COLUMNS)


def report_notes(rep: pd.DataFrame) -> list[str]:
    """Plain-language caveats printed under the report table."""
    notes = [
        "mean_clv / CI / ROI use final settlements only (lake closes: Betfair Exchange close from fd_matches, or "
        "the last pre-kickoff Polymarket bar); provisional settlements (our own post-decision snapshot or no "
        "close) are shown separately as provisional_mean_clv and never enter the metric.",
        f"{', '.join(NO_BENCHMARK_LEAGUES)}: no lake close or result source (football-data.co.uk has no "
        f"{'/'.join(NO_BENCHMARK_LEAGUES)} division; covered: {', '.join(sorted(FD_DIVISIONS))}), so these "
        "decisions are collected but reported as no benchmark (n_no_benchmark). This is a deviation from the "
        "registered universe of bookmaker-line-structure-live-soft-vs-sharp-v1, which lists UCL.",
        "pm-vs-sharp-gap is NOT R1's registered rule: its parameters are placeholders (hypothesis 'pending-r1'), "
        "so all of its decisions are exploratory/unregistered.",
        "soft-vs-sharp-live: the registered sanity rule 'price <= the max quote for that outcome in the response' "
        "is tautological for live data (the best soft price is itself one of the response's quotes), so it never "
        "binds; only the <= 1.10 x sharp-price cap can reject a price.",
    ]
    notes.append(
        "L5 (forward) evidence is reported with its coverage: the loop runs hourly but the machine sleeps, so "
        "some hours have no cycle and their snapshots are lost. window_coverage is the share of hours in a "
        "primary segment's evaluation window with a cycle; gaps are never filled and CLV is computed on the "
        "decisions that were made. `paper status` lists the gaps."
    )
    if "window_coverage" in rep.columns:
        for r in rep[rep["segment"] == "primary"].itertuples():
            if pd.notna(r.window_coverage):
                notes.append(f"{r.strategy_id} primary: {r.window_coverage:.0%} of evaluation-window hours covered.")
    cuts = sorted({c for c in rep.get("registration_cut", pd.Series(dtype=str)) if c})
    for c in cuts:
        notes.append(f"primary segment = {c}.")
    return notes


V2_HYPOTHESIS = "bookmaker-line-structure-live-soft-vs-sharp-v2"
V2_ARMS = ("pinnacle", "r6_live_model")


def _registration_meta(registry_dir: Path, hypothesis_id: str) -> tuple[tuple[Any, Any] | None, int | None]:
    path = registry_dir / f"{hypothesis_id}.yaml"
    if not path.exists():
        return None, None
    import yaml

    data = yaml.safe_load(path.read_text()) or {}
    return _evaluation_window(registry_dir, hypothesis_id), data.get("min_n")


def v2_summary(
    store: PaperStore,
    *,
    registry_dir: Path | None = None,
    ledger: LedgerLike | None = None,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """The v2 two-arm report: per arm (primary segment, final lake closes only) n decided / final, mean CLV
    with a match-day block-bootstrap CI, progress against `min_n`, decisions per week and the share of bets at
    the informative books (betvictor, betfair_sb_uk); and the paired per-fixture difference model - pinnacle
    over fixtures where either arm bet (an arm that did not bet scores 0; a fixture whose bet has no final
    close yet is pending and left out). Void and no-benchmark (UCL) decisions are excluded.

    Returns {"arms": DataFrame, "paired": dict, "cut": str} (empty arms frame when there are no v2 decisions).
    """
    registry_dir = registry_dir or default_registry_dir()
    if ledger is None:
        from pitch_edge.research.ledger import Ledger

        ledger = Ledger()
    empty = {"arms": pd.DataFrame(), "paired": {}, "cut": ""}
    reg = ledger.registration(V2_HYPOTHESIS)
    window, min_n = _registration_meta(registry_dir, V2_HYPOTHESIS)
    if reg is None or window is None:
        return empty
    recorded = pd.Timestamp(reg["recorded_at"])
    recorded = recorded.tz_localize("UTC") if recorded.tzinfo is None else recorded.tz_convert("UTC")
    cut = f"evaluation {window[0]}..{window[1]}, after ledger registration {recorded.isoformat()}"
    d = store.decisions()
    if d.empty:
        return {**empty, "cut": cut}
    d = d[d["hypothesis_id"] == V2_HYPOTHESIS]
    day = d["decided_at"].dt.date
    d = d[(day >= window[0]) & (day <= window[1]) & (d["decided_at"] > recorded)]
    d = d[~d["decision_id"].isin(store.voided_decision_ids())]
    d = d[~d["fixture_key"].map(_league).isin(NO_BENCHMARK_LEAGUES)]
    if d.empty:
        return {**empty, "cut": cut}
    final = store.settlements()[["decision_id", "clv_log"]]
    d = d.merge(final, on="decision_id", how="left")
    d["final"] = d["decision_id"].isin(set(final["decision_id"])) & d["clv_log"].notna()
    d["arm"] = d["strategy_id"].str.removeprefix("soft-vs-sharp-live-v2-")
    import json as _json

    d["informative"] = d["params_json"].map(lambda p: bool(_json.loads(p or "{}").get("informative_book")))
    log = store.decide_log()
    rows = []
    for arm in V2_ARMS:
        g = d[d["arm"] == arm]
        f = g[g["final"]]
        lo = hi = np.nan
        days = f["kickoff_at"].dt.date.astype(str)
        if len(f) >= 2 and days.nunique() >= 2:
            b = block_bootstrap(f["clv_log"].to_numpy(), days.to_numpy(), n_boot=n_boot, seed=seed)
            lo, hi = b.lo, b.hi
        runs = log[log["strategy_id"] == f"soft-vs-sharp-live-v2-{arm}"] if not log.empty else log
        if not runs.empty:
            runs = runs[runs["decided_at"] > recorded]
        weeks = (
            max((runs["decided_at"].max() - runs["decided_at"].min()).total_seconds() / (7 * 86400), 1 / 7)
            if len(runs)
            else np.nan
        )
        rows.append(
            {
                "arm": arm,
                "n_decided": len(g),
                "n_final": len(f),
                "min_n": min_n,
                "mean_clv": float(f["clv_log"].mean()) if len(f) else np.nan,
                "clv_ci_lo": lo,
                "clv_ci_hi": hi,
                "decisions_per_week": len(g) / weeks if len(g) and np.isfinite(weeks) else np.nan,
                "share_informative": float(g["informative"].mean()) if len(g) else np.nan,
            }
        )
    by_fixture = d.pivot_table(index="fixture_key", columns="arm", values="decision_id", aggfunc="count")
    diffs, blocks, pending = [], [], 0
    for fk in by_fixture.index:
        f = d[d["fixture_key"] == fk]
        clv = {}
        ok = True
        for arm in V2_ARMS:
            a = f[f["arm"] == arm]
            if a.empty:
                clv[arm] = 0.0
            elif bool(a["final"].iloc[0]):
                clv[arm] = float(a["clv_log"].iloc[0])
            else:
                ok = False
        if not ok:
            pending += 1
            continue
        diffs.append(clv["r6_live_model"] - clv["pinnacle"])
        blocks.append(str(f["kickoff_at"].iloc[0].date()))
    paired: dict[str, Any] = {
        "n_fixtures": len(diffs),
        "n_pending": pending,
        "mean_diff": np.nan,
        "ci_lo": np.nan,
        "ci_hi": np.nan,
    }
    if diffs:
        paired["mean_diff"] = float(np.mean(diffs))
        if len(diffs) >= 2 and len(set(blocks)) >= 2:
            b = block_bootstrap(diffs, blocks, n_boot=n_boot, seed=seed)
            paired["ci_lo"], paired["ci_hi"] = b.lo, b.hi
    return {"arms": pd.DataFrame(rows), "paired": paired, "cut": cut}
