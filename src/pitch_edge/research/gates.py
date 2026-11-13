"""The `research/STANDARDS.md` evidence ladder (L0-L5) as explicit, ledger-driven criteria.

`grade(hypothesis_id, ledger)` evaluates every rung's criteria from what the ledger holds — the
registration rows, the trial rows (with the analysis outputs `run_trial` stores in `extra_json`), the
review rows and the ledger's hash chain — and returns a `Grade`: the highest rung cleared plus one
`Criterion` per check with its status and the number behind it. Rungs are cumulative: once a rung
fails, every criterion above it is reported as `not-evaluated` ("blocked"), never computed. Grading only
reads; it never creates or migrates the ledger file.

Statuses: `pass`; `fail`; `n/a` (the criterion does not apply, e.g. PBO without a parameter grid);
`not-evaluated` (it applies but the evidence it needs is missing, which blocks the rung exactly like a
fail); `partial` (only for an edgelab data-fit refusal, giving rung `L4-partial`).

Definitions used throughout:

- A *test* is one (hypothesis, grid point, window). Its *result of record* is the FIRST primary-metric
  run of that test; later runs of the same test are reproducibility checks only (L1), so re-running on
  new data until something passes changes nothing but the L1 verdict.
- The *family* for multiple testing is every test of every registration in the family: every registered
  grid point in the evaluation window (p = 1 while unrun) plus every test that has a result of record
  (exploratory-window results count in m with p = 1, never as discoveries).
- Only grid points in the registration's evaluation window are *gradable*. Other registered segments (e.g.
  a `post_segment` window) are secondary: they count in the family's m with their own p-value but are never
  graded and never the headline (`results_by_role` lists both). Imported study rows whose config does not
  map onto a registered grid point (`off_grid`) are never gradable either; they too count in m.
- Per grid point, the *record* is its first primary-metric row — except that when several imported study
  rows map onto the same grid point (study variants the grid does not distinguish), the record is the
  earliest one a native `run_trial` rerun reproduces, since that rerun of the registered grid point is what
  identifies which variant the grid point is. Every native row after the record, and every duplicate of
  the record's imported row, is a rerun and must reproduce it: a native record to 1e-9 on n, estimate and
  CI; an imported record on n exactly and the estimate to relative 1e-5 (the precision study CSVs keep),
  without comparing CIs (the study's bootstrap engine differs).
- The *effective* row is the record itself, unless the record lacks code provenance (a legacy native row)
  or was imported: then it is the earliest native rerun that has a code hash and reproduces the record.
  L1's clean flag and provenance, and the CI, p-value, bootstrap settings and per-block statistics used by
  L2/L3, are read from the effective row. This cannot buy a better number: the effective row must match the
  record's n and estimate (and, for a native record, its CI), the choice is the earliest qualifying rerun
  rather than any the study prefers, and for an imported record the only change is replacing the study's
  bootstrap with the harness's standard one around the same estimate.
- The *selected* test is the gradable result of record with the smallest p-value in its registered
  direction (ties: the larger effect), or the grid point named by `config_hash`.

What each rung reads:

- L0: the registration row, the file's current sha256 (and the sha256 each trial was run against), trial
  timestamps (vs the ledger's recording time, or the declared `registered_at` for imported trials), and
  the ledger hash chain.
- L1 (on the effective row): the study's `clean` flag; provenance — a clean git commit, or a `code_hash`
  whose archived code is on disk; reproduction — every rerun reproduces the record, or `run_trial`'s own
  re-run.
- L2: CI vs the registered direction, decisions >= max(min_n, 300), >= 2000 bootstrap draws, >= 30
  blocks, alpha <= 0.05.
- L3: (a) BH over the family; (b) DSR for `return`-kind metrics with N = family size and var(SR) over
  same-kind results of record (per-block-sum Sharpe ratios); (c) PBO by CSCV over every grid point's
  per-block series when the grid has >= 4 points; (d) the study's `within_ceiling` flag; (e) computed
  chronological-thirds stability, and the computed leave-one-league-out CIs — every one must exclude 0
  in the registered direction (n/a for a registered `single_league`). DSR and PBO are `n/a` for
  probability/score metrics, never "pass".
- L4: latest `audit` review in scope `SIGNED`. For `probability` / `score` metrics (predictive or
  descriptive findings) edgelab is `n/a` and L4 is the auditor's signature; L5 is `n/a` and L4 is their top
  rung. For `return` strategies: (a review whose detail carries `scope: {key: value}` applies
  only to configs with those values); latest `edgelab` review `promising`/`supported` (`not_assessable`
  is a data-fit refusal -> `L4-partial`).
- L5: latest `paper` review: >= 150 settled or >= 8 weeks, and `clv_ci_lo` > 0.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from pitch_edge.research.ledger import Ledger
from pitch_edge.research.registry import Registration, load, resolve_registry_path, sha256
from pitch_edge.research.registry import config_hash as hash_config
from pitch_edge.research.stats import benjamini_hochberg, deflated_sharpe_ratio, pbo_cscv

Status = Literal["pass", "fail", "n/a", "not-evaluated", "partial"]

RUNGS = ("L0", "L1", "L2", "L3", "L4", "L5")
MIN_DECISIONS = 300
MIN_BOOT = 2000
MIN_BLOCKS = 30
MAX_ALPHA = 0.05
BH_Q = 0.10
DSR_MIN = 0.95
PBO_MAX = 0.25
PBO_MIN_GRID = 4
PBO_SPLITS = 16
SUBPERIOD_MIN_SHARE = 2 / 3
PAPER_MIN_SETTLED = 150
PAPER_MIN_WEEKS = 8
REPRO_RTOL = 1e-9
REPRO_ATOL = 1e-12
IMPORTED_RTOL = 1e-5
EDGELAB_PASS = frozenset({"promising", "supported"})
EDGELAB_DATA_FIT_REFUSAL = frozenset({"not_assessable"})
EXPLORATORY = "exploratory"
EVALUATION = "evaluation"
SR_UNIT = "per-block sum"

CRITERIA: dict[str, tuple[str, ...]] = {
    "L0": (
        "registered",
        "file unchanged since registration",
        "registered before first trial",
        "ledger hash chain intact",
    ),
    "L1": ("clean (point-in-time + reproducible)", "provenance recorded", "reruns reproduce"),
    "L2": (
        "CI excludes 0 in registered direction",
        "decisions >= max(min_n, 300)",
        "bootstrap draws >= 2000",
        "blocks >= 30",
        "alpha <= 0.05",
    ),
    "L3": (
        "BH q < 0.10 across family",
        "deflated Sharpe >= 0.95",
        "PBO < 0.25",
        "within literature ceiling",
        "sub-period sign stability",
        "not carried by one league",
    ),
    "L4": ("skeptic-auditor signed", "edgelab grade >= promising"),
    "L5": ("paper sample (>=150 settled or >=8 weeks)", "paper CLV CI lower bound > 0"),
}


@dataclass(frozen=True)
class Criterion:
    """One check: which rung, its name, the status, the number (or text) it was decided on, the bar
    it was held to, and a short explanation."""

    rung: str
    name: str
    status: Status
    value: Any = None
    threshold: str = ""
    detail: str = ""


@dataclass(frozen=True)
class Grade:
    """Highest rung cleared (`none`, `L0`..`L5` or `L4-partial`), every criterion, the graded (effective)
    trial `trial_id`, and `record_trial_id`, the grid point's result of record it stands for (the same
    trial unless the record was reproduced by a later native rerun; see the module docstring)."""

    hypothesis_id: str
    rung: str
    criteria: list[Criterion] = field(default_factory=list)
    trial_id: str | None = None
    config_hash: str | None = None
    record_trial_id: str | None = None

    def table(self) -> pd.DataFrame:
        """One row per criterion: rung, criterion, status, value, threshold, detail."""
        return pd.DataFrame(
            [
                {
                    "rung": c.rung,
                    "criterion": c.name,
                    "status": c.status,
                    "value": c.value,
                    "threshold": c.threshold,
                    "detail": c.detail,
                }
                for c in self.criteria
            ],
            columns=["rung", "criterion", "status", "value", "threshold", "detail"],
        )


# --------------------------------------------------------------------------- ledger views


def _annotate(trials: pd.DataFrame, primary_metric: dict[str, str]) -> pd.DataFrame:
    """Parsed extra plus window / imported / is_primary columns. `primary_metric` maps hypothesis id to
    its registered primary metric, used when a row does not say whether it is the primary metric."""
    if trials.empty:
        return trials.assign(extra=[], window=[], imported=[], is_primary=[], off_grid=[], test_key=[])
    extras = [json.loads(e) if isinstance(e, str) else {} for e in trials["extra_json"]]
    is_primary = [
        bool(e["is_primary_metric"]) if "is_primary_metric" in e else primary_metric.get(h, m) == m
        for e, h, m in zip(extras, trials["hypothesis_id"], trials["metric_name"], strict=True)
    ]
    imported = [e.get("imported") is True for e in extras]
    return trials.assign(
        extra=extras,
        window=[str(e.get("window", EVALUATION)) for e in extras],
        imported=imported,
        is_primary=is_primary,
        off_grid=[imp and e.get("off_grid") is True for imp, e in zip(imported, extras, strict=True)],
        # Imported rows are told apart by the study's own config: several study variants can map onto one
        # registered grid point and are separate tests, not reruns of each other.
        test_key=[
            f"{c}:{hash_config(e.get('original_config') or {})}" if imp else str(c)
            for c, imp, e in zip(trials["config_hash"], imported, extras, strict=True)
        ],
    )


def _results_of_record(trials: pd.DataFrame) -> pd.DataFrame:
    """First primary-metric run per (hypothesis, test key, window)."""
    prim = trials[trials["is_primary"]] if not trials.empty else trials
    if prim.empty:
        return prim
    ordered = prim.sort_values(["run_at", "trial_id"])
    return ordered.groupby(["hypothesis_id", "test_key", "window"], sort=False).head(1)


def _reproduces(rerun: pd.Series, record: pd.Series) -> bool:
    """Whether `rerun` reproduces `record`: for a native record, n, estimate and CI agree to 1e-9; for an
    imported record, n agrees exactly and the estimate to the precision a study CSV keeps (relative 1e-5).
    Imported CIs are not compared: they come from the study's own bootstrap engine."""
    if int(rerun["n"]) != int(record["n"]):
        return False
    if record["imported"]:
        a, b = float(rerun["metric_value"]), float(record["metric_value"])
        return abs(a - b) <= max(IMPORTED_RTOL * abs(b), REPRO_ATOL)
    cols = ["metric_value", "ci_lo", "ci_hi"]
    return bool(
        np.allclose(
            rerun[cols].to_numpy(dtype=float),
            record[cols].to_numpy(dtype=float),
            rtol=REPRO_RTOL,
            atol=REPRO_ATOL,
            equal_nan=True,
        )
    )


def _has_code_provenance(row: pd.Series) -> bool:
    return bool(row["extra"].get("code_hash")) or (
        row["git_sha"] not in ("", "unknown") and row["extra"].get("git_dirty") is False
    )


@dataclass
class _GridPoint:
    """One registered grid point in the evaluation window: its result of record, the effective row the
    gates read (see the module docstring), the reruns checked against the record, and which of them do
    not reproduce it."""

    config_hash: str
    record: pd.Series | None = None
    effective: pd.Series | None = None
    reruns: list[pd.Series] = field(default_factory=list)
    mismatches: list[str] = field(default_factory=list)

    @property
    def consumed(self) -> set[str]:
        rows = [r for r in (self.record, self.effective, *self.reruns) if r is not None]
        return {str(r["trial_id"]) for r in rows}


def _grid_points(trials: pd.DataFrame, registration: Registration) -> list[_GridPoint]:
    """Per registered grid point: result of record, effective row, reruns (see the module docstring)."""
    eligible = (
        trials[trials["is_primary"] & (trials["window"] == EVALUATION) & ~trials["off_grid"]]
        if not trials.empty
        else trials
    )
    points = []
    for cfg in registration.expand_grid():
        gp = _GridPoint(hash_config(cfg))
        points.append(gp)
        rows = (
            eligible[eligible["config_hash"] == gp.config_hash].sort_values(["run_at", "trial_id"])
            if not eligible.empty
            else eligible
        )
        if rows.empty:
            continue
        native = [r for _, r in rows.iterrows() if not r["imported"]]
        imported = [r for _, r in rows.iterrows() if r["imported"]]
        if imported:
            reproduced = [i for i in imported if any(_reproduces(n, i) for n in native)]
            record = reproduced[0] if reproduced else imported[0]
            reruns = native + [
                i for i in imported if i["test_key"] == record["test_key"] and i["trial_id"] != record["trial_id"]
            ]
        else:
            record, reruns = native[0], native[1:]
        gp.record = record
        gp.reruns = reruns
        gp.mismatches = [str(r["trial_id"]) for r in reruns if not _reproduces(r, record)]
        effective = record
        if record["imported"] or not _has_code_provenance(record):
            for r in reruns:
                if not r["imported"] and _has_code_provenance(r) and _reproduces(r, record):
                    effective = r
                    break
        gp.effective = effective
    return points


@dataclass
class _Context:
    hypothesis_id: str
    ledger: Ledger
    registered: dict[str, Any] | None = None
    registration: Registration | None = None
    trials: pd.DataFrame | None = None  # every row of this hypothesis, annotated
    points: list[_GridPoint] = field(default_factory=list)  # this hypothesis' evaluation grid points
    point: _GridPoint | None = None  # the selected grid point
    selected: pd.Series | None = None  # its effective row
    extra: dict[str, Any] = field(default_factory=dict)
    _registrations: dict[str, Registration | None] = field(default_factory=dict)

    def registration_of(self, hypothesis_id: str, path: str) -> Registration | None:
        if hypothesis_id not in self._registrations:
            try:
                self._registrations[hypothesis_id] = load(resolve_registry_path(path))
            except (OSError, ValueError):
                self._registrations[hypothesis_id] = None
        return self._registrations[hypothesis_id]


def _c(rung: str, i: int, status: Status, value: Any = None, threshold: str = "", detail: str = "") -> Criterion:
    return Criterion(rung, CRITERIA[rung][i], status, value, threshold, detail)


def _sign(ctx: _Context) -> int:
    """+1 / -1: the sign an effect must have to agree with the selected test's registered direction."""
    assert ctx.selected is not None
    direction = ctx.selected["direction"]
    if direction == "positive":
        return 1
    if direction == "negative":
        return -1
    return 1 if float(ctx.selected["metric_value"]) >= 0 else -1


# --------------------------------------------------------------------------- L0


def _chain_criterion(ctx: _Context) -> Criterion:
    report = ctx.ledger.verify_chain()
    family = ctx.registered["family"] if ctx.registered else None
    family_ids = set(ctx.ledger.registrations(family)["id"]) if family else set()
    fam_trials = set(ctx.ledger.trials(family=family)["trial_id"]) if family else set()
    around = report.unchained_after_start
    relevant = [k for k in around.get("trials", []) if k in fam_trials]
    relevant += [k for k in around.get("registrations", []) if k in family_ids]
    relevant += around.get("reviews", [])
    legacy = sum(report.legacy_unchained.values())
    if report.broken:
        return _c("L0", 3, "fail", len(report.broken), "every chained row unchanged", "; ".join(report.broken[:3]))
    if relevant:
        return _c("L0", 3, "fail", len(relevant), "no rows written around the ledger API", f"unchained: {relevant[:3]}")
    detail = f"{report.n_chained} chained rows" + (f", {legacy} legacy rows predate the chain" if legacy else "")
    return _c("L0", 3, "pass", report.n_chained, "every chained row unchanged", detail)


def _l0(ctx: _Context) -> list[Criterion]:
    reg = ctx.ledger.registration(ctx.hypothesis_id)
    if reg is None:
        return [_c("L0", 0, "fail", None, "row in registrations", "never registered")] + [
            _c("L0", i, "not-evaluated", detail="not registered") for i in (1, 2, 3)
        ]
    ctx.registered = reg
    out = [_c("L0", 0, "pass", str(reg["recorded_at"]), "row in registrations", f"sha256 {reg['sha256'][:12]}…")]

    trials = ctx.ledger.trials(hypothesis_id=ctx.hypothesis_id)
    try:
        current = sha256(reg["path"])
    except OSError:
        out.append(_c("L0", 1, "fail", None, "sha256 == registered", f"registration file missing at {reg['path']}"))
    else:
        bar = f"== {reg['sha256'][:12]}"
        if current != reg["sha256"]:
            out.append(_c("L0", 1, "fail", current[:12], bar, "edited after registration: bump the version"))
        else:
            ctx.registration = ctx.registration_of(ctx.hypothesis_id, reg["path"])
            primary = {ctx.hypothesis_id: ctx.registration.primary_metric} if ctx.registration else {}
            ctx.trials = _annotate(trials, primary)
            run_against = (
                {e.get("registration_sha256") for e in ctx.trials["extra"]} - {None} if not trials.empty else set()
            )
            stale = sorted(s for s in run_against if s != reg["sha256"])
            if ctx.registration is None:
                out.append(_c("L0", 1, "fail", current[:12], bar, "file no longer loads"))
            elif stale:
                out.append(_c("L0", 1, "fail", stale[0][:12], bar, "trials were run against a different version"))
            else:
                out.append(_c("L0", 1, "pass", current[:12], bar))

    if trials.empty:
        out.append(_c("L0", 2, "pass", "0 trials", ">= registration time"))
    else:
        extras = [json.loads(e) for e in trials["extra_json"]]
        imported = np.array([e.get("imported") is True for e in extras])
        run_at = pd.to_datetime(trials["run_at"])
        recorded, declared = pd.Timestamp(reg["recorded_at"]), pd.Timestamp(reg["registered_at"])
        early_native = int((run_at[~imported] < recorded).sum())
        early_imported = int((run_at[imported] < declared).sum())
        ok = early_native == 0 and early_imported == 0
        out.append(
            _c(
                "L0",
                2,
                "pass" if ok else "fail",
                str(run_at.min()),
                f"native >= {recorded}, imported >= declared {declared}",
                "" if ok else f"{early_native} native and {early_imported} imported trials predate the registration",
            )
        )
    out.append(_chain_criterion(ctx))
    return out


# --------------------------------------------------------------------------- L1


def _select(ctx: _Context, wanted_hash: str | None) -> str | None:
    """Fill ctx.points/point/selected; return a reason when there is nothing to grade."""
    assert ctx.registration is not None and ctx.trials is not None
    ctx.points = _grid_points(ctx.trials, ctx.registration)
    run = [gp for gp in ctx.points if gp.effective is not None]
    if not run:
        return (
            f"no primary-metric ({ctx.registration.primary_metric!r}) result at a grid point in the evaluation window"
        )
    if wanted_hash is not None:
        run = [gp for gp in run if gp.config_hash == wanted_hash]
        if not run:
            return f"no gradable result for config_hash {wanted_hash[:12]}…"

    def rank(gp: _GridPoint) -> tuple[float, float]:
        row = gp.effective
        assert row is not None
        est = float(row["metric_value"])
        effect = -est if row["direction"] == "negative" else (abs(est) if row["direction"] == "two-sided" else est)
        p = row["p_value"]
        return (1.0 if p is None or pd.isna(p) else float(p), -effect)

    ctx.point = min(run, key=rank)
    ctx.selected = ctx.point.effective
    assert ctx.selected is not None
    ctx.extra = dict(ctx.selected["extra"])
    return None


def results_by_role(hypothesis_id: str, ledger: Ledger) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(gradable, secondary) results of `hypothesis_id`'s primary metric: gradable = the effective row of
    every evaluation-window grid point that has a result (one per grid point); secondary = the result of
    record of every other non-exploratory registered segment. Both carry the parsed `extra` and `window`
    columns; both are empty when the hypothesis is not registered or its registration no longer loads."""
    raw = ledger.trials(hypothesis_id=hypothesis_id)
    empty = _annotate(raw.iloc[0:0], {})
    reg_row = ledger.registration(hypothesis_id)
    if reg_row is None:
        return empty, empty
    try:
        registration = load(resolve_registry_path(reg_row["path"]))
    except (OSError, ValueError):
        return empty, empty
    trials = _annotate(raw, {hypothesis_id: registration.primary_metric})
    effective = [gp.effective for gp in _grid_points(trials, registration) if gp.effective is not None]
    gradable = pd.DataFrame(effective) if effective else empty
    records = _results_of_record(trials)
    secondary = records[~records["window"].isin([EVALUATION, EXPLORATORY])] if not records.empty else empty
    return gradable, secondary


def _l1(ctx: _Context) -> list[Criterion]:
    sel, extra, gp = ctx.selected, ctx.extra, ctx.point
    assert sel is not None and gp is not None and gp.record is not None
    via = "" if gp.effective is gp.record else f" (from rerun {str(sel['trial_id'])[:12]}, which reproduces the record)"
    clean = extra.get("clean")
    if clean is None:
        c0 = _c("L1", 0, "not-evaluated", None, "clean flag true", "the study recorded no `clean` flag" + via)
    else:
        c0 = _c("L1", 0, "pass" if clean is True else "fail", clean, "clean flag true", via.strip())

    bar = "clean git commit, or archived code_hash; plus snapshot and n"
    basics = bool(sel["data_snapshot"]) and int(sel["n"]) > 0
    code = extra.get("code_hash")
    archive = ctx.ledger.path.parent / ctx.hypothesis_id / "code" / str(code) if code else None
    if not basics:
        c1 = _c("L1", 1, "fail", f"snapshot {sel['data_snapshot']!r}, n {int(sel['n'])}", bar)
    elif sel["git_sha"] not in ("", "unknown") and extra.get("git_dirty") is False:
        c1 = _c("L1", 1, "pass", f"git {str(sel['git_sha'])[:10]} (clean)", bar, via.strip())
    elif archive is not None and archive.is_dir():
        c1 = _c("L1", 1, "pass", f"code {str(code)[:12]} archived", bar, f"{archive}{via}")
    else:
        why = "no code_hash recorded" if not code else f"archived code missing at {archive}"
        c1 = _c("L1", 1, "fail", f"git {str(sel['git_sha'])[:10]}", bar, f"tree not clean and {why}")

    bar2 = "every rerun reproduces the result of record"
    if gp.mismatches:
        short = [m[:12] for m in gp.mismatches]
        c2 = _c("L1", 2, "fail", f"{len(gp.reruns) + 1} runs", bar2, f"reruns that do not reproduce: {short}")
    elif gp.reruns:
        c2 = _c("L1", 2, "pass", f"{len(gp.reruns) + 1} runs", bar2)
    elif extra.get("reproduced") is True:
        c2 = _c("L1", 2, "pass", "run_trial re-run", bar2)
    else:
        c2 = _c("L1", 2, "not-evaluated", 1, bar2, "no rerun and no reproduction flag")
    return [c0, c1, c2]


# --------------------------------------------------------------------------- L2


def _l2(ctx: _Context) -> list[Criterion]:
    sel, reg, extra = ctx.selected, ctx.registration, ctx.extra
    assert sel is not None and reg is not None
    lo, hi, est = float(sel["ci_lo"]), float(sel["ci_hi"]), float(sel["metric_value"])
    direction = sel["direction"]
    if direction == "positive":
        ok, bar = lo > 0, "ci_lo > 0"
    elif direction == "negative":
        ok, bar = hi < 0, "ci_hi < 0"
    else:
        ok, bar = lo > 0 or hi < 0, "ci_lo > 0 or ci_hi < 0"
    out = [
        _c(
            "L2",
            0,
            "pass" if ok else "fail",
            f"{est:.6g} [{lo:.6g}, {hi:.6g}]",
            bar,
            f"{sel['metric_name']}, {direction}",
        )
    ]
    floor = max(reg.min_n, MIN_DECISIONS)
    out.append(_c("L2", 1, "pass" if int(sel["n"]) >= floor else "fail", int(sel["n"]), f">= {floor}"))
    for i, key, test, bar_text in (
        (2, "n_boot", lambda v: int(v) >= MIN_BOOT, f">= {MIN_BOOT}"),
        (3, "n_blocks", lambda v: int(v) >= MIN_BLOCKS, f">= {MIN_BLOCKS}"),
        (4, "alpha", lambda v: float(v) <= MAX_ALPHA, f"<= {MAX_ALPHA}"),
    ):
        v = extra.get(key)
        if v is None:
            out.append(_c("L2", i, "not-evaluated", None, bar_text, f"trial recorded no {key}"))
        else:
            out.append(_c("L2", i, "pass" if test(v) else "fail", v, bar_text))
    return out


# --------------------------------------------------------------------------- L3


def _family_tests(ctx: _Context) -> pd.DataFrame:
    """One row per family test: hypothesis_id, key, window, p (1 when unrun or exploratory), metric_kind,
    sr (per-block-sum SR, if any). Each registered grid point is one test in the evaluation window (keyed
    by its config hash, p from its effective row); every primary-metric row not used by a grid point —
    off-grid or other-variant imported rows, other windows — is one more test per (key, window)."""
    assert ctx.registered is not None
    family = ctx.registered["family"]
    regs = ctx.ledger.registrations(family)
    primary: dict[str, str] = {}
    kinds: dict[str, str | None] = {}
    registrations: dict[str, Registration] = {}
    for _, r in regs.iterrows():
        registration = ctx.registration_of(r["id"], r["path"])
        kinds[r["id"]] = registration.metric_kind if registration else None
        if registration is not None:
            primary[r["id"]] = registration.primary_metric
            registrations[r["id"]] = registration
    trials = _annotate(ctx.ledger.trials(family=family), primary)

    def p_of(row: pd.Series | None) -> float:
        if row is None or row["window"] == EXPLORATORY:
            return 1.0
        p = row["p_value"]
        return 1.0 if p is None or pd.isna(p) else float(p)

    def sr_of(row: pd.Series | None) -> float | None:
        if row is None or row["extra"].get("sr_unit") != SR_UNIT:
            return None
        sr = row["extra"].get("sr")
        return None if sr is None else float(sr)

    tests: dict[tuple[str, str, str], dict[str, Any]] = {}
    consumed: set[str] = set()
    for hid, registration in registrations.items():
        own = trials[trials["hypothesis_id"] == hid] if not trials.empty else trials
        for gp in _grid_points(own, registration):
            tests[(hid, gp.config_hash, EVALUATION)] = {"p": p_of(gp.effective), "sr": sr_of(gp.effective)}
            consumed |= gp.consumed
    rest = trials[~trials["trial_id"].isin(consumed)] if not trials.empty else trials
    for _, row in _results_of_record(rest).iterrows():
        key = (row["hypothesis_id"], row["test_key"], row["window"])
        if key not in tests:
            tests[key] = {"p": p_of(row), "sr": sr_of(row)}
    return pd.DataFrame(
        [
            {"hypothesis_id": h, "config_hash": c, "window": w, "p": v["p"], "sr": v["sr"], "metric_kind": kinds.get(h)}
            for (h, c, w), v in tests.items()
        ],
        columns=["hypothesis_id", "config_hash", "window", "p", "sr", "metric_kind"],
    )


def _bh(ctx: _Context, fam: pd.DataFrame) -> Criterion:
    assert ctx.selected is not None and ctx.registered is not None
    _, q = benjamini_hochberg(fam["p"].astype(float).to_numpy(), q=BH_Q)
    key = (ctx.hypothesis_id, ctx.selected["config_hash"], ctx.selected["window"])
    mask = ((fam["hypothesis_id"] == key[0]) & (fam["config_hash"] == key[1]) & (fam["window"] == key[2])).to_numpy()
    qv = float(q[np.flatnonzero(mask)[0]])
    detail = f"p={float(ctx.selected['p_value']):.4g}, m={len(fam)} tests in family {ctx.registered['family']!r}"
    return _c("L3", 0, "pass" if qv < BH_Q else "fail", qv, f"< {BH_Q}", detail)


def _dsr(ctx: _Context, fam: pd.DataFrame) -> Criterion:
    assert ctx.registration is not None
    kind = ctx.registration.metric_kind
    bar = f">= {DSR_MIN}"
    if kind != "return":
        return _c("L3", 1, "n/a", None, bar, f"metric_kind {kind!r} (DSR applies to return metrics)")
    extra = ctx.extra
    needed = ("sr", "skew", "kurt", "sr_n")
    if any(extra.get(k) is None for k in needed) or extra.get("sr_unit") != SR_UNIT:
        return _c("L3", 1, "not-evaluated", None, bar, f"trial lacks per-block {needed}")
    n_trials = len(fam)
    srs = [float(s) for s in fam.loc[fam["metric_kind"] == kind, "sr"] if s is not None and not pd.isna(s)]
    if n_trials >= 2 and len(srs) < 2:
        return _c("L3", 1, "not-evaluated", None, bar, f"N={n_trials} tests but {len(srs)} same-kind SR estimate(s)")
    var_sr = float(np.var(srs, ddof=1)) if len(srs) >= 2 else 0.0
    try:
        dsr = deflated_sharpe_ratio(
            float(extra["sr"]), int(extra["sr_n"]), float(extra["skew"]), float(extra["kurt"]), n_trials, var_sr
        )
    except ValueError as exc:
        return _c("L3", 1, "not-evaluated", None, bar, str(exc))
    return _c(
        "L3",
        1,
        "pass" if dsr >= DSR_MIN else "fail",
        dsr,
        bar,
        f"SR={float(extra['sr']):.4g} per block, n={int(extra['sr_n'])} blocks, N={n_trials}, var(SR)={var_sr:.3g} over {len(srs)}",
    )


def _pbo(ctx: _Context) -> Criterion:
    reg, sel = ctx.registration, ctx.selected
    assert reg is not None and sel is not None
    bar = f"< {PBO_MAX}"
    if reg.metric_kind != "return":
        return _c("L3", 2, "n/a", None, bar, f"metric_kind {reg.metric_kind!r} (PBO applies to return metrics)")
    grid = reg.expand_grid()
    if len(grid) < PBO_MIN_GRID:
        return _c("L3", 2, "n/a", None, bar, f"grid has {len(grid)} point(s) (< {PBO_MIN_GRID})")
    by_hash = {gp.config_hash: gp.effective["extra"] for gp in ctx.points if gp.effective is not None}
    hashes = [hash_config(c) for c in grid]
    missing = [h for h in hashes if h not in by_hash]
    if missing:
        return _c("L3", 2, "not-evaluated", None, bar, f"{len(hashes) - len(missing)} of {len(hashes)} grid points run")
    series = [by_hash[h].get("block_series") for h in hashes]
    if any(not s or len(s[0]) != 3 for s in series):
        return _c("L3", 2, "not-evaluated", None, bar, "a grid point has no per-block [label, sum, count] series")
    labels = sorted({str(x[0]) for s in series if s for x in s})
    index = {lab: i for i, lab in enumerate(labels)}
    sums = np.zeros((len(labels), len(hashes)))
    counts = np.zeros((len(labels), len(hashes)))
    for j, s in enumerate(series):
        for lab, total, cnt in s or []:
            sums[index[str(lab)], j] = float(total)
            counts[index[str(lab)], j] = float(cnt)
    _, keep = np.unique(np.vstack([sums, counts]).T, axis=0, return_index=True)
    keep = np.sort(keep)
    if len(keep) < 2:
        return _c("L3", 2, "n/a", None, bar, f"all {len(hashes)} grid points produced identical results")
    if len(labels) < PBO_SPLITS:
        return _c("L3", 2, "not-evaluated", None, bar, f"{len(labels)} blocks (< {PBO_SPLITS})")
    res = pbo_cscv(sums[:, keep] * _sign(ctx), n_splits=PBO_SPLITS, weights=counts[:, keep])
    dup = len(hashes) - len(keep)
    return _c(
        "L3",
        2,
        "pass" if res.pbo < PBO_MAX else "fail",
        res.pbo,
        bar,
        f"CSCV, {len(labels)} blocks (union, zero-filled) x {len(keep)} configs"
        + (f" ({dup} identical dropped)" if dup else "")
        + f", {res.n_combinations} splits",
    )


def _l3(ctx: _Context) -> list[Criterion]:
    assert ctx.registration is not None
    extra = ctx.extra
    fam = _family_tests(ctx)
    out = [_bh(ctx, fam), _dsr(ctx, fam), _pbo(ctx)]

    within = extra.get("within_ceiling")
    if within is None:
        out.append(_c("L3", 3, "not-evaluated", None, "within_ceiling true", "the study recorded no ceiling check"))
    else:
        ok = within is True
        out.append(
            _c("L3", 3, "pass" if ok else "fail", within, "within_ceiling true", "" if ok else "leakage suspect")
        )

    sign = _sign(ctx)
    subs = extra.get("subperiod_estimates")
    if not subs:
        out.append(_c("L3", 4, "not-evaluated", None, ">= 2 of 3 agree", "no sub-period estimates"))
    else:
        agree = sum(1 for s in subs if s is not None and float(s) * sign > 0)
        ok = agree / len(subs) >= SUBPERIOD_MIN_SHARE - 1e-12
        shown = ", ".join("None" if s is None else f"{float(s):.4g}" for s in subs)
        out.append(_c("L3", 4, "pass" if ok else "fail", f"{agree} of {len(subs)}", ">= 2 of 3 agree", f"[{shown}]"))

    bar = "95% CI excludes 0 with any one league removed"
    loo = extra.get("leave_one_league_out")
    if ctx.registration.single_league:
        out.append(_c("L3", 5, "n/a", None, bar, "registered as a single-league universe"))
    elif not isinstance(loo, dict) or len(loo) < 2:
        why = (
            "not computed: pass a `league` column to run_trial"
            if loo is None
            else "fewer than 2 leagues in the data; declare `single_league: true` in the registration instead"
        )
        out.append(_c("L3", 5, "not-evaluated", None, bar, why))
    else:

        def excludes_zero(v: Any) -> bool:
            if not isinstance(v, dict) or v.get("lo") is None or v.get("hi") is None:
                return False
            return float(v["lo"]) > 0 if sign > 0 else float(v["hi"]) < 0

        bad = sorted(k for k, v in loo.items() if not excludes_zero(v))
        shown = "; ".join(
            f"without {k}: {v['est']:.4g} [{v['lo']:.4g}, {v['hi']:.4g}]"
            for k, v in loo.items()
            if k in bad and isinstance(v, dict) and v.get("lo") is not None
        )
        out.append(
            _c(
                "L3",
                5,
                "fail" if bad else "pass",
                f"{len(loo) - len(bad)} of {len(loo)}",
                bar,
                f"carried by {bad}: {shown}" if bad else "",
            )
        )
    return out


# --------------------------------------------------------------------------- L4 / L5


def _latest_review(reviews: pd.DataFrame, kind: str) -> pd.Series | None:
    sub = reviews[reviews["kind"] == kind]
    return None if sub.empty else sub.iloc[-1]


def _in_scope(review: pd.Series, ctx: _Context) -> bool:
    """A review applies when it has no `scope`, or every key/value of its `scope` mapping appears in the
    selected trial's config (or the study's original config, for imported trials)."""
    scope = json.loads(review["detail_json"]).get("scope")
    if not scope:
        return True
    assert ctx.selected is not None
    configs = [json.loads(ctx.selected["config_json"]), ctx.extra.get("original_config") or {}]
    return any(all(k in cfg and cfg[k] == v for k, v in scope.items()) for cfg in configs)


def _l4(ctx: _Context) -> list[Criterion]:
    reviews = ctx.ledger.reviews(ctx.hypothesis_id)
    audits = reviews[
        [k == "audit" and _in_scope(r, ctx) for k, (_, r) in zip(reviews["kind"], reviews.iterrows(), strict=True)]
    ]
    audit = None if audits.empty else audits.iloc[-1]
    if audit is None:
        c0 = _c("L4", 0, "not-evaluated", None, "SIGNED", "no audit recorded")
    else:
        ok = str(audit["verdict"]).strip().upper() == "SIGNED"
        c0 = _c("L4", 0, "pass" if ok else "fail", audit["verdict"], "SIGNED")
    edgelab = _latest_review(reviews, "edgelab")
    bar = "promising or supported"
    assert ctx.registration is not None
    if ctx.registration.metric_kind != "return":
        c1 = _c(
            "L4", 1, "n/a", None, bar, f"metric_kind {ctx.registration.metric_kind!r}: edgelab grades strategies only"
        )
    elif edgelab is None:
        c1 = _c("L4", 1, "not-evaluated", None, bar, "no edgelab grade recorded")
    else:
        verdict = str(edgelab["verdict"]).strip().lower()
        if verdict in EDGELAB_PASS:
            c1 = _c("L4", 1, "pass", verdict, bar)
        elif verdict in EDGELAB_DATA_FIT_REFUSAL:
            c1 = _c("L4", 1, "partial", verdict, bar, "edgelab refused on data-fit grounds: L4-partial")
        else:
            c1 = _c("L4", 1, "fail", verdict, bar)
    return [c0, c1]


def _l5(ctx: _Context) -> list[Criterion]:
    bar = f">= {PAPER_MIN_SETTLED} settled or >= {PAPER_MIN_WEEKS} weeks"
    assert ctx.registration is not None
    if ctx.registration.metric_kind != "return":
        why = f"metric_kind {ctx.registration.metric_kind!r}: L4 is the top rung (L5 applies to strategies only)"
        return [_c("L5", 0, "n/a", None, bar, why), _c("L5", 1, "n/a", None, "> 0", why)]
    paper = _latest_review(ctx.ledger.reviews(ctx.hypothesis_id), "paper")
    if paper is None:
        return [
            _c("L5", 0, "not-evaluated", None, bar, "no paper review"),
            _c("L5", 1, "not-evaluated", None, "> 0", "no paper review"),
        ]
    detail = json.loads(paper["detail_json"])
    n_settled, weeks, ci_lo = detail.get("n_settled"), detail.get("weeks"), detail.get("clv_ci_lo")
    enough = (n_settled is not None and int(n_settled) >= PAPER_MIN_SETTLED) or (
        weeks is not None and float(weeks) >= PAPER_MIN_WEEKS
    )
    c0 = _c("L5", 0, "pass" if enough else "fail", f"{n_settled} settled, {weeks} weeks", bar)
    if ci_lo is None:
        c1 = _c("L5", 1, "not-evaluated", None, "> 0", "paper review has no clv_ci_lo")
    else:
        c1 = _c("L5", 1, "pass" if float(ci_lo) > 0 else "fail", float(ci_lo), "> 0", "CLV vs sharp close")
    return [c0, c1]


# --------------------------------------------------------------------------- driver


def _blocked(rung: str, reason: str) -> list[Criterion]:
    return [Criterion(rung, name, "not-evaluated", detail=reason) for name in CRITERIA[rung]]


def grade(hypothesis_id: str, ledger: Ledger, *, config_hash: str | None = None) -> Grade:
    """Grade `hypothesis_id` on the L0-L5 ladder from the ledger's contents (read-only).

    Takes the hypothesis id, the ledger, and optionally the `config_hash` of the grid point to grade
    (default: the best gradable result of record, see the module docstring). Returns a `Grade` whose
    `rung` is `none` when L0 fails, otherwise the highest consecutive rung whose criteria are all `pass`
    or `n/a` (or `L4-partial` when the only shortfall at L4 is an edgelab data-fit refusal; L5 is then
    not checked).
    """
    ctx = _Context(hypothesis_id, ledger)
    criteria: list[Criterion] = []
    reached = "none"
    for rung in RUNGS:
        if rung == "L0":
            got = _l0(ctx)
        elif rung == "L1":
            reason = _select(ctx, config_hash)
            if reason is not None:
                criteria.extend(Criterion("L1", name, "not-evaluated", detail=reason) for name in CRITERIA["L1"])
                criteria.extend(c for r in RUNGS[2:] for c in _blocked(r, "blocked: L1 not cleared"))
                break
            got = _l1(ctx)
        else:
            got = {"L2": _l2, "L3": _l3, "L4": _l4, "L5": _l5}[rung](ctx)
        criteria.extend(got)
        statuses = {c.status for c in got}
        if statuses == {"n/a"}:
            break  # a rung that does not apply is never "reached": predictive findings top out at L4
        if statuses <= {"pass", "n/a"}:
            reached = rung
            continue
        if rung == "L4" and statuses <= {"pass", "n/a", "partial"}:
            reached = "L4-partial"
        blocker = reached if reached == "L4-partial" else rung
        later = RUNGS[RUNGS.index(rung) + 1 :]
        criteria.extend(c for r in later for c in _blocked(r, f"blocked: {blocker} not cleared"))
        break
    sel = ctx.selected
    return Grade(
        hypothesis_id=hypothesis_id,
        rung=reached,
        criteria=criteria,
        trial_id=None if sel is None else str(sel["trial_id"]),
        config_hash=None if sel is None else str(sel["config_hash"]),
        record_trial_id=None if ctx.point is None or ctx.point.record is None else str(ctx.point.record["trial_id"]),
    )
