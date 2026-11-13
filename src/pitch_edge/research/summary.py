"""Read-only summaries of the evidence behind every strategy, for the `ask` agent and the dashboard.

- `evidence_summary(ledger)`: per multiple-testing family, each registered hypothesis with its rung on
  the `research/STANDARDS.md` ladder (`research.gates.grade`), the headline of the graded trial
  (metric, estimate, CI, n), and beside it — labelled, never graded — the evaluation-window grid point
  with the largest n when the headline was picked by lowest p among several, and every secondary
  registered segment (e.g. `post_segment`); its audit / edgelab / paper reviews with their scope; and
  the family's champion — the champion-eligible hypothesis holding the highest rung (ties: the most
  recent graded trial, else the most recent registration). A hypothesis is not champion-eligible, and
  never shown as an edge, when its registration is post-hoc or its latest audit review in scope of its
  headline is REJECTED.
- `display_floor(summary)`: "The ratchet" in STANDARDS.md — the minimum rung a strategy needs before
  its numbers may be shown as an *edge*: max(L4, the highest champion rung). While no strategy holds L4,
  nothing is shown as an edge. `L4-partial` (edgelab refused on data-fit grounds) ranks below L4.
- `load_status(ledger_path, paper_path)`: both of the above as one `StatusSnapshot`, recomputed only
  when either file (or its write-ahead log) changes on disk — what the `ask` agent and the dashboard read.
- `paper_summary(store, ledger)` / `load_paper_status(path, ledger)`: the forward paper ledger per
  strategy, taken from `paper.report.summarise` — decisions in the hypothesis' registered evaluation
  window, how many are settled against a lake-sourced close, and the mean CLV with its match-day
  block-bootstrap CI; decisions outside that window are counted as exploratory.

Nothing here writes: the ledger is read through `Ledger`'s read-only connections, grading never creates
the ledger file, and the paper ledger is opened read-only.
"""

from __future__ import annotations

import json
import math
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from pitch_edge.paper.report import BETFAIR_BENCHMARK, summarise
from pitch_edge.paper.store import PaperStore
from pitch_edge.research.gates import PAPER_MIN_SETTLED, grade, results_by_role
from pitch_edge.research.ledger import Ledger
from pitch_edge.research.registry import load, resolve_registry_path

RUNG_ORDER = ("none", "L0", "L1", "L2", "L3", "L4-partial", "L4", "L5")
BASE_DISPLAY_FLOOR = "L4"
HEADLINE_SELECTION = (
    "headline = the graded trial: the first primary-metric run, in the registered evaluation window, of the "
    "grid point with the smallest p in the registered direction (research.gates); when several grid points "
    "were run, the one with the largest n is listed beside it; secondary segments are listed but never "
    "graded; for imported study rows this need not be the study's own headline number, and below L2 it is "
    "not evidence of anything"
)
LARGEST_N_LABEL = "evaluation-window grid point with the largest n (shown for comparison; not the graded trial)"
SECONDARY_LABEL = "secondary registered segment: reported, never graded, never the headline"


def rung_rank(rung: str) -> int:
    """Position of `rung` in `RUNG_ORDER` (unknown labels rank lowest, like `none`)."""
    return RUNG_ORDER.index(rung) if rung in RUNG_ORDER else 0


def meets_floor(rung: str, floor: str) -> bool:
    """True when `rung` is at or above `floor` on the ladder."""
    return rung_rank(rung) >= rung_rank(floor)


def _num(x: Any) -> float | None:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def _compact(x: float | None) -> float | None:
    """Four significant figures, so the JSON handed to the judge stays short."""
    return None if x is None else float(f"{x:.4g}")


@dataclass(frozen=True)
class Headline:
    """The graded trial's primary result: metric name, estimate with its CI, decisions, window, config."""

    trial_id: str
    metric: str
    estimate: float | None
    ci_lo: float | None
    ci_hi: float | None
    n: int
    window: str
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "estimate": _compact(self.estimate),
            "ci95": [_compact(self.ci_lo), _compact(self.ci_hi)],
            "n": self.n,
            "window": self.window,
            "config": self.config,
        }


@dataclass(frozen=True)
class ReviewSummary:
    """One ledger review: kind (`audit` / `edgelab` / `paper`), verdict, the config `scope` it is limited
    to (None = the whole hypothesis), and whether that scope covers the headline's config."""

    kind: str
    verdict: str
    scope: dict[str, Any] | None
    applies_to_headline: bool
    recorded_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "verdict": self.verdict,
            "scope": self.scope,
            "applies_to_headline": self.applies_to_headline,
        }


@dataclass(frozen=True)
class HypothesisSummary:
    """One registered hypothesis: its rung, the graded trial's headline (None when nothing was graded),
    its reviews, and the timestamp used to break rung ties (`last_activity`)."""

    id: str
    family: str
    rung: str
    headline: Headline | None
    reviews: list[ReviewSummary]
    last_activity: pd.Timestamp
    metric_kind: str = "return"
    largest_n: Headline | None = None
    n_gradable: int = 0
    secondary_segments: list[Headline] = field(default_factory=list)
    champion_eligible: bool = True
    ineligible_reason: str | None = None

    def headline_dict(self) -> dict[str, Any] | None:
        """The headline as JSON with how it was selected (None when nothing was graded)."""
        if self.headline is None:
            return None
        how = (
            f"lowest p among {self.n_gradable} evaluation-window grid points"
            if self.n_gradable > 1
            else "the only evaluation-window result"
        )
        return {**self.headline.to_dict(), "selected_by": how}


@dataclass(frozen=True)
class FamilySummary:
    """A multiple-testing family: its hypotheses, the champion's id and rung (`best_rung`; None / `none`
    when no champion-eligible hypothesis in it clears L0), and the same for its return-kind strategies only
    (`return_champion`, `return_best_rung`) — the champion may be a predictive finding, but only return-kind
    strategies set the display floor or are ever shown as an edge."""

    family: str
    hypotheses: list[HypothesisSummary]
    best_rung: str
    champion: str | None
    return_champion: str | None = None
    return_best_rung: str = "none"


@dataclass(frozen=True)
class EvidenceSummary:
    """Every family in the ledger (sorted by name)."""

    families: list[FamilySummary]

    def hypothesis(self, hypothesis_id: str) -> HypothesisSummary:
        """The summary of `hypothesis_id`. Raises KeyError when it is not registered."""
        for fam in self.families:
            for h in fam.hypotheses:
                if h.id == hypothesis_id:
                    return h
        raise KeyError(hypothesis_id)

    def rung_of(self, hypothesis_id: str) -> str | None:
        """The rung of `hypothesis_id`, or None when it is not registered."""
        try:
            return self.hypothesis(hypothesis_id).rung
        except KeyError:
            return None

    @property
    def validated_edge_exists(self) -> bool:
        """True when at least one eligible return-kind strategy is at or above the display floor."""
        floor = display_floor(self)
        return any(_shown_as_edge(h, floor) for fam in self.families for h in fam.hypotheses)

    def to_dict(self) -> dict[str, Any]:
        """Compact JSON-ready view: the display floor, whether any validated edge exists, and per family
        the best rung, the champion and each hypothesis (rung, headline, reviews, shown_as_edge)."""
        floor = display_floor(self)
        families = []
        for fam in self.families:
            champion = None
            if fam.champion is not None:
                champion = {"id": fam.champion, "rung": self.hypothesis(fam.champion).rung}
            families.append(
                {
                    "family": fam.family,
                    "best_rung": fam.best_rung,
                    "champion": champion,
                    "return_champion": None
                    if fam.return_champion is None
                    else {"id": fam.return_champion, "rung": fam.return_best_rung},
                    "hypotheses": [
                        {
                            "id": h.id,
                            "rung": h.rung,
                            "metric_kind": h.metric_kind,
                            "shown_as_edge": _shown_as_edge(h, floor),
                            "champion_eligible": h.champion_eligible,
                            "ineligible_reason": h.ineligible_reason,
                            "headline": h.headline_dict(),
                            "largest_n_grid_point": None
                            if h.largest_n is None
                            else {"label": LARGEST_N_LABEL, **h.largest_n.to_dict()},
                            "secondary_segments": [
                                {"label": SECONDARY_LABEL, **s.to_dict()} for s in h.secondary_segments
                            ],
                            "reviews": [r.to_dict() for r in h.reviews],
                        }
                        for h in fam.hypotheses
                    ],
                }
            )
        return {
            "display_floor": floor,
            "validated_edge_exists": self.validated_edge_exists,
            "rung_order": list(RUNG_ORDER),
            "headline_selection": HEADLINE_SELECTION,
            "families": families,
        }


def display_floor(summary: EvidenceSummary) -> str:
    """The minimum rung for a strategy's numbers to be shown as an edge: max(L4, the highest return-kind
    champion rung). Predictive (probability / score) champions never move it."""
    floor = BASE_DISPLAY_FLOOR
    for fam in summary.families:
        if rung_rank(fam.return_best_rung) > rung_rank(floor):
            floor = fam.return_best_rung
    return floor


def _shown_as_edge(h: HypothesisSummary, floor: str) -> bool:
    """Only an eligible return-kind strategy at or above the floor is ever presented as an edge."""
    return h.metric_kind == "return" and h.champion_eligible and meets_floor(h.rung, floor)


def _scope_covers(scope: dict[str, Any] | None, configs: list[dict[str, Any]]) -> bool:
    if not scope:
        return True
    return any(all(k in cfg and cfg[k] == v for k, v in scope.items()) for cfg in configs)


def _registration_facts(reg: pd.Series) -> tuple[bool, str]:
    """(post-hoc?, metric_kind) of a registration row; a file that no longer loads counts as not post-hoc
    and of kind `return`, the kind held to the strictest bar."""
    try:
        registration = load(resolve_registry_path(str(reg["path"])))
    except (OSError, ValueError):
        return False, "return"
    return registration.is_post_hoc, str(registration.metric_kind)


def _headline(trial: pd.Series) -> Headline:
    extra = json.loads(trial["extra_json"]) if isinstance(trial["extra_json"], str) else {}
    return Headline(
        trial_id=str(trial["trial_id"]),
        metric=str(trial["metric_name"]),
        estimate=_num(trial["metric_value"]),
        ci_lo=_num(trial["ci_lo"]),
        ci_hi=_num(trial["ci_hi"]),
        n=int(trial["n"]),
        window=str(extra.get("window", "evaluation")),
        config=json.loads(trial["config_json"]),
    )


def _hypothesis_summary(
    ledger: Ledger, reg: pd.Series, trials: pd.DataFrame, reviews: pd.DataFrame
) -> HypothesisSummary:
    g = grade(str(reg["id"]), ledger)
    headline: Headline | None = None
    configs: list[dict[str, Any]] = []
    last_activity = pd.Timestamp(reg["recorded_at"])
    if g.trial_id is not None:
        rows = trials[trials["trial_id"] == g.trial_id]
        if not rows.empty:
            row = rows.iloc[0]
            headline = _headline(row)
            extra = json.loads(row["extra_json"]) if isinstance(row["extra_json"], str) else {}
            configs = [headline.config, extra.get("original_config") or {}]
            last_activity = pd.Timestamp(row["run_at"])
    gradable, secondary = results_by_role(str(reg["id"]), ledger)
    largest_n: Headline | None = None
    if len(gradable) > 1 and headline is not None:
        biggest = gradable.sort_values(["n", "run_at"], ascending=[False, True]).iloc[0]
        if int(biggest["n"]) > headline.n:
            largest_n = _headline(biggest)
    secondary_segments = [_headline(r) for _, r in secondary.sort_values("run_at").iterrows()]
    own = reviews[reviews["hypothesis_id"] == reg["id"]]
    review_summaries = []
    for _, r in own.iterrows():
        scope = json.loads(r["detail_json"]).get("scope") or None
        review_summaries.append(
            ReviewSummary(
                kind=str(r["kind"]),
                verdict=str(r["verdict"]),
                scope=scope,
                applies_to_headline=headline is not None and _scope_covers(scope, configs),
                recorded_at=str(r["recorded_at"])[:19],
            )
        )
    in_scope_audits = [r for r in review_summaries if r.kind == "audit" and _scope_covers(r.scope, configs)]
    reason = None
    post_hoc, kind = _registration_facts(reg)
    if post_hoc:
        reason = "post-hoc registration"
    elif in_scope_audits and in_scope_audits[-1].verdict.strip().upper() == "REJECTED":
        reason = "latest in-scope audit review is REJECTED"
    return HypothesisSummary(
        id=str(reg["id"]),
        family=str(reg["family"]),
        rung=g.rung,
        headline=headline,
        reviews=review_summaries,
        last_activity=last_activity,
        metric_kind=kind,
        largest_n=largest_n,
        n_gradable=len(gradable),
        secondary_segments=secondary_segments,
        champion_eligible=reason is None,
        ineligible_reason=reason,
    )


def _top(hyps: list[HypothesisSummary]) -> HypothesisSummary | None:
    """The highest-rung hypothesis (ties: most recent activity), or None when none clears L0."""
    if not hyps:
        return None
    top = max(hyps, key=lambda h: (rung_rank(h.rung), h.last_activity))
    return None if top.rung == "none" else top


def evidence_summary(ledger: Ledger) -> EvidenceSummary:
    """Grade every registered hypothesis in `ledger` (read-only) and group them by family."""
    regs = ledger.registrations()
    if regs.empty:
        return EvidenceSummary(families=[])
    trials = ledger.trials()
    reviews = ledger.reviews()
    by_family: dict[str, list[HypothesisSummary]] = {}
    for _, reg in regs.iterrows():
        h = _hypothesis_summary(ledger, reg, trials, reviews)
        by_family.setdefault(h.family, []).append(h)
    families = []
    for name in sorted(by_family):
        hyps = by_family[name]
        eligible = [h for h in hyps if h.champion_eligible]
        top = _top(eligible)
        top_return = _top([h for h in eligible if h.metric_kind == "return"])
        families.append(
            FamilySummary(
                family=name,
                hypotheses=hyps,
                best_rung="none" if top is None else top.rung,
                champion=None if top is None else top.id,
                return_champion=None if top_return is None else top_return.id,
                return_best_rung="none" if top_return is None else top_return.rung,
            )
        )
    return EvidenceSummary(families=families)


# --------------------------------------------------------------------------- forward paper ledger


@dataclass(frozen=True)
class PaperStrategyStatus:
    """One paper strategy's forward record, as `paper.report.summarise` segments it.

    For a registered hypothesis the counts and the CLV are the `primary` segment (decisions in the
    registered evaluation window, made after the ledger recorded the registration) and `n_exploratory`
    counts its other decisions. For an unregistered hypothesis (`counts_as_evidence` False) they are the
    `unregistered` segment. `n_decided` includes `n_void`; `n_settled` counts final settlements (lake
    closes) and `n_provisional` those with only our own snapshot. `mean_clv` (mean log of taken odds over
    the de-vigged close) and its 95% match-day block-bootstrap CI use final settlements only; the CI is
    None until there are two settled bets on two match-days."""

    strategy_id: str
    hypothesis_id: str
    counts_as_evidence: bool
    registration_cut: str
    n_decided: int
    n_void: int
    n_no_benchmark: int
    n_settled: int
    n_provisional: int
    n_unsettled: int
    n_exploratory: int
    mean_clv: float | None
    clv_ci_lo: float | None
    clv_ci_hi: float | None
    n_match_days: int

    @property
    def l5_sample_met(self) -> bool:
        return self.n_settled >= PAPER_MIN_SETTLED

    def to_dict(self) -> dict[str, Any]:
        ci = None
        if self.clv_ci_lo is not None and self.clv_ci_hi is not None:
            ci = [_compact(self.clv_ci_lo), _compact(self.clv_ci_hi)]
        return {
            "strategy_id": self.strategy_id,
            "hypothesis_id": self.hypothesis_id,
            "counts_as_evidence": self.counts_as_evidence,
            "n_decided": self.n_decided,
            "n_void": self.n_void,
            "n_settled": self.n_settled,
            "n_provisional": self.n_provisional,
            "n_exploratory": self.n_exploratory,
            "mean_clv": _compact(self.mean_clv),
            "clv_ci95": ci,
            "n_match_days": self.n_match_days,
            "l5_sample_met": self.l5_sample_met,
        }


@dataclass(frozen=True)
class PaperStatus:
    """The paper ledger's per-strategy status; `available` is False (with a `reason`) when it could not be
    read."""

    available: bool
    strategies: list[PaperStrategyStatus] = field(default_factory=list)
    reason: str | None = None

    def strategy(self, strategy_id: str) -> PaperStrategyStatus:
        """The status of `strategy_id`. Raises KeyError when it has no decisions."""
        for s in self.strategies:
            if s.strategy_id == strategy_id:
                return s
        raise KeyError(strategy_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "reason": self.reason,
            "metric": "mean log(taken odds / de-vigged lake close), 95% match-day block-bootstrap CI",
            "l5_needs": f">= {PAPER_MIN_SETTLED} settled (or >= 8 weeks) with CI lower bound > 0",
            "strategies": [s.to_dict() for s in self.strategies],
        }


def _int(row: pd.Series | None, col: str) -> int:
    return 0 if row is None else int(row[col])


def paper_summary(
    store: PaperStore, ledger: Ledger, registry_dir: Path | None = None, n_boot: int = 2000, seed: int = 0
) -> PaperStatus:
    """Per-strategy forward status of the paper ledger in `store` (see `PaperStrategyStatus`), built from
    `paper.report.summarise` with `ledger` deciding which hypotheses are registered and `registry_dir`
    (default `research/registry/`) holding their registration files."""
    rep = summarise(store, registry_dir=registry_dir, ledger=ledger, n_boot=n_boot, seed=seed)
    rep = rep[rep["benchmark"] == BETFAIR_BENCHMARK]  # Pinnacle-benchmarked decisions are a different benchmark
    statuses = []
    for (sid, hid), g in rep.groupby(["strategy_id", "hypothesis_id"], sort=True):
        by_segment = {str(r["segment"]): r for _, r in g.iterrows()}
        registered = "unregistered" not in by_segment
        main = by_segment.get("primary") if registered else by_segment["unregistered"]
        exploratory = by_segment.get("exploratory")
        statuses.append(
            PaperStrategyStatus(
                strategy_id=str(sid),
                hypothesis_id=str(hid),
                counts_as_evidence=registered,
                registration_cut=str(g["registration_cut"].iloc[0] or ""),
                n_decided=_int(main, "n_decided"),
                n_void=_int(main, "n_void"),
                n_no_benchmark=_int(main, "n_no_benchmark"),
                n_settled=_int(main, "n_final"),
                n_provisional=_int(main, "n_provisional"),
                n_unsettled=_int(main, "n_unsettled"),
                n_exploratory=_int(exploratory, "n_decided"),
                mean_clv=None if main is None else _num(main["mean_clv"]),
                clv_ci_lo=None if main is None else _num(main["clv_ci_lo"]),
                clv_ci_hi=None if main is None else _num(main["clv_ci_hi"]),
                n_match_days=_int(main, "n_match_days"),
            )
        )
    return PaperStatus(available=True, strategies=statuses)


def load_paper_status(
    path: str | Path,
    ledger: Ledger,
    registry_dir: Path | None = None,
    n_boot: int = 2000,
    seed: int = 0,
    attempts: int = 3,
    wait_s: float = 1.0,
) -> PaperStatus:
    """`paper_summary` over the paper ledger file at `path`, opened read-only. Returns an unavailable
    status (never raises, never creates the file) when the file is missing or a writer keeps it locked
    for `attempts` tries `wait_s` seconds apart."""
    try:
        store = PaperStore.open_read_only(path, attempts=attempts, wait_s=wait_s)
    except duckdb.Error as exc:
        return PaperStatus(available=False, reason=f"paper ledger not readable right now ({type(exc).__name__})")
    if store is None:
        return PaperStatus(available=False, reason="no paper ledger yet")
    with store:
        return paper_summary(store, ledger, registry_dir=registry_dir, n_boot=n_boot, seed=seed)


# --------------------------------------------------------------------------- combined, cached


@dataclass(frozen=True)
class StatusSnapshot:
    """The research ladder summary and the paper ledger's forward status, read together."""

    research: EvidenceSummary
    paper: PaperStatus

    def to_dict(self) -> dict[str, Any]:
        return {"research": self.research.to_dict(), "paper": self.paper.to_dict()}


_RESEARCH_CACHE: dict[str, tuple[tuple[Any, ...], EvidenceSummary]] = {}
_PAPER_CACHE: dict[tuple[str, str], tuple[tuple[Any, ...], PaperStatus]] = {}
_STATUS_LOCK = threading.Lock()


def _file_state(path: Path) -> tuple[Any, ...]:
    """(mtime_ns, size) of a DuckDB file and its write-ahead log, None for a missing file."""
    out: list[Any] = []
    for p in (path, path.with_name(path.name + ".wal")):
        try:
            st = p.stat()
            out.append((st.st_mtime_ns, st.st_size))
        except OSError:
            out.append(None)
    return tuple(out)


def load_status(
    ledger_path: str | Path | None = None, paper_path: str | Path | None = None, lock_timeout_s: float = 5.0
) -> StatusSnapshot:
    """Summaries of the research ledger (default `<data_dir>/research/ledger.duckdb`) and the paper ledger
    (default `$PITCH_EDGE_PAPER_DB` or `<data_dir>/paper/paper.duckdb`), both read-only. Each half is
    cached: the research summary until the ledger file changes on disk, the paper status until either
    file changes (the ledger decides which paper strategies are registered)."""
    from pitch_edge.paper.cli import paper_db_path
    from pitch_edge.research.ledger import default_ledger_path

    lp = Path(ledger_path) if ledger_path is not None else default_ledger_path()
    pp = Path(paper_path) if paper_path is not None else paper_db_path()
    with _STATUS_LOCK:
        ledger = Ledger(lp, lock_timeout_s=lock_timeout_s)
        ledger_state = _file_state(lp)
        cached_research = _RESEARCH_CACHE.get(str(lp))
        if cached_research is not None and cached_research[0] == ledger_state:
            research = cached_research[1]
        else:
            research = evidence_summary(ledger)
            _RESEARCH_CACHE[str(lp)] = (ledger_state, research)
        paper_state = ledger_state + _file_state(pp)
        cached_paper = _PAPER_CACHE.get((str(lp), str(pp)))
        if cached_paper is not None and cached_paper[0] == paper_state:
            paper = cached_paper[1]
        else:
            paper = load_paper_status(pp, ledger)
            _PAPER_CACHE[(str(lp), str(pp))] = (paper_state, paper)
        return StatusSnapshot(research=research, paper=paper)
