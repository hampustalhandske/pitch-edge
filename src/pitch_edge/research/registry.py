"""Hypothesis registration files (`research/registry/<id>.yaml`) — evidence-ladder rung L0.

A registration states, before any outcome-bearing data is scored, what will be tested and how: the
direction, the primary metric and its kind, what would disconfirm it, the data windows, the decision
time, the multiple-testing family and the parameter grid (every grid point is one counted trial). The
core schema is the one in `.claude/skills/research-protocol/SKILL.md`; it is strict about the fields the
gates read and lenient about everything else: any other key (a study's `filters`, `strategy`,
`point_in_time`, an extra window such as `post_segment`, an `h1_predict` variant of `h1`, ...) is kept
verbatim under the model's `extensions`. `post_hoc: true` marks a hypothesis formed after seeing its
data: it may be graded, but it is never a family champion or shown as an edge.

`load(path)` parses and validates a file and remembers its path and the sha256 of its bytes, so the
ledger can later detect an edit made after registration. `expand_grid()` lists every trial config,
`direction_for(config)` gives the registered direction of one grid point, and `config_hash(config)`
gives each config a stable identity.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import re
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator

from pitch_edge.research.errors import ProtocolError

ID_PATTERN = r"^[a-z0-9][a-z0-9-]*-v[0-9]+$"
REPO_ROOT = Path(__file__).resolve().parents[3]
# Registrations dated at or after this instant must state `metric_kind`; older ones get it inferred.
METRIC_KIND_REQUIRED_FROM = datetime(2026, 10, 21, tzinfo=UTC)

Direction = Literal["positive", "negative", "two-sided"]
MetricKind = Literal["return", "probability", "score"]

_RETURN_WORDS = ("clv", "roi", "pnl", "profit", "return")
_PROBABILITY_WORDS = ("prob", "brier", "calibration", "logloss", "log_loss", "log-loss", "bits")


def _split_extensions(data: Any, core: set[str]) -> Any:
    """Move every key of a mapping that is not a core field into `extensions`."""
    if not isinstance(data, dict):
        return data
    out = {k: v for k, v in data.items() if k in core}
    extra = {k: v for k, v in data.items() if k not in core}
    if extra:
        out["extensions"] = (
            {**data.get("extensions", {}), **extra} if isinstance(data.get("extensions"), dict) else extra
        )
    return out


def infer_metric_kind(primary_metric: str) -> MetricKind:
    """Kind of a primary metric guessed from its text, for registrations that predate `metric_kind`:
    `return` when it names CLV/ROI/PnL/profit/return, `probability` when it names a probability,
    Brier score, calibration, log-loss or bits, else `return` — the conservative choice, since return
    metrics are the ones held to the deflated-Sharpe bar."""
    text = primary_metric.lower()
    if any(w in text for w in _RETURN_WORDS):
        return "return"
    if any(w in text for w in _PROBABILITY_WORDS):
        return "probability"
    return "return"


class Hypothesis(BaseModel):
    """The tested claim: `h1` (or study-specific `h1_*` variants, kept in `extensions`) and `h0` in
    words, the registered `direction` of the primary metric — `positive`, `negative`, `two-sided`, or
    `per-grid-point` when each grid point carries its own `direction` — and the result that would count
    against it."""

    model_config = ConfigDict(extra="forbid")

    h1: str | None = None
    h0: str
    direction: Literal["positive", "negative", "two-sided", "per-grid-point"]
    disconfirming: str
    extensions: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _prepare(cls, data: Any) -> Any:
        data = _split_extensions(data, set(cls.model_fields))
        if isinstance(data, dict):
            d = data.get("direction")
            if isinstance(d, str) and re.match(r"^\s*per[\s-]", d.lower()):
                data["direction"] = "per-grid-point"
        return data

    @model_validator(mode="after")
    def _has_h1(self) -> Hypothesis:
        if self.h1 is None and not any(k.startswith("h1") for k in self.extensions):
            raise ValueError("hypothesis needs an `h1` (or an `h1_*` variant)")
        return self


class Window(BaseModel):
    """Inclusive date ranges: `exploratory` may be looked at while designing; `evaluation` is scored
    once, after registration. Other windows (e.g. `post_segment`) and notes go to `extensions`."""

    model_config = ConfigDict(extra="forbid")

    exploratory: tuple[date, date]
    evaluation: tuple[date, date]
    extensions: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _prepare(cls, data: Any) -> Any:
        return _split_extensions(data, set(cls.model_fields))

    @field_validator("exploratory", "evaluation")
    @classmethod
    def _ordered(cls, v: tuple[date, date]) -> tuple[date, date]:
        if v[0] > v[1]:
            raise ValueError(f"window start {v[0]} is after its end {v[1]}")
        return v


class Registration(BaseModel):
    """One parsed registration file: strict core fields plus every other key under `extensions`.

    `metric_kind` (`return` | `probability` | `score`) decides which multiple-testing statistics apply
    (deflated Sharpe only for `return`). It is required for registrations dated from
    `METRIC_KIND_REQUIRED_FROM`; for older ones it is inferred from `primary_metric` and
    `metric_kind_inferred` is True. `single_league` declares a one-league universe, which makes the
    leave-one-league-out check not applicable.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=ID_PATTERN)
    family: str = Field(min_length=1)
    registered_at: AwareDatetime
    author: str
    question: str
    hypothesis: Hypothesis
    decision_time: str
    universe: str
    window: Window
    benchmark: str
    primary_metric: str
    metric_kind: MetricKind | None = None
    secondary_metrics: list[str] = Field(default_factory=list)
    costs: str
    grid: dict[str, list[Any]] = Field(default_factory=dict)
    min_n: int = Field(ge=1)
    literature_ceiling: str
    single_league: bool = False
    post_hoc: bool = False
    notes: str | None = None
    extensions: dict[str, Any] = Field(default_factory=dict)

    _source_path: Path | None = PrivateAttr(default=None)
    _source_sha256: str | None = PrivateAttr(default=None)
    _metric_kind_inferred: bool = PrivateAttr(default=False)

    @model_validator(mode="before")
    @classmethod
    def _prepare(cls, data: Any) -> Any:
        data = _split_extensions(data, set(cls.model_fields))
        if isinstance(data, dict) and isinstance(data.get("min_n"), str):
            m = re.match(r"^\s*(\d[\d,_]*)", data["min_n"])
            if m is None:
                raise ValueError(f"min_n must start with an integer, got {data['min_n']!r}")
            data["min_n"] = int(re.sub(r"[,_]", "", m.group(1)))
        return data

    @field_validator("grid")
    @classmethod
    def _non_empty_axes(cls, v: dict[str, list[Any]]) -> dict[str, list[Any]]:
        empty = [k for k, values in v.items() if not values]
        if empty:
            raise ValueError(f"grid axes with no values: {empty}")
        return v

    @model_validator(mode="after")
    def _metric_kind(self) -> Registration:
        if self.metric_kind is None:
            if self.registered_at >= METRIC_KIND_REQUIRED_FROM:
                raise ValueError(
                    "metric_kind (return | probability | score) is required for registrations dated from "
                    f"{METRIC_KIND_REQUIRED_FROM.date()}"
                )
            self.metric_kind = infer_metric_kind(self.primary_metric)
            self._metric_kind_inferred = True
        if self.hypothesis.direction == "per-grid-point":
            for cfg in self.expand_grid():
                self.direction_for(cfg)  # every grid point must carry a direction
        return self

    @property
    def source_path(self) -> Path | None:
        """Absolute path of the file this registration was loaded from (None if built in memory)."""
        return self._source_path

    @property
    def source_sha256(self) -> str | None:
        """sha256 of the file's bytes at load time (None if built in memory)."""
        return self._source_sha256

    @property
    def is_post_hoc(self) -> bool:
        """True when the hypothesis was formed after seeing the data it is tested on: the `post_hoc` field,
        a truthy `post_hoc`/`posthoc` key among the extensions, or the words "post hoc" (any spelling) in
        the question, notes, hypothesis or extensions of a registration that predates the field."""
        if self.post_hoc:
            return True
        if any(bool(self.extensions.get(k)) for k in ("post_hoc", "posthoc", "post-hoc")):
            return True
        text = json.dumps(
            [self.question, self.notes, self.hypothesis.model_dump(), self.extensions], default=str
        ).lower()
        return re.search(r"post[\s_-]?hoc", text) is not None

    @property
    def metric_kind_inferred(self) -> bool:
        """True when `metric_kind` was not in the file and was inferred from `primary_metric`."""
        return self._metric_kind_inferred

    def expand_grid(self) -> list[dict[str, Any]]:
        """Every trial config: the Cartesian product of the grid axes, axes in file order, values in
        listed order. An empty grid is one trial with the empty config `{}`."""
        keys = list(self.grid)
        return [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*(self.grid[k] for k in keys))]

    def direction_for(self, config: dict[str, Any]) -> Direction:
        """Registered direction of one grid point: the hypothesis direction, or — for `per-grid-point` —
        the `direction` of the (single) mapping-valued grid entry in `config` that carries one. Raises
        ProtocolError when a per-grid-point config carries none or more than one."""
        d = self.hypothesis.direction
        if d != "per-grid-point":
            return d
        found = [v["direction"] for v in config.values() if isinstance(v, dict) and "direction" in v]
        if len(found) != 1 or found[0] not in ("positive", "negative", "two-sided"):
            raise ProtocolError(f"config {config} must carry exactly one valid per-grid-point direction, found {found}")
        return found[0]  # type: ignore[no-any-return]


def sha256(path: str | Path) -> str:
    """Hex sha256 of the file's raw bytes (comments and whitespace included: any edit changes it)."""
    return hashlib.sha256(resolve_registry_path(path).read_bytes()).hexdigest()


def config_hash(config: dict[str, Any]) -> str:
    """Hex sha256 of the config's canonical JSON (sorted keys, no whitespace, non-JSON values via
    `str`), so the same config always hashes the same regardless of key order or process."""
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def relative_registry_path(path: str | Path) -> str:
    """The path as stored in the ledger: relative to the repository root when inside it (so a moved
    checkout still resolves), else absolute."""
    p = Path(path).resolve()
    try:
        return p.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(p)


def resolve_registry_path(path: str | Path) -> Path:
    """Inverse of `relative_registry_path`: a relative path is taken relative to the repository root."""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def load(path: str | Path) -> Registration:
    """Parse and validate a registration file.

    Raises pydantic.ValidationError for a schema violation and ValueError when the file name (without
    `.yaml`) differs from the `id` inside it, so `research/registry/<id>.yaml` always means what it says.
    """
    p = resolve_registry_path(path).resolve()
    raw = p.read_bytes()
    data = yaml.safe_load(raw)
    if not isinstance(data, dict):
        raise ValueError(f"{p} does not contain a YAML mapping")
    reg = Registration.model_validate(data)
    if p.stem != reg.id:
        raise ValueError(f"file name {p.name!r} does not match registration id {reg.id!r}")
    reg._source_path = p
    reg._source_sha256 = hashlib.sha256(raw).hexdigest()
    return reg
