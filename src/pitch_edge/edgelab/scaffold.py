"""Write an edgelab strategy folder for a pitch-edge model forecast on binary markets.

The folder holds `strategy.py` (a `PredictionMarketStrategy` that stakes fractional Kelly of the
model's `pe_model_p` forecast against the book mid, once per market, `entry_hours_before_kickoff`
hours before kickoff, when the edge exceeds `edge_threshold`), `strategy.yaml` (pre-registered grid,
hypothesis and `parquet` data block), `strategy.json` and `notes.md`. The `grading:` policy block is
not written: append edgelab's current one with
`edgelab strategy init <folder name> --asset-class prediction_market --root <strategies root>`, which
adds it to an existing `strategy.yaml` that lacks one.

The generated rule stakes a fractional-Kelly amount on the model forecast against the market mid.
"""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path

from pitch_edge.edgelab.export import MODEL_FEATURE


class ScaffoldError(ValueError):
    """The scaffold specification is invalid."""


@dataclasses.dataclass(frozen=True)
class ScaffoldSpec:
    """Everything a scaffolded strategy fixes before any run.

    Attributes:
        slug: Short name; the folder is `pitch-edge-<slug>` (lower-case letters, digits, hyphens).
        title: Human-readable strategy name.
        rationale: Economic rationale, written before any counted run.
        data_path: Dataset folder the `parquet` provider reads (absolute, or relative to the parent
            of the strategies root).
        range_start: First day of the tested window, `YYYY-MM-DD`.
        range_end: Last day of the tested window, `YYYY-MM-DD`.
        edge_thresholds: Values of the minimum |forecast - mid| acted on. More than one is swept.
        kelly_fractions: Values of the Kelly fraction. More than one is swept.
        selected: Grid point reported as the headline, keyed by parameter name; required when any
            parameter has more than one value.
        stake_cap: Largest fraction of NAV in one market.
        entry_hours_before_kickoff: Hours before kickoff of the single decision bar; must exceed 1
            because an order fills one bar after its decision.
        leagues: League codes in `data_requirements`; the parquet provider loads every market in the
            folder regardless, so they only need to be non-empty.
        capital: Starting capital of the simulated book (required by edgelab's config).
        seed: Seed of every stochastic stage (required by edgelab's config).
        author: Author field of `strategy.yaml`.
        created: Creation date, `YYYY-MM-DD`.
    """

    slug: str
    title: str
    rationale: str
    data_path: str
    range_start: str
    range_end: str
    edge_thresholds: tuple[float, ...] = (0.03,)
    kelly_fractions: tuple[float, ...] = (0.25,)
    selected: dict[str, float] = dataclasses.field(default_factory=dict)
    stake_cap: float = 0.05
    entry_hours_before_kickoff: float = 3.0
    leagues: tuple[str, ...] = ("epl",)
    capital: float = 100000.0
    seed: int = 7
    author: str = "pitch-edge"
    created: str = "2026-01-01"

    @property
    def folder(self) -> str:
        """Folder name under edgelab's strategies root."""
        return f"pitch-edge-{self.slug}"

    @property
    def class_name(self) -> str:
        """CamelCase strategy class name derived from the slug."""
        return "PitchEdge" + "".join(p.capitalize() for p in self.slug.split("-"))

    @property
    def n_trials(self) -> int:
        """Size of the pre-registered grid, which edgelab counts as trials."""
        return len(self.edge_thresholds) * len(self.kelly_fractions)

    def validate(self) -> None:
        """Raise `ScaffoldError` if the specification cannot produce a valid strategy."""
        if not re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", self.slug):
            raise ScaffoldError("slug must be lower-case letters, digits and single hyphens.")
        if not self.rationale.strip():
            raise ScaffoldError("rationale must be written before the run.")
        if not self.leagues:
            raise ScaffoldError("leagues must not be empty.")
        if self.entry_hours_before_kickoff <= 1.0:
            raise ScaffoldError("entry_hours_before_kickoff must exceed 1 (fills lag one bar).")
        if not 0.0 < self.stake_cap <= 1.0:
            raise ScaffoldError("stake_cap must be in (0, 1].")
        for name, values in (
            ("edge_thresholds", self.edge_thresholds),
            ("kelly_fractions", self.kelly_fractions),
        ):
            if not values or len(set(values)) != len(values):
                raise ScaffoldError(f"{name} must be non-empty and without duplicates.")
        if any(not 0.0 <= v < 1.0 for v in self.edge_thresholds):
            raise ScaffoldError("edge_thresholds must be in [0, 1).")
        if any(not 0.0 < v <= 1.0 for v in self.kelly_fractions):
            raise ScaffoldError("kelly_fractions must be in (0, 1].")
        swept = {
            k: v
            for k, v in (
                ("edge_threshold", self.edge_thresholds),
                ("kelly_fraction", self.kelly_fractions),
            )
            if len(v) > 1
        }
        if swept:
            for key, values in swept.items():
                if key not in self.selected or self.selected[key] not in values:
                    raise ScaffoldError(f"selected[{key!r}] must be one of the grid values.")
        extra = set(self.selected) - set(swept)
        if extra:
            raise ScaffoldError(f"selected names parameters that are not swept: {sorted(extra)}.")


_STRATEGY_TEMPLATE = '''"""Stake fractional Kelly on a pitch-edge model forecast once per market, before kickoff.

Subclasses `PredictionMarketStrategy`. For each market the strategy acts on exactly one decision
bar: the one whose hours to kickoff lie in `[entry_hours_before_kickoff - 0.5,
entry_hours_before_kickoff + 0.5)`. On that bar:

1. The forecast is the latest `{feature}` row of the market with `known_at` at or before the
   decision time: the model's probability of Yes.
2. The edge is `forecast - mid`, where `mid` is the book's Yes mid. An edge smaller in magnitude
   than `edge_threshold` stakes nothing.
3. The stake is fractional Kelly of the forecast against the mid
   (`kelly_fraction_binary(forecast, mid, kelly_fraction, stake_cap)`): Yes when positive, No when
   negative, cut to the unspent part of `risk_limits.max_gross`.

The stake is made once and held to resolution; every other bar returns `None`.
"""

from datetime import timedelta

import pandas as pd

from edgelab.strategy import DataRequirements, MarketView, PredictionMarketStrategy, Stake
from edgelab.strategy.helpers import kelly_fraction_binary, remaining_stake_budget

_LEAGUES = {leagues}
_FEATURE = "{feature}"
_ENTRY_TOLERANCE_HOURS = 0.5
_FORECAST_FLOOR = 0.001
_FORECAST_CEILING = 0.999


class {cls}(PredictionMarketStrategy):
    """Fractional Kelly on the pitch-edge forecast against the market mid, once, before kickoff.

    Attributes:
        entry_hours_before_kickoff: Hours before kickoff of the bar the stake is made on.
        edge_threshold: Smallest absolute `forecast - mid` acted on.
        kelly_fraction: Fraction of full Kelly staked.
        stake_cap: Largest fraction of NAV in one market.
    """

    entry_hours_before_kickoff: float
    edge_threshold: float
    kelly_fraction: float
    stake_cap: float

    def data_requirements(self) -> DataRequirements:
        """Hourly books plus the model forecast feature."""
        return DataRequirements(
            instruments={{"prediction_market": _LEAGUES}},
            fields=("ohlc", "order_book_depth"),
            features=(_FEATURE,),
            signal_bars=(),
            signal_series=(),
            frequency=self.schedule.frequency,
            history=timedelta(0),
            warmup=timedelta(0),
        )

    def _on_entry_bar(self, view: MarketView, market: str) -> bool:
        """Whether `view.decision_time` is the market's entry bar."""
        kickoff = view.instruments[market].kickoff
        if kickoff is None:
            return False
        hours = (pd.Timestamp(kickoff) - pd.Timestamp(view.decision_time)).total_seconds() / 3600.0
        entry = self.entry_hours_before_kickoff
        return entry - _ENTRY_TOLERANCE_HOURS <= hours < entry + _ENTRY_TOLERANCE_HOURS

    def _forecast(self, view: MarketView, market: str) -> float | None:
        """The latest known model probability of Yes for `market`, clipped, or `None`."""
        rows = view.features
        rows = rows.loc[(rows["symbol"] == market) & (rows["name"] == _FEATURE)]
        if "known_at" in rows.columns:
            rows = rows.loc[rows["known_at"] <= pd.Timestamp(view.decision_time)]
        if rows.empty:
            return None
        value = float(rows.sort_values("ts")["value"].iloc[-1])
        if not 0.0 < value < 1.0:
            return None
        return min(max(value, _FORECAST_FLOOR), _FORECAST_CEILING)

    def stake(self, view: MarketView, market: str) -> Stake | None:
        """Stake fractional Kelly on the model forecast on the entry bar, else nothing.

        Returns:
            A `Stake` whose forecast is the model probability, or `None` off the entry bar, without
            a book mid or a forecast, for an edge below `edge_threshold`, or when the Kelly
            fraction or the unspent gross budget is zero.
        """
        book = view.bars.loc[view.bars["symbol"] == market, "close"]
        if book.empty or not self._on_entry_bar(view, market):
            return None
        mid = float(book.iloc[-1])
        if not 0.0 < mid < 1.0:
            return None
        forecast = self._forecast(view, market)
        if forecast is None or abs(forecast - mid) < self.edge_threshold:
            return None
        kelly = kelly_fraction_binary(forecast, mid, self.kelly_fraction, self.stake_cap)
        fraction = min(abs(kelly), remaining_stake_budget(view, self.risk_limits.max_gross))
        if fraction <= 0.0:
            return None
        return Stake(side="yes" if kelly > 0.0 else "no", fraction=fraction, forecast=forecast)
'''


def _yaml_list(values: tuple[float, ...]) -> str:
    return "[" + ", ".join(repr(float(v)) for v in values) + "]"


def _yaml_text(spec: ScaffoldSpec) -> str:
    fixed: dict[str, float] = {"entry_hours_before_kickoff": spec.entry_hours_before_kickoff}
    grid_lines: list[str] = []
    for key, values in (
        ("edge_threshold", spec.edge_thresholds),
        ("kelly_fraction", spec.kelly_fractions),
    ):
        if len(values) > 1:
            grid_lines.append(f"    {key}: {_yaml_list(values)}")
        else:
            fixed[key] = values[0]
    fixed["stake_cap"] = spec.stake_cap
    fixed_text = "\n".join(f"  {k}: {v!r}" for k, v in fixed.items())
    grid = "\n".join(grid_lines)
    if grid_lines:
        sel = ", ".join(f"{k}: {v!r}" for k, v in spec.selected.items())
        parameters = f"parameters:\n  grid:\n{grid}\n  selected: {{{sel}}}\n"
    else:
        parameters = "parameters:\n  grid: {}\n"
    rationale = " ".join(spec.rationale.split())
    return f"""# Pre-registered before any counted run: {spec.n_trials} trial(s) in the grid below.
# The grading: block is appended by `edgelab strategy init {spec.folder} --asset-class prediction_market`.
name: {json.dumps(spec.title)}
version: "1.0.0"
author: {json.dumps(spec.author)}
created: {spec.created}
strategy: strategies/{spec.folder}/strategy.py:{spec.class_name}
params:
  universe: {{kind: index, index: polymarket_soccer, point_in_time: false}}
  schedule:
    frequency: "1h"
    decision_time: null
    rebalance: every_bar
    calendar: "24/7"
    bars_per_year: 8760
  risk_limits: {{max_gross: 1.0, max_net: 1.0, max_weight_per_instrument: {spec.stake_cap!r}}}
  rebalance_band: 0.02
  on_missing_data: hold
  order_timing: open
  entry_window: {{earliest_before_event: PT24H, latest_before_event: PT1H30M}}
  trade_in_play: false
{fixed_text}
{parameters}hypothesis:
  rationale: >
    {rationale}
  null_hypothesis: "mean net return per bet <= 0"
  alternative_hypothesis: "mean net return per bet > 0"
  primary_metric: sharpe
  threshold: 0.0
  alpha: 0.05
  sided: one
  benchmark: {{kind: risk_free, annual_rate: 0.0}}
data:
  provider: parquet
  range: {{start: {spec.range_start}, end: {spec.range_end}}}
  frequency: "1h"
  params: {{path: {json.dumps(spec.data_path)}}}
capital: {float(spec.capital)!r}
seed: {int(spec.seed)}
"""


def scaffold_strategy(spec: ScaffoldSpec, strategies_root: str | Path) -> Path:
    """Write `<strategies_root>/pitch-edge-<slug>/` and return the folder.

    Args:
        spec: The pre-registered specification.
        strategies_root: edgelab's `strategies/` directory.

    Returns:
        The strategy folder.

    Raises:
        ScaffoldError: `spec` is invalid, or the folder already holds a `strategy.yaml` (a
            registered strategy is never overwritten).
    """
    spec.validate()
    folder = Path(strategies_root) / spec.folder
    if (folder / "strategy.yaml").exists():
        raise ScaffoldError(f"{folder} already has a strategy.yaml; refusing to overwrite.")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "strategy.py").write_text(
        _STRATEGY_TEMPLATE.format(feature=MODEL_FEATURE, leagues=repr(tuple(spec.leagues)), cls=spec.class_name)
    )
    (folder / "strategy.yaml").write_text(_yaml_text(spec))
    (folder / "strategy.json").write_text(
        json.dumps(
            {
                "slug": spec.folder,
                "title": spec.title,
                "authors": [spec.author],
                "year": int(spec.created[:4]),
                "source_url": None,
                "ssrn_id": None,
                "status": "inbox",
                "asset_class": "prediction_market",
                "frequency": "1h",
                "tags": ["prediction-market", "pitch-edge"],
                "data": {
                    "source": "Exported by pitch_edge.edgelab.export from the pitch-edge warehouse.",
                    "rebuild": "Re-run the pitch-edge export for the same cut-off.",
                },
                "notes": "",
            },
            indent=2,
        )
        + "\n"
    )
    (folder / "notes.md").write_text(
        f"# {spec.title}\n\n## Hypothesis\n\n{spec.rationale.strip()}\n\n## Rules\n\n"
        f"One stake per market {spec.entry_hours_before_kickoff:g} h before kickoff on the model "
        f"forecast `{MODEL_FEATURE}`; fractional Kelly, cap {spec.stake_cap:g}, held to resolution.\n"
        f"Grid: {spec.n_trials} trial(s).\n"
    )
    return folder
