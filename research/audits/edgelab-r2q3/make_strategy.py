"""Write the edgelab strategy folders for the R2-Q3 stale-soft-open strategies.

Usage: python make_strategy.py <max|bet365|max-nopersist> <strategies-root>

The folder comes from `pitch_edge.edgelab.scaffold_strategy`; `strategy.py`, the rule parameters in
`strategy.yaml` and `notes.md` are then replaced, because the scaffold's rule (fractional Kelly on the
forecast against the mid) is not the R2-Q3 rule (back the single largest-edge outcome of a match at the
quoted price). `strategy init` appends the grading policy afterwards.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from pitch_edge.edgelab import ScaffoldSpec, scaffold_strategy

DATA = Path(__file__).resolve().parents[3] / "data" / "research" / "r2" / "edgelab"
RANGE = ("2012-07-27", "2025-06-01")
DIVISIONS = ("B1 D1 D2 E0 E1 E2 E3 EC F1 F2 G1 I1 I2 N1 P1 SC0 SC1 SC2 SC3 SP1 SP2 T1").split()
STAKE_FRACTION = 0.005
HALF_SPREAD = 0.01
EDGE_THRESHOLD = 0.02

STRATEGY = '''"""R2-Q3 stale soft lines: back the candidate outcome of a match at the quoted open price.

Subclasses `PredictionMarketStrategy` on a bookmaker-odds universe: one binary Yes market per match, for the
outcome with the largest positive edge `log(p_pin_open * odds)` at the football-data odds snapshot (the dataset
holds no other outcome of the match). The Yes ask is 1 / the quoted decimal odds ({price_source}); the bid and
mid follow from a fixed relative spread (`quote_half_spread`). The market's `pe_model_p` feature is the
power-de-vigged Pinnacle open probability of the outcome, stamped at the snapshot time.

On the bar at which a forecast row is stamped (`ts == decision_time`), the strategy computes
`edge = log(model_p / ask)` with `ask = mid / (1 - quote_half_spread)` and, if the edge exceeds `edge_threshold`,
stakes `stake_fraction` of NAV on Yes, cut to the unspent part of `risk_limits.max_gross`. Every other bar
returns `None`, which leaves a position as it is. The stake is held to resolution.
"""

import math
from datetime import timedelta

import pandas as pd

from edgelab.strategy import DataRequirements, MarketView, PredictionMarketStrategy, Stake
from edgelab.strategy.helpers import remaining_stake_budget

_LEAGUES = {leagues}
_FEATURE = "pe_model_p"
_FORECAST_FLOOR = 0.001
_FORECAST_CEILING = 0.999


class {cls}(PredictionMarketStrategy):
    """Back the candidate outcome of each match once, at the odds snapshot, held to resolution.

    Attributes:
        edge_threshold: Smallest `log(model_p * odds)` that is staked.
        stake_fraction: Fraction of NAV staked.
        quote_half_spread: Half of the relative spread between the Yes bid and ask of the dataset;
            the ask is the mid divided by `1 - quote_half_spread`.
    """

    edge_threshold: float
    stake_fraction: float
    quote_half_spread: float

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

    def _forecast_now(self, view: MarketView, market: str) -> float | None:
        """The model probability stamped at the decision time for `market`, clipped, or `None`."""
        rows = view.features
        rows = rows.loc[rows["ts"] == pd.Timestamp(view.decision_time)]
        rows = rows.loc[rows["symbol"] == market, "value"]
        if rows.empty:
            return None
        value = float(rows.iloc[-1])
        if value != value:
            return None
        return min(max(value, _FORECAST_FLOOR), _FORECAST_CEILING)

    def stake(self, view: MarketView, market: str) -> Stake | None:
        """Stake `stake_fraction` of NAV on Yes if the edge at the forecast bar clears the threshold.

        Returns:
            A `Stake` on Yes with the model probability as forecast, or `None` off the forecast bar,
            without a book mid, for an edge not above `edge_threshold`, or when the unspent gross
            budget is zero.
        """
        book = view.bars.loc[view.bars["symbol"] == market, "close"]
        if book.empty:
            return None
        forecast = self._forecast_now(view, market)
        if forecast is None:
            return None
        mid = float(book.iloc[-1])
        if not 0.0 < mid < 1.0:
            return None
        ask = mid / (1.0 - self.quote_half_spread)
        if ask >= 1.0 or math.log(forecast / ask) <= self.edge_threshold:
            return None
        fraction = min(self.stake_fraction, remaining_stake_budget(view, self.risk_limits.max_gross))
        if fraction <= 0.0:
            return None
        return Stake(side="yes", fraction=fraction, forecast=forecast)
'''

NOTES = """# {title}

Pitch-edge study R2-Q3 (`bookmaker-line-structure-stale-soft-open-v1`) as an edgelab prediction-market
strategy. {what}

## Rule (one trial, theta = 0.02, no sweep)

At the football-data pre-closing ("open") snapshot of each match, for the three outcomes:
`edge = log(p_pin_open * odds)` with `p_pin_open` the power-de-vigged Pinnacle open probability and `odds` the
{price_source}. Back the single outcome with the largest edge if `edge > 0.02`. Quotes above 1.5 x the Pinnacle
open are data errors: that outcome is never the candidate. Flat stake of {stake} of NAV per bet, held to resolution.

## Modelling the bookmaker market as binary markets (every choice)

- One binary Yes market per match, `<division>-<yyyymmdd>-<home>-<away>:<win_home|draw|win_away>`, for the
  match's candidate outcome only: the outcome with the largest edge `log(p_pin_open * odds)` among the three,
  after the 1.5 x data-error cap, if that edge is positive. This selection uses the open quotes alone (known at
  the snapshot time) and is made in the export, not inside edgelab. Reason: edgelab's bar loop costs time per live
  market per hour, and a first design with all three outcomes per match (288,898 markets for the max source,
  sibling comparison inside the strategy) had not finished one dry-run pass after more than 200 CPU-minutes.
  The strategy still applies the registered threshold (theta = 0.02) to the quoted price and decides the fill;
  candidates with an edge in (0, 0.02] are markets the strategy declines. Resolution from `ft_result`,
  `resolved_at` = kickoff + 3 h.
- Universe: the 22 main football-data divisions (`mmz4281/`), 2012-07-27 (first quote) to 2025-06-01 (the
  data-fit span rule needs the first and last quote within 5 days of the range ends), matches with all of Pinnacle open, Pinnacle close and the price source's three open odds
  > 1 and a result: the study's own eligibility function. The study's evaluation window is 2019-07 to
  2025-06; edgelab is given the exploratory era 2012-2019 as well to reach the usable years the data-fit rule
  needs, and 20% of the window (the last ~2.6 years) is the locked holdout, so edgelab's headline is scored
  mostly on 2012-2022.
- Snapshot time `open_known_at` = latest Tuesday or Friday 18:00 Europe/London strictly before kickoff (the
  r1-common helper). It is the latest moment the open odds can have been collected, used as both the book
  timestamp and the `known_at` of `pe_model_p`.
- Kickoff = `kickoff_time` (UK local) where recorded (46% of the matches: 2019/20 onward). Elsewhere (all of
  2012-2019) the default is 15:00 UK local on Saturday and Sunday and 19:45 on Monday to Friday. This differs
  from the brief's flat 15:00: a flat 15:00 would place Tuesday, Friday and Monday evening matches' snapshot four
  days earlier (the previous Friday or Tuesday) than football-data collects. The default only moves the timing
  of the snapshot, not any price.
- Matches with less than 1 h 30 min between the snapshot and kickoff cannot be filled one bar later before
  kickoff and are left out of the dataset ({dropped:,} of {eligible:,} eligible matches); the entry window enforces the same limit.
- Ask (Yes) = 1 / odds of the price source. Bid = ask x (1 - 0.02): a fixed 2% relative spread, mid = ask x 0.99.
  The bid never matters for a back bet; it only sets the mid the position is marked at and the stake size
  (`shares = fraction x NAV / mid`, paid at the ask, so the cash staked is 1.01 x the fraction). Depth is 1e7 shares
  per side: stake limits are not modelled.
- Persistence assumption: the open quote stayed available for at least one hour. The book has snapshots at the
  snapshot hour and the next hour with the same prices (edgelab fills one bar after the decision, and the
  policy accepts only a quote less than one bar old). A dry-run variant (`max-nopersist`) replaces the
  later quote with Pinnacle's own open price (the soft quote is gone) to show the sensitivity; it is not counted.
- No closing snapshot. edgelab's data-fit density rule needs a market's quotes to cover at least 90% of the hourly
  buckets between its first and last quote, so a snapshot just before kickoff would force an hourly series for every
  market (about 9 million rows). Each market has two snapshots, the open hour and the next hour. A position is
  marked at the open mid until it settles; edgelab has no CLV and the close enters nothing it computes. The
  pitch-edge side of the reconciliation holds the close.
- `pe_model_p` = `p_pin_open` (power de-vig of Pinnacle open) with `ts = known_at` = the snapshot time. The strategy
  declares only this feature and acts on the bar where it is stamped (`ts == decision_time`), so the decision hour
  is the snapshot hour of the data.
- Fees: no Polymarket fee. The dataset carries no `market_type` feature, so edgelab does not charge the
  Polymarket sports taker-fee curve; the margin is in the price. The grading policy's flat commission (1 bp of
  notional) still applies; the policy is not edited.
- The sizing is flat (0.5% of NAV, max weight per market 1%), not Kelly: the study scores unit stakes.

## What edgelab can and cannot show here

Quoted-price fact only. The auditor signed the max-odds CLV as a description of quoted prices; executability
(stake limits, stale or erroneous quotes, whether the price existed at the Pinnacle open) cannot be tested from
this data, and the persistence assumption above is the whole fill model. edgelab scores realised P&L (stake x
outcome), not CLV, so a CLV-positive strategy whose ROI is noisy can still grade `inconclusive`.
"""

ENTRY_WINDOW_OLD = "{earliest_before_event: PT24H, latest_before_event: PT1H30M}"
ENTRY_WINDOW_NEW = "{earliest_before_event: PT100H, latest_before_event: PT1H30M}"


def main(variant: str, root: Path) -> None:
    source = "bet365_open" if variant == "bet365" else "max_open"
    price_source = "bet365 open odds" if variant == "bet365" else "football-data market-maximum open odds"
    slug = f"r2q3-{variant}"
    title = f"R2-Q3 stale soft open ({source})" + (" negative control" if variant == "bet365" else "")
    if variant == "max-nopersist":
        title += " quote does not persist"
    what = (
        "Negative control: the same rule with bet365's own open odds as the price (pitch-edge: CLV +0.46% "
        "[-0.06, 0.95], rejected)."
        if variant == "bet365"
        else "Price = the market-maximum open quote, an upper bound across many books (pitch-edge: CLV +2.08% "
        "[1.91, 2.26], n = 11,172, 2019-07 to 2025-06; the auditor signed it only as a quoted-price fact)."
    )
    rationale = (
        f"Back the outcome whose {price_source} most exceeds the power-de-vigged Pinnacle open fair price "
        "when the edge is above 2%. "
        + (
            "Hypothesis: the best open quote across bookmakers is stale relative to the sharp price, so it "
            "beats the close and earns a positive return. "
            if variant != "bet365"
            else "Negative control: a single named book's open quote is not a stale soft line (the study rejected "
            "it), so edgelab should not find an edge. "
        )
        + "Null: mean net return per bet <= 0."
    )
    spec = ScaffoldSpec(
        slug=slug,
        title=title,
        rationale=rationale,
        data_path=str(DATA / slug),
        range_start=RANGE[0],
        range_end=RANGE[1],
        edge_thresholds=(EDGE_THRESHOLD,),
        kelly_fractions=(0.25,),
        stake_cap=0.01,
        entry_hours_before_kickoff=3.0,
        leagues=tuple(DIVISIONS),
        capital=100000.0,
        seed=20261020,
        author="pitch-edge",
        created="2026-10-20",
    )
    folder = scaffold_strategy(spec, root)
    (folder / "strategy.py").write_text(
        STRATEGY.format(cls=spec.class_name, leagues=repr(tuple(DIVISIONS)), price_source=price_source)
    )
    yaml_path = folder / "strategy.yaml"
    text = yaml_path.read_text()
    assert ENTRY_WINDOW_OLD in text
    text = text.replace(ENTRY_WINDOW_OLD, ENTRY_WINDOW_NEW)
    old = "  entry_hours_before_kickoff: 3.0\n  edge_threshold: 0.02\n  kelly_fraction: 0.25\n  stake_cap: 0.01\n"
    assert old in text
    text = text.replace(
        old,
        f"  edge_threshold: {EDGE_THRESHOLD}\n  stake_fraction: {STAKE_FRACTION}\n  quote_half_spread: {HALF_SPREAD}\n",
    )
    yaml_path.write_text(text)
    summary = json.loads((DATA / f"{slug}-summary.json").read_text())
    (folder / "notes.md").write_text(
        NOTES.format(
            title=title,
            what=what,
            price_source=price_source,
            stake=STAKE_FRACTION,
            dropped=summary["dropped_lead_lt_1h30"],
            eligible=summary["eligible_matches"],
        )
    )
    print(folder)


if __name__ == "__main__":
    main(sys.argv[1], Path(sys.argv[2]))
