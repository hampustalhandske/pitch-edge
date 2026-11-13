"""R2-Q3 stale soft lines: back the candidate outcome of a match at the quoted open price.

Subclasses `PredictionMarketStrategy` on a bookmaker-odds universe: one binary Yes market per match, for the
outcome with the largest positive edge `log(p_pin_open * odds)` at the football-data odds snapshot (the dataset
holds no other outcome of the match). The Yes ask is 1 / the quoted decimal odds (football-data market-maximum open odds); the bid and
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

_LEAGUES = ('B1', 'D1', 'D2', 'E0', 'E1', 'E2', 'E3', 'EC', 'F1', 'F2', 'G1', 'I1', 'I2', 'N1', 'P1', 'SC0', 'SC1', 'SC2', 'SC3', 'SP1', 'SP2', 'T1')
_FEATURE = "pe_model_p"
_FORECAST_FLOOR = 0.001
_FORECAST_CEILING = 0.999


class PitchEdgeR2q3Max(PredictionMarketStrategy):
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
            instruments={"prediction_market": _LEAGUES},
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
