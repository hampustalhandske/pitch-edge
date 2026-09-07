"""MatchModel adapter around the Dixon-Coles goal model (the mandatory baseline).

One Dixon-Coles fit per league: attack/defence ratings are only identified
within a competition (clubs from different leagues rarely meet), and a
per-league fit keeps each optimisation small (~40-60 parameters) instead of
one 600-parameter joint problem over eleven leagues.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from pitch_edge.models.base import MatchModel, to_frame
from pitch_edge.models.dixon_coles import DixonColesModel

logger = logging.getLogger(__name__)

_GLOBAL = "__all__"


class DixonColesMatchModel(MatchModel):
    name = "dixon_coles"

    def __init__(
        self,
        xi: float = 0.0018,
        max_goals: int = 10,
        lookback_days: int | None = 730,
        per_league: bool = True,
        min_league_rows: int = 120,
    ):
        self.xi = xi
        self.max_goals = max_goals
        self.lookback_days = lookback_days
        self.per_league = per_league
        self.min_league_rows = min_league_rows
        self._models: dict[str, DixonColesModel] = {}
        self._train_rows = 0

    def _window(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.lookback_days:
            cutoff = pd.to_datetime(df["date"]).max() - pd.Timedelta(days=self.lookback_days)
            recent = df[pd.to_datetime(df["date"]) >= cutoff]
            if len(recent) >= self.min_league_rows:
                return recent
        return df

    def fit(self, train: pd.DataFrame) -> DixonColesMatchModel:
        cols = ["home_team", "away_team", "home_goals", "away_goals", "date"]
        self._models = {}
        groups: list[tuple[str, pd.DataFrame]]
        if self.per_league and "league_code" in train.columns:
            groups = [(str(k), g) for k, g in train.groupby("league_code")]
        else:
            groups = [(_GLOBAL, train)]
        rows = 0
        for key, g in groups:
            g = self._window(g)
            if len(g) < 20:
                continue
            try:
                self._models[key] = DixonColesModel(max_goals=self.max_goals, xi=self.xi).fit(g[cols])
                rows += len(g)
            except Exception as exc:  # noqa: BLE001 - a degenerate league must not kill the fold
                logger.warning("Dixon-Coles fit failed for %s: %s", key, exc)
        if not self._models:
            # fall back to one global fit
            g = self._window(train)
            self._models[_GLOBAL] = DixonColesModel(max_goals=self.max_goals, xi=self.xi).fit(g[cols])
            rows = len(g)
        self._train_rows = rows
        return self

    def _model_for(self, league_code: str | None) -> DixonColesModel:
        if league_code is not None and str(league_code) in self._models:
            return self._models[str(league_code)]
        if _GLOBAL in self._models:
            return self._models[_GLOBAL]
        return next(iter(self._models.values()))

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._models:
            raise RuntimeError("fit() first")
        codes = X["league_code"] if "league_code" in X.columns else pd.Series([None] * len(X), index=X.index)
        rows = [
            list(self._model_for(c).predict_outcome_probabilities(h, a).values())
            for h, a, c in zip(X["home_team"], X["away_team"], codes, strict=True)
        ]
        return to_frame(np.array(rows), X.index)

    def expected_goals(self, home_team: str, away_team: str, league_code: str | None = None) -> tuple[float, float]:
        if not self._models:
            raise RuntimeError("fit() first")
        return self._model_for(league_code)._expected_goals(home_team, away_team)

    def card(self) -> dict:
        return {
            **super().card(),
            "family": "bivariate Poisson (Dixon & Coles 1997) with tau low-score correction, one fit per league",
            "time_decay_xi": self.xi,
            "lookback_days": self.lookback_days,
            "leagues_fitted": sorted(self._models),
            "train_rows_last_fit": self._train_rows,
            "features": ["home_team", "away_team", "home_goals", "away_goals", "date", "league_code"],
            "leakage_checks": ["fit uses matches strictly before the prediction date (enforced by backtester)"],
        }
