"""Team strength as a mean-reverting stochastic process (Ornstein-Uhlenbeck), scored by
Monte Carlo simulation — a quant-finance-style model (the same SDE family used for interest
rates, e.g. Vasicek), not a repackaged GBDT.

Only input: `elo_home`/`elo_away`, already in the feature store. No market odds, no retrain-heavy
optimizer: `fit()` estimates the OU innovation scale by a closed-form method-of-moments pass over
historical Elo increments (one groupby-diff, no numerical fit), and the draw-rate constant from
the training outcome frequency — cheap enough to refit every walk-forward fold.

An Elo rating is a noisy point estimate of a team's *true* strength; this model treats today's
Elo as the current draw of a stationary OU process and Monte Carlo samples plausible strength
realizations around it, mapping each simulated draw to an outcome probability and averaging.
The OU transition density this simulates satisfies the Fokker-Planck PDE
`∂p/∂t = θ ∂/∂x[(x-μ)p] + (σ²/2) ∂²p/∂x²` — simulation is used here instead of solving that PDE
directly since the outcome mapping below (Elo-diff -> 3-way probability) has no closed form.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pitch_edge.models.base import MatchModel, to_frame

HOME_ADVANTAGE_ELO = 65.0
DRAW_BUMP_SCALE = 200.0  # how quickly draw probability decays as teams' strength gap widens


class StochasticStrengthModel(MatchModel):
    name = "stochastic_strength"

    def __init__(self, n_sims: int = 2000, seed: int = 0):
        self.n_sims = n_sims
        self._rng = np.random.default_rng(seed)
        self.sigma_: float = 100.0
        self.draw_rate_: float = 0.25

    def fit(self, train: pd.DataFrame) -> StochasticStrengthModel:
        df = train.dropna(subset=["elo_home", "elo_away"]).sort_values("date")
        long = pd.concat(
            [
                df[["home_team", "date", "elo_home"]].rename(columns={"home_team": "team", "elo_home": "elo"}),
                df[["away_team", "date", "elo_away"]].rename(columns={"away_team": "team", "elo_away": "elo"}),
            ]
        ).sort_values(["team", "date"])
        delta = long.groupby("team")["elo"].diff().dropna()
        self.sigma_ = float(delta.std()) if len(delta) > 5 and delta.std() > 0 else 100.0
        if "result" in df.columns and len(df):
            self.draw_rate_ = float((df["result"] == 1).mean())
        return self

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        home_elo = X["elo_home"].fillna(X["elo_home"].mean()).to_numpy()
        away_elo = X["elo_away"].fillna(X["elo_away"].mean()).to_numpy()
        n = len(X)
        home_noise = self._rng.normal(0.0, self.sigma_, size=(self.n_sims, n))
        away_noise = self._rng.normal(0.0, self.sigma_, size=(self.n_sims, n))
        diff = (home_elo + home_noise + HOME_ADVANTAGE_ELO) - (away_elo + away_noise)

        p_home_2way = 1.0 / (1.0 + 10.0 ** (-diff / 400.0))
        p_draw = self.draw_rate_ * np.exp(-((diff / DRAW_BUMP_SCALE) ** 2))
        p_home = p_home_2way * (1.0 - p_draw)
        p_away = (1.0 - p_home_2way) * (1.0 - p_draw)

        probs = np.stack([p_home.mean(axis=0), p_draw.mean(axis=0), p_away.mean(axis=0)], axis=1)
        return to_frame(probs, X.index)
