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
        # One fixed set of standard-normal draws per model (common random numbers): every match is
        # simulated with the same draws scaled by the fitted sigma, so a match's probabilities are a
        # pure function of the fitted parameters and its own row — independent of batch size, batch
        # composition, row order and how many predictions came before.
        rng = np.random.default_rng(seed)
        self._z_home = rng.standard_normal(n_sims)
        self._z_away = rng.standard_normal(n_sims)
        self.sigma_: float = 100.0
        self.draw_rate_: float = 0.25
        self.elo_fill_: float = 1500.0

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
        if long["elo"].notna().any():
            self.elo_fill_ = float(long["elo"].mean())
        if "result" in df.columns and len(df):
            self.draw_rate_ = float((df["result"] == 1).mean())
        return self

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Monte Carlo outcome probabilities per row of X. A missing Elo is filled with the mean
        training Elo (`elo_fill_`), never a statistic of X itself."""
        home_elo = X["elo_home"].astype(float).fillna(self.elo_fill_).to_numpy()
        away_elo = X["elo_away"].astype(float).fillna(self.elo_fill_).to_numpy()
        noise = self.sigma_ * (self._z_home - self._z_away)
        diff = (home_elo + HOME_ADVANTAGE_ELO - away_elo)[None, :] + noise[:, None]

        p_home_2way = 1.0 / (1.0 + 10.0 ** (-diff / 400.0))
        p_draw = self.draw_rate_ * np.exp(-((diff / DRAW_BUMP_SCALE) ** 2))
        p_home = p_home_2way * (1.0 - p_draw)
        p_away = (1.0 - p_home_2way) * (1.0 - p_draw)

        probs = np.stack([p_home.mean(axis=0), p_draw.mean(axis=0), p_away.mean(axis=0)], axis=1)
        return to_frame(probs, X.index)
