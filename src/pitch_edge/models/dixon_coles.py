"""Dixon-Coles bivariate Poisson goal model.

The canonical baseline in football analytics (Dixon & Coles, 1997). Every
other model in this project is compared against this one — if a fancier
model can't beat Dixon-Coles out of sample, it isn't earning its complexity.

Model:
    home_goals ~ Poisson(lambda_home), away_goals ~ Poisson(lambda_away)
    lambda_home = exp(home_advantage + attack[home] - defence[away])
    lambda_away = exp(attack[away] - defence[home])

with a low-score correlation correction (the "tau" adjustment) for the
(0,0), (1,0), (0,1), (1,1) scorelines, and an optional exponential time
decay so recent matches count more than old ones.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson


def _tau(home_goals: int, away_goals: int, lambda_home: float, lambda_away: float, rho: float) -> float:
    """Dixon-Coles low-score dependence correction."""
    if home_goals == 0 and away_goals == 0:
        return 1 - lambda_home * lambda_away * rho
    if home_goals == 0 and away_goals == 1:
        return 1 + lambda_home * rho
    if home_goals == 1 and away_goals == 0:
        return 1 + lambda_away * rho
    if home_goals == 1 and away_goals == 1:
        return 1 - rho
    return 1.0


@dataclass
class DixonColesModel:
    max_goals: int = 10
    xi: float = 0.0018  # time-decay rate (per day); 0 disables decay
    teams_: list[str] = field(default_factory=list)
    attack_: dict[str, float] = field(default_factory=dict)
    defence_: dict[str, float] = field(default_factory=dict)
    home_advantage_: float = 0.0
    rho_: float = 0.0
    fitted_: bool = False

    def fit(self, matches: pd.DataFrame) -> DixonColesModel:
        """matches needs columns: home_team, away_team, home_goals, away_goals, date."""
        required = {"home_team", "away_team", "home_goals", "away_goals", "date"}
        missing = required - set(matches.columns)
        if missing:
            raise ValueError(f"DixonColesModel.fit missing columns: {sorted(missing)}")
        if matches.empty:
            raise ValueError("DixonColesModel.fit called with an empty DataFrame")

        df = matches.copy()
        df["date"] = pd.to_datetime(df["date"])
        max_date = df["date"].max()
        df["days_ago"] = (max_date - df["date"]).dt.days
        df["weight"] = np.exp(-self.xi * df["days_ago"])

        teams = sorted(set(df["home_team"]) | set(df["away_team"]))
        self.teams_ = teams
        n = len(teams)
        idx = {t: i for i, t in enumerate(teams)}

        # params: [attack_0..n-1, defence_0..n-1, home_adv, rho]
        # constraint: mean(attack) = 0 to avoid identifiability issues, enforced post-hoc
        x0 = np.zeros(2 * n + 2)
        x0[:n] = 0.0
        x0[n : 2 * n] = 0.0
        x0[-2] = 0.1  # home advantage
        x0[-1] = -0.01  # rho

        home_idx = df["home_team"].map(idx).to_numpy()
        away_idx = df["away_team"].map(idx).to_numpy()
        hg = df["home_goals"].to_numpy(dtype=int)
        ag = df["away_goals"].to_numpy(dtype=int)
        w = df["weight"].to_numpy()

        def neg_log_likelihood(params: np.ndarray) -> float:
            attack = params[:n]
            defence = params[n : 2 * n]
            home_adv = params[-2]
            rho = params[-1]

            lam_h = np.exp(home_adv + attack[home_idx] - defence[away_idx])
            lam_a = np.exp(attack[away_idx] - defence[home_idx])

            log_lik = poisson.logpmf(hg, lam_h) + poisson.logpmf(ag, lam_a)
            # vectorised Dixon-Coles tau for the four low-score cells
            tau_vals = np.ones_like(lam_h)
            m00 = (hg == 0) & (ag == 0)
            m01 = (hg == 0) & (ag == 1)
            m10 = (hg == 1) & (ag == 0)
            m11 = (hg == 1) & (ag == 1)
            tau_vals[m00] = 1 - lam_h[m00] * lam_a[m00] * rho
            tau_vals[m01] = 1 + lam_h[m01] * rho
            tau_vals[m10] = 1 + lam_a[m10] * rho
            tau_vals[m11] = 1 - rho
            log_lik = log_lik + np.log(np.clip(tau_vals, 1e-10, None))
            return -np.sum(w * log_lik)

        bounds = [(-3.0, 3.0)] * (2 * n) + [(-1.0, 1.0), (-0.3, 0.3)]
        result = minimize(neg_log_likelihood, x0, method="L-BFGS-B", bounds=bounds)
        params = result.x if np.all(np.isfinite(result.x)) else x0
        attack = params[:n]
        defence = params[n : 2 * n]
        attack -= attack.mean()  # identifiability normalization

        self.attack_ = dict(zip(teams, attack, strict=False))
        self.defence_ = dict(zip(teams, defence, strict=False))
        self.home_advantage_ = float(params[-2])
        self.rho_ = float(params[-1])
        self.fitted_ = True
        return self

    def _expected_goals(self, home_team: str, away_team: str) -> tuple[float, float]:
        if not self.fitted_:
            raise RuntimeError("Model must be fit() before prediction")
        a_h = self.attack_.get(home_team, 0.0)
        d_h = self.defence_.get(home_team, 0.0)
        a_a = self.attack_.get(away_team, 0.0)
        d_a = self.defence_.get(away_team, 0.0)
        lam_h = np.exp(self.home_advantage_ + a_h - d_a)
        lam_a = np.exp(a_a - d_h)
        return float(lam_h), float(lam_a)

    def predict_score_matrix(self, home_team: str, away_team: str) -> np.ndarray:
        """Return a (max_goals+1) x (max_goals+1) probability matrix P[i,j] = P(home=i, away=j)."""
        lam_h, lam_a = self._expected_goals(home_team, away_team)
        goals = np.arange(self.max_goals + 1)
        ph = poisson.pmf(goals, lam_h)
        pa = poisson.pmf(goals, lam_a)
        matrix = np.outer(ph, pa)

        for i in range(2):
            for j in range(2):
                matrix[i, j] *= _tau(i, j, lam_h, lam_a, self.rho_)
        total = matrix.sum()
        if not np.isfinite(total) or total <= 0:
            # degenerate parameters (diverged fit / extreme lambdas): fall back to plain independent Poisson
            matrix = np.outer(ph, pa)
            total = matrix.sum()
            if not np.isfinite(total) or total <= 0:
                matrix = np.full_like(matrix, 1.0 / matrix.size)
                total = 1.0
        return matrix / total

    def predict_outcome_probabilities(self, home_team: str, away_team: str) -> dict[str, float]:
        matrix = self.predict_score_matrix(home_team, away_team)
        home_p = float(np.tril(matrix, -1).sum())
        draw_p = float(np.trace(matrix))
        away_p = float(np.triu(matrix, 1).sum())
        return {"home": home_p, "draw": draw_p, "away": away_p}
