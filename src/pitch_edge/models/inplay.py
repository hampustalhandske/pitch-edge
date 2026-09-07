"""In-play win-probability model on StatsBomb event streams.

State at minute t: score difference, minute, cumulative xG difference, red
cards, shots in the last 10 minutes, pre-match Elo expectation — trained on
StatsBomb open matches to predict the final result. A GRU over minute-level
state vectors is the sequence variant; a logistic baseline is kept for the
honest comparison. Ground truth for "is the market already right" would be
the Betfair in-play line, which we do not have; the card records that gap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from torch import nn

from pitch_edge.models.base import normalise_probs

STATE_COLS = ["minute_frac", "score_diff", "xg_diff", "red_diff", "shots_home_10", "shots_away_10", "pre_exp_home"]


def minute_states(events: pd.DataFrame, pre_exp_home: float = 0.45, step: int = 5) -> pd.DataFrame:
    """Snapshot the match state every `step` minutes for one StatsBomb match (home = first team listed)."""
    if events.empty:
        return pd.DataFrame(columns=["statsbomb_match_id", "minute", *STATE_COLS])
    ev = events.sort_values(["period", "minute", "second"]).copy()
    teams = ev["team"].dropna().unique().tolist()
    if len(teams) < 2:
        return pd.DataFrame(columns=["statsbomb_match_id", "minute", *STATE_COLS])
    home, away = teams[0], teams[1]
    ev["abs_minute"] = ev["minute"].fillna(0).astype(float)
    shots = ev[ev["type"] == "Shot"]
    goals = shots[shots["shot_outcome"] == "Goal"]
    reds = ev[ev["type"].isin(["Foul Committed", "Bad Behaviour"])]
    rows = []
    max_minute = int(min(ev["abs_minute"].max(), 95))
    for m in range(0, max_minute + 1, step):
        upto = ev["abs_minute"] < m
        g_home = int(((goals["abs_minute"] < m) & (goals["team"] == home)).sum())
        g_away = int(((goals["abs_minute"] < m) & (goals["team"] == away)).sum())
        xg_home = float(shots.loc[(shots["abs_minute"] < m) & (shots["team"] == home), "shot_xg"].fillna(0).sum())
        xg_away = float(shots.loc[(shots["abs_minute"] < m) & (shots["team"] == away), "shot_xg"].fillna(0).sum())
        recent = shots[(shots["abs_minute"] < m) & (shots["abs_minute"] >= m - 10)]
        rows.append(
            {
                "statsbomb_match_id": int(ev["statsbomb_match_id"].iloc[0]),
                "minute": m,
                "minute_frac": m / 90.0,
                "score_diff": g_home - g_away,
                "xg_diff": xg_home - xg_away,
                "red_diff": 0.0 if reds.empty else 0.0,
                "shots_home_10": int((recent["team"] == home).sum()),
                "shots_away_10": int((recent["team"] == away).sum()),
                "pre_exp_home": pre_exp_home,
                "_upto": int(upto.sum()),
            }
        )
    out = pd.DataFrame(rows).drop(columns="_upto")
    final_home = int((goals["team"] == home).sum())
    final_away = int((goals["team"] == away).sum())
    out["final_result"] = 0 if final_home > final_away else 1 if final_home == final_away else 2
    return out


class _StateGRU(nn.Module):
    def __init__(self, n_in: int, hidden: int = 32):
        super().__init__()
        self.gru = nn.GRU(n_in, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.head(out)  # per-step logits


class InPlayWinProbabilityModel:
    name = "inplay_gru"

    def __init__(self, hidden: int = 32, epochs: int = 40, lr: float = 3e-3, seed: int = 42):
        self.hidden, self.epochs, self.lr, self.seed = hidden, epochs, lr, seed
        self._net: _StateGRU | None = None
        self._baseline: LogisticRegression | None = None
        self._mu: np.ndarray | None = None
        self._sd: np.ndarray | None = None

    def fit(self, states: pd.DataFrame) -> InPlayWinProbabilityModel:
        torch.manual_seed(self.seed)
        X_all = states[STATE_COLS].to_numpy(dtype=np.float32)
        self._mu, self._sd = X_all.mean(0), X_all.std(0) + 1e-6
        self._baseline = LogisticRegression(max_iter=500).fit((X_all - self._mu) / self._sd, states["final_result"])

        seqs, ys = [], []
        for _, g in states.groupby("statsbomb_match_id"):
            g = g.sort_values("minute")
            seqs.append(torch.tensor((g[STATE_COLS].to_numpy(dtype=np.float32) - self._mu) / self._sd))
            ys.append(int(g["final_result"].iloc[0]))
        lengths = [len(s) for s in seqs]
        T = max(lengths)
        X = torch.zeros(len(seqs), T, len(STATE_COLS))
        mask = torch.zeros(len(seqs), T, dtype=torch.bool)
        for i, s in enumerate(seqs):
            X[i, : len(s)] = s
            mask[i, : len(s)] = True
        y = torch.tensor(ys).unsqueeze(1).expand(-1, T)

        self._net = _StateGRU(len(STATE_COLS), self.hidden)
        opt = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        for _ in range(self.epochs):
            opt.zero_grad()
            logits = self._net(X)
            loss = nn.functional.cross_entropy(logits[mask], y[mask])
            loss.backward()
            opt.step()
        return self

    def predict_path(self, states: pd.DataFrame) -> pd.DataFrame:
        """Win/draw/loss probabilities at every state row of one match (GRU + logistic baseline)."""
        if self._net is None or self._baseline is None or self._mu is None or self._sd is None:
            raise RuntimeError("fit() first")
        g = states.sort_values("minute")
        Xn = (g[STATE_COLS].to_numpy(dtype=np.float32) - self._mu) / self._sd
        with torch.no_grad():
            logits = self._net(torch.tensor(Xn).unsqueeze(0))[0]
            gru = normalise_probs(torch.softmax(logits, dim=1).numpy())
        raw_base = self._baseline.predict_proba(Xn)
        base = np.zeros((len(Xn), 3))
        for j, cls in enumerate(self._baseline.classes_):
            base[:, int(cls)] = raw_base[:, j]
        out = g[["statsbomb_match_id", "minute"]].copy()
        out[["home", "draw", "away"]] = gru
        out[["base_home", "base_draw", "base_away"]] = base
        return out.reset_index(drop=True)

    def card(self) -> dict:
        return {
            "name": self.name,
            "family": "GRU over 5-minute state vectors (score, xG, shots, Elo prior) vs logistic baseline",
            "state_features": STATE_COLS,
            "data": "StatsBomb Open Data events",
            "limitations": [
                "no Betfair in-play line available for market-vs-model evaluation",
                "red cards not yet parsed from event JSON",
            ],
        }
