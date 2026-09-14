"""A model that sees *only* news-sentiment/injury signal — nothing else.

Deliberately narrow: this exists to answer one question honestly — does the LLM/news-derived
signal (`features/context.py::news_context`, and eventually the LLM-extracted injury-severity
feature) carry any real predictive value on its own? Feeding it into a bigger GBDT would hide
that answer inside a pile of other features; this model isolates it so an ablation/backtest can
show its edge_bits/Sharpe in isolation.

Multinomial logistic regression — deliberately not another GBDT (the point is variety, and this
feature set is small/low-dimensional, exactly logistic regression's sweet spot). If none of its
input columns exist yet (Phase 3's news wiring hasn't run, or no news was scraped for these
teams), it falls back to the training set's raw home/draw/away base rates — an honest "no
signal available yet" prediction rather than an error or a fabricated number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from pitch_edge.models.base import MatchModel, to_frame

SENTIMENT_FEATURES = [
    "news_sent_home",
    "news_sent_away",
    "news_sent_diff",
    "news_injury_items_home",
    "news_injury_items_away",
]


class SentimentOnlyModel(MatchModel):
    name = "sentiment_only"

    def __init__(self):
        self._clf: LogisticRegression | None = None
        self._base_rates = np.array([0.45, 0.25, 0.30])  # overwritten by fit(); a sane football prior

    def _X(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df.reindex(columns=SENTIMENT_FEATURES)
        return X.fillna(0.0)

    def fit(self, train: pd.DataFrame) -> SentimentOnlyModel:
        if "result" in train.columns and len(train):
            counts = train["result"].value_counts(normalize=True)
            self._base_rates = np.array([counts.get(i, 0.0) for i in range(3)])
        has_signal = any(c in train.columns and train[c].notna().any() for c in SENTIMENT_FEATURES)
        if not has_signal or "result" not in train.columns or len(train) < 30:
            self._clf = None
            return self
        X = self._X(train)
        y = train["result"].to_numpy()
        if len(set(y)) < 2:
            self._clf = None
            return self
        self._clf = LogisticRegression(max_iter=500, multi_class="multinomial")
        self._clf.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._clf is None:
            probs = np.tile(self._base_rates, (len(X), 1))
            return to_frame(probs, X.index)
        raw = self._clf.predict_proba(self._X(X))
        # LogisticRegression only learns classes seen in fit(); map back onto the fixed 3-way order.
        full = np.zeros((len(X), 3))
        for j, cls in enumerate(self._clf.classes_):
            full[:, int(cls)] = raw[:, j]
        return to_frame(full, X.index)

    def card(self) -> dict:
        return {**super().card(), "features": SENTIMENT_FEATURES, "has_signal": self._clf is not None}
