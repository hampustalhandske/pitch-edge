"""Common interface for every match-outcome model.

The backtester, the LangGraph inference node and the dashboard all talk to
this interface only, so Dixon-Coles, gradient boosting and the PyTorch
sequence model are interchangeable and compared on identical footing.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

OUTCOMES = ("home", "draw", "away")


class MatchModel(ABC):
    name: str = "model"
    uses_market_features: bool = False

    @abstractmethod
    def fit(self, train: pd.DataFrame) -> MatchModel:
        """train: feature frame from FeatureBuilder incl. home_goals/away_goals/result/date/teams."""

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a frame indexed like X with columns home, draw, away summing to 1."""

    def observe(self, realized: pd.DataFrame) -> None:
        """Optional: incorporate newly realized matches between retrains (sequence models)."""
        return None

    def card(self) -> dict:
        """Model card metadata; extended by subclasses."""
        return {"name": self.name, "uses_market_features": self.uses_market_features}


def normalise_probs(arr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Clip to `eps` and renormalize each row to sum to 1.

    `np.clip` does not touch `NaN` (a `NaN` compared to anything is `False`), so a row with any
    non-finite raw score — degenerate optimizer output, an unseen-team edge case, whatever a model
    produced — would otherwise pass through as `NaN/NaN/NaN` and silently reach the caller (the
    dashboard, the backtester, a paper-trade proposal), violating "no unvalidated number reaches the
    UI." Any such row is replaced with the uniform distribution and logged, never silently passed on.
    """
    arr = np.asarray(arr, dtype=float)
    bad_rows = ~np.isfinite(arr).all(axis=1)
    if bad_rows.any():
        logger.warning(
            "normalise_probs: %d/%d row(s) had a non-finite probability — replaced with uniform (1/3, 1/3, 1/3)",
            int(bad_rows.sum()),
            len(arr),
        )
        arr = arr.copy()
        arr[bad_rows] = 1.0
    arr = np.clip(arr, eps, None)
    return arr / arr.sum(axis=1, keepdims=True)


def to_frame(arr: np.ndarray, index) -> pd.DataFrame:
    return pd.DataFrame(normalise_probs(arr), columns=list(OUTCOMES), index=index)
