"""Post-hoc probability calibration and calibration diagnostics.

Per the project brief, raw model probabilities must be calibrated (isotonic
regression) before they're used for staking decisions, and calibration
quality (Brier score, log-loss, reliability diagram data) must be reported,
not just accuracy.
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Mean squared error between predicted probability and binary outcome."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    return float(np.mean((y_prob - y_true) ** 2))


def log_loss_score(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-15) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), eps, 1 - eps)
    return float(-np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)))


def reliability_curve(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Returns (mean_predicted_prob_per_bin, empirical_frequency_per_bin) for a
    reliability diagram. Empty bins are omitted."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.digitize(y_prob, bin_edges[1:-1])

    mean_pred = []
    empirical_freq = []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        mean_pred.append(y_prob[mask].mean())
        empirical_freq.append(y_true[mask].mean())
    return np.array(mean_pred), np.array(empirical_freq)


class IsotonicCalibrator:
    """Wraps sklearn's isotonic regression for one binary outcome (e.g. "home win")."""

    def __init__(self) -> None:
        self._iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.fitted_ = False

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> IsotonicCalibrator:
        self._iso.fit(np.asarray(y_prob, dtype=float), np.asarray(y_true, dtype=float))
        self.fitted_ = True
        return self

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("IsotonicCalibrator must be fit() before transform()")
        return self._iso.predict(np.asarray(y_prob, dtype=float))
