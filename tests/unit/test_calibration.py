from __future__ import annotations

import numpy as np
import pytest

from pitch_edge.models.calibration import (
    IsotonicCalibrator,
    brier_score,
    log_loss_score,
    reliability_curve,
)

pytestmark = pytest.mark.unit


def test_brier_score_perfect_predictions_is_zero():
    y_true = np.array([1, 0, 1, 0])
    y_prob = np.array([1.0, 0.0, 1.0, 0.0])
    assert brier_score(y_true, y_prob) == pytest.approx(0.0)


def test_brier_score_worst_case_is_one():
    y_true = np.array([1, 0])
    y_prob = np.array([0.0, 1.0])
    assert brier_score(y_true, y_prob) == pytest.approx(1.0)


def test_brier_score_uninformative_half_prediction():
    y_true = np.array([1, 0])
    y_prob = np.array([0.5, 0.5])
    assert brier_score(y_true, y_prob) == pytest.approx(0.25)


def test_log_loss_perfect_predictions_near_zero():
    y_true = np.array([1, 0])
    y_prob = np.array([1 - 1e-10, 1e-10])
    assert log_loss_score(y_true, y_prob) < 1e-6


def test_log_loss_clips_extreme_probabilities():
    # Without clipping this would be -inf/nan; must return a finite number.
    y_true = np.array([1, 0])
    y_prob = np.array([0.0, 1.0])
    result = log_loss_score(y_true, y_prob)
    assert np.isfinite(result)


def test_reliability_curve_shapes_match_and_omit_empty_bins():
    y_true = np.array([1, 1, 0, 0])
    y_prob = np.array([0.9, 0.85, 0.1, 0.15])
    mean_pred, empirical = reliability_curve(y_true, y_prob, n_bins=10)
    assert len(mean_pred) == len(empirical)
    assert len(mean_pred) <= 10


def test_isotonic_calibrator_requires_fit_before_transform():
    calibrator = IsotonicCalibrator()
    with pytest.raises(RuntimeError, match="fit"):
        calibrator.transform(np.array([0.5]))


def test_isotonic_calibrator_improves_or_maintains_calibration():
    rng = np.random.default_rng(0)
    n = 500
    true_prob = rng.uniform(0, 1, n)
    y_true = (rng.uniform(0, 1, n) < true_prob).astype(int)
    # Deliberately mis-calibrated (overconfident) predictions
    miscalibrated = np.clip(true_prob * 1.3 - 0.1, 0.01, 0.99)

    calibrator = IsotonicCalibrator().fit(miscalibrated, y_true)
    calibrated = calibrator.transform(miscalibrated)

    assert brier_score(y_true, calibrated) <= brier_score(y_true, miscalibrated) + 1e-9
    assert (calibrated >= 0).all() and (calibrated <= 1).all()
