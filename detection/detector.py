"""Prototype residual/J-statistic detector, deliberately not a paper LRT."""

from __future__ import annotations

import numpy as np


class ResidualDetector:
    def __init__(self, measurement_covariance: np.ndarray, threshold_multiplier: float = 4.0) -> None:
        covariance = np.asarray(measurement_covariance, dtype=float)
        self.inverse_covariance = np.linalg.inv(covariance)
        self.threshold_multiplier = float(threshold_multiplier)
        self.threshold: float | None = None

    def score(self, measurement: np.ndarray, predicted_measurement: np.ndarray) -> float:
        residual = np.asarray(measurement, dtype=float) - np.asarray(predicted_measurement, dtype=float)
        value = float(residual @ self.inverse_covariance @ residual)
        if not np.isfinite(value):
            raise FloatingPointError("detector score is non-finite")
        return value

    def calibrate(self, normal_scores: np.ndarray) -> float:
        normal_scores = np.asarray(normal_scores, dtype=float)
        if normal_scores.size < 2 or not np.all(np.isfinite(normal_scores)):
            raise ValueError("normal calibration scores must be finite and contain at least two values")
        self.threshold = float(np.mean(normal_scores) + self.threshold_multiplier * np.std(normal_scores))
        return self.threshold

    def flags(self, scores: np.ndarray) -> np.ndarray:
        if self.threshold is None:
            raise RuntimeError("calibrate the detector before requesting flags")
        return np.asarray(scores, dtype=float) > self.threshold
