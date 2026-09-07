"""Residual and particle-likelihood attack detectors.

``ResidualDetector`` is the original prototype J-statistic detector.  The
likelihood detector below is deliberately separate: it uses the *predictive
measurement likelihood of the particle clouds* produced by SP-PF, rather than
renaming a residual norm as a likelihood-ratio statistic.
"""

from __future__ import annotations

import numpy as np


def _regularized_cholesky(covariance: np.ndarray) -> np.ndarray:
    """Return a Cholesky factor after small, explicit covariance regularisation.

    Measurement covariances are expected to be positive definite.  The tiny
    diagonal loading only protects against round-off; a genuinely indefinite
    covariance still raises a clear error rather than being silently used.
    """
    covariance = np.asarray(covariance, dtype=float)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("measurement covariance must be square")
    if not np.all(np.isfinite(covariance)):
        raise ValueError("measurement covariance must be finite")
    covariance = 0.5 * (covariance + covariance.T)
    scale = max(float(np.max(np.abs(np.diag(covariance)))), 1.0)
    for multiplier in (1e-12, 1e-10, 1e-8):
        try:
            return np.linalg.cholesky(covariance + np.eye(covariance.shape[0]) * scale * multiplier)
        except np.linalg.LinAlgError:
            continue
    raise ValueError("measurement covariance is not positive definite after regularisation")


def logsumexp(values: np.ndarray) -> float:
    """Stable ``log(sum(exp(values)))`` for finite one-dimensional inputs."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("logsumexp requires a non-empty finite one-dimensional array")
    maximum = float(np.max(values))
    result = maximum + float(np.log(np.sum(np.exp(values - maximum))))
    if not np.isfinite(result):
        raise FloatingPointError("logsumexp produced NaN/Inf")
    return result


def compute_particle_log_likelihood(
    measurement: np.ndarray,
    predicted_measurements: np.ndarray,
    measurement_covariance: np.ndarray,
) -> np.ndarray:
    """Evaluate ``log p(y_k | x_k^(i))`` for every particle.

    The expression is the full multivariate Gaussian measurement log-density

    ``-0.5 * ((y-h_i)^T R^-1 (y-h_i) + log|R| + m log(2*pi))``.

    This is the same Gaussian measurement-noise model used by the particle
    filter weight update; retaining the normalising constant makes the output
    a genuine, comparable log likelihood rather than only a residual score.
    """
    measurement = np.asarray(measurement, dtype=float)
    predicted_measurements = np.asarray(predicted_measurements, dtype=float)
    if measurement.ndim != 1 or predicted_measurements.ndim != 2 or predicted_measurements.shape[1] != measurement.size:
        raise ValueError("particle measurement dimensions do not match observation")
    if predicted_measurements.shape[0] < 1 or not (np.all(np.isfinite(measurement)) and np.all(np.isfinite(predicted_measurements))):
        raise ValueError("particle measurements and observation must be finite")
    factor = _regularized_cholesky(measurement_covariance)
    residual = measurement[None, :] - predicted_measurements
    whitened = np.linalg.solve(factor, residual.T).T
    quadratic = np.sum(whitened * whitened, axis=1)
    log_determinant = 2.0 * float(np.sum(np.log(np.diag(factor))))
    dimension = measurement.size
    values = -0.5 * (quadratic + log_determinant + dimension * np.log(2.0 * np.pi))
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("particle log likelihood contains NaN/Inf")
    return values


def compute_weighted_particle_log_likelihood(particle_log_likelihoods: np.ndarray, weights: np.ndarray) -> float:
    """Compute ``log(sum_i w_i p(y_k | x_k^(i)))`` with log-sum-exp."""
    particle_log_likelihoods = np.asarray(particle_log_likelihoods, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if particle_log_likelihoods.ndim != 1 or weights.shape != particle_log_likelihoods.shape:
        raise ValueError("particle likelihoods and weights must be matching vectors")
    if not np.all(np.isfinite(particle_log_likelihoods)) or not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("particle likelihoods must be finite and weights non-negative")
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("particle weights must have positive total")
    normalised = weights / total
    return logsumexp(np.log(np.maximum(normalised, np.finfo(float).tiny)) + particle_log_likelihoods)


def compute_global_log_likelihood(partition_log_likelihoods: np.ndarray) -> float:
    """Combine conditionally factorised SP-PF partition likelihoods in log form."""
    partition_log_likelihoods = np.asarray(partition_log_likelihoods, dtype=float)
    if partition_log_likelihoods.ndim != 1 or partition_log_likelihoods.size == 0 or not np.all(np.isfinite(partition_log_likelihoods)):
        raise ValueError("partition log likelihoods must be a non-empty finite vector")
    value = float(np.sum(partition_log_likelihoods))
    if not np.isfinite(value):
        raise FloatingPointError("global log likelihood is non-finite")
    return value


def compute_log_likelihood_ratio(log_likelihood: float, reference_log_likelihood: float) -> float:
    """Return ``log(L_reference / L_current)`` without raw-likelihood division.

    The current prototype has no separately parameterised attack likelihood.
    Its likelihood-ratio evidence is therefore a normal-reference degradation:
    ``L_reference = exp(mean normal log predictive likelihood)`` and
    ``L_current`` is the SP-PF predictive likelihood under H0.  Larger positive
    values mean the measurement is less likely under the normal model.
    """
    value = float(reference_log_likelihood) - float(log_likelihood)
    if not np.isfinite(value):
        raise FloatingPointError("log likelihood ratio is non-finite")
    return value


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


class LikelihoodRatioDetector:
    """One-sided normal-reference likelihood-ratio detector for SP-PF.

    Calibration uses attack-free global predictive log likelihoods ``ell_k``.
    Let ``mu_0`` and ``sigma_0`` be their mean and standard deviation.  The
    per-sample score is ``g_k = mu_0 - ell_k = log(L_ref / L_k)`` and the
    threshold is ``threshold_multiplier * sigma_0``.  This is a calibrated
    likelihood-degradation test, not a claim of the paper's exact H1 model.
    """

    def __init__(self, threshold_multiplier: float = 4.0, minimum_std: float = 1e-12) -> None:
        if threshold_multiplier <= 0.0 or minimum_std <= 0.0:
            raise ValueError("threshold multiplier and minimum standard deviation must be positive")
        self.threshold_multiplier = float(threshold_multiplier)
        self.minimum_std = float(minimum_std)
        self.reference_log_likelihood: float | None = None
        self.reference_std: float | None = None
        self.threshold: float | None = None

    def calibrate(self, normal_log_likelihoods: np.ndarray) -> float:
        values = np.asarray(normal_log_likelihoods, dtype=float)
        if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
            raise ValueError("normal log likelihood calibration requires at least two finite samples")
        self.reference_log_likelihood = float(np.mean(values))
        self.reference_std = max(float(np.std(values)), self.minimum_std)
        self.threshold = float(self.threshold_multiplier * self.reference_std)
        return self.threshold

    def score(self, log_likelihood: float) -> float:
        if self.reference_log_likelihood is None:
            raise RuntimeError("calibrate the likelihood detector before scoring")
        return compute_log_likelihood_ratio(float(log_likelihood), self.reference_log_likelihood)

    def scores(self, log_likelihoods: np.ndarray) -> np.ndarray:
        values = np.asarray(log_likelihoods, dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("log likelihoods must be finite")
        scores = np.asarray([self.score(value) for value in values], dtype=float)
        if not np.all(np.isfinite(scores)):
            raise FloatingPointError("likelihood scores contain NaN/Inf")
        return scores

    def flags(self, scores: np.ndarray) -> np.ndarray:
        if self.threshold is None:
            raise RuntimeError("calibrate the likelihood detector before requesting flags")
        scores = np.asarray(scores, dtype=float)
        if not np.all(np.isfinite(scores)):
            raise ValueError("likelihood scores must be finite")
        return scores > self.threshold
