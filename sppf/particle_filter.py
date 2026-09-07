"""Numerically stable generic NumPy particle filter."""

from __future__ import annotations

import numpy as np


class ParticleFilter:
    def __init__(self, particle_count: int, state_dimension: int, rng: np.random.Generator) -> None:
        if particle_count < 2 or state_dimension < 1:
            raise ValueError("particle_count must be >= 2 and state_dimension >= 1")
        self.particle_count = int(particle_count)
        self.state_dimension = int(state_dimension)
        self.rng = rng
        self.particles = np.empty((particle_count, state_dimension), dtype=float)
        self.weights = np.full(particle_count, 1.0 / particle_count, dtype=float)

    def initialize(self, mean: np.ndarray, covariance: np.ndarray) -> None:
        mean = np.asarray(mean, dtype=float)
        covariance = np.asarray(covariance, dtype=float)
        if mean.shape != (self.state_dimension,) or covariance.shape != (self.state_dimension, self.state_dimension):
            raise ValueError("initial mean or covariance has incorrect shape")
        covariance = 0.5 * (covariance + covariance.T) + np.eye(self.state_dimension) * 1e-12
        self.particles = self.rng.multivariate_normal(mean, covariance, size=self.particle_count)
        self.weights.fill(1.0 / self.particle_count)
        self._check_finite()

    def predict(self, transition, process_noise_covariance: np.ndarray) -> None:
        predicted = np.asarray(transition(self.particles), dtype=float)
        covariance = np.asarray(process_noise_covariance, dtype=float)
        if predicted.shape != self.particles.shape or covariance.shape != (self.state_dimension, self.state_dimension):
            raise ValueError("prediction shape does not match particle state")
        covariance = 0.5 * (covariance + covariance.T) + np.eye(self.state_dimension) * 1e-12
        self.particles = predicted + self.rng.multivariate_normal(np.zeros(self.state_dimension), covariance, size=self.particle_count)
        self._check_finite()

    def predicted_mean(self) -> np.ndarray:
        return np.average(self.particles, axis=0, weights=self.weights)

    def update(self, observation: np.ndarray, predicted_observations: np.ndarray, measurement_covariance: np.ndarray) -> None:
        observation = np.asarray(observation, dtype=float)
        predicted_observations = np.asarray(predicted_observations, dtype=float)
        covariance = np.asarray(measurement_covariance, dtype=float)
        dimension = observation.size
        if predicted_observations.shape != (self.particle_count, dimension) or covariance.shape != (dimension, dimension):
            raise ValueError("measurement dimensions do not match particle filter")
        covariance = 0.5 * (covariance + covariance.T) + np.eye(dimension) * 1e-12
        residual = observation[None, :] - predicted_observations
        try:
            solved = np.linalg.solve(covariance, residual.T).T
        except np.linalg.LinAlgError as error:
            raise ValueError("measurement covariance is singular") from error
        log_likelihood = -0.5 * np.sum(residual * solved, axis=1)
        # Log normalization avoids likelihood underflow for sharp measurements.
        log_weights = np.log(np.maximum(self.weights, np.finfo(float).tiny)) + log_likelihood
        log_weights -= np.max(log_weights)
        weights = np.exp(log_weights)
        total = float(np.sum(weights))
        if not np.isfinite(total) or total <= 0.0:
            self.weights.fill(1.0 / self.particle_count)
        else:
            self.weights = weights / total
        self._check_finite()

    def effective_sample_size(self) -> float:
        value = 1.0 / float(np.sum(self.weights ** 2))
        return float(np.clip(value, 1.0, self.particle_count))

    def systematic_resample(self) -> None:
        positions = (self.rng.random() + np.arange(self.particle_count)) / self.particle_count
        cumulative = np.cumsum(self.weights)
        indices = np.searchsorted(cumulative, positions, side="right")
        indices = np.minimum(indices, self.particle_count - 1)
        self.particles = self.particles[indices].copy()
        self.weights.fill(1.0 / self.particle_count)

    def estimate(self) -> np.ndarray:
        estimate = np.average(self.particles, axis=0, weights=self.weights)
        if not np.all(np.isfinite(estimate)):
            raise FloatingPointError("particle estimate is non-finite")
        return estimate

    def _check_finite(self) -> None:
        if not np.all(np.isfinite(self.particles)) or not np.all(np.isfinite(self.weights)):
            raise FloatingPointError("particles or weights contain NaN/Inf")
        total = float(np.sum(self.weights))
        if total <= 0.0:
            raise FloatingPointError("particle weights have zero total")
        self.weights /= total
