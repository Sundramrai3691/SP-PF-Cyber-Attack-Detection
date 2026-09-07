"""Controllable, time-bounded false-data-injection attack."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class FDIAConfig:
    start_index: int = 120
    end_index: int = 190
    magnitude: float = 0.30
    measurement_indices: tuple[int, ...] = (6, 7)
    # Relative attacks use magnitude*y. The floor keeps near-zero power data
    # observable while preserving the same signed percentage interpretation.
    relative_floor: float = 0.05


def inject_fdia(measurements: np.ndarray, config: FDIAConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return attacked data, boolean attack mask and the injected vector.

    The attack is ``a_i = alpha * sign(y_i) * max(abs(y_i), floor)`` for the
    configured time interval and measurement channels.
    """
    measurements = np.asarray(measurements, dtype=float)
    if measurements.ndim != 2:
        raise ValueError("measurements must have shape (time, measurement_dimension)")
    n_steps, n_measurements = measurements.shape
    if not (0 <= config.start_index < config.end_index <= n_steps):
        raise ValueError("FDIA interval must satisfy 0 <= start < end <= number of time steps")
    indices = np.asarray(config.measurement_indices, dtype=int)
    if indices.size == 0 or np.any(indices < 0) or np.any(indices >= n_measurements):
        raise ValueError("FDIA measurement indices are empty or outside the measurement vector")
    attacked = measurements.copy()
    attack_vector = np.zeros_like(measurements)
    selected = measurements[config.start_index:config.end_index, indices]
    sign = np.where(selected >= 0.0, 1.0, -1.0)
    attack_vector[config.start_index:config.end_index, indices] = config.magnitude * sign * np.maximum(np.abs(selected), config.relative_floor)
    attacked += attack_vector
    attack_mask = np.any(np.abs(attack_vector) > 0.0, axis=1)
    if not np.all(np.isfinite(attacked)):
        raise FloatingPointError("FDIA generated non-finite measurement values")
    return attacked, attack_mask, attack_vector
