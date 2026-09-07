"""Configurable partial/full measurement replay attack."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ReplayConfig:
    start_index: int
    end_index: int
    delay_steps: int
    measurement_indices: tuple[int, ...]
    # 1.0 is a full replay. Lower values blend historical and present values,
    # providing a clearly labelled replay-intensity prototype sweep.
    blend: float = 1.0
    reference_start_index: int | None = None


def inject_replay(measurements: np.ndarray, config: ReplayConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replay delayed selected measurements over a bounded interval.

    With no explicit reference start, sample k receives the reference sample
    k-delay_steps. With reference_start_index, the source is a contiguous
    window beginning there. The returned vector is the actual perturbation.
    """
    measurements = np.asarray(measurements, dtype=float)
    if measurements.ndim != 2:
        raise ValueError("measurements must be two-dimensional")
    steps, dimension = measurements.shape
    indices = np.asarray(config.measurement_indices, dtype=int)
    if not (0 <= config.start_index < config.end_index <= steps) or config.delay_steps < 1:
        raise ValueError("replay interval and positive delay must be valid")
    if indices.size == 0 or np.any(indices < 0) or np.any(indices >= dimension):
        raise ValueError("replay measurement indices are invalid")
    if not (0.0 <= config.blend <= 1.0):
        raise ValueError("replay blend must be within [0, 1]")
    length = config.end_index - config.start_index
    source_start = config.start_index - config.delay_steps if config.reference_start_index is None else config.reference_start_index
    source_end = source_start + length
    if source_start < 0 or source_end > steps:
        raise ValueError("replay reference window is outside available measurements")
    attacked = measurements.copy()
    source = measurements[source_start:source_end, indices]
    present = measurements[config.start_index:config.end_index, indices]
    attacked[config.start_index:config.end_index, indices] = (1.0 - config.blend) * present + config.blend * source
    vector = attacked - measurements
    # Intensity zero is explicitly the attack-free control.
    mask = np.zeros(steps, dtype=bool)
    if config.blend > 0.0:
        mask[config.start_index:config.end_index] = True
    if not np.all(np.isfinite(attacked)):
        raise FloatingPointError("replay attack generated NaN/Inf")
    return attacked, mask, vector
