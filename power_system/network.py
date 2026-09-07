"""Nonlinear three-generator swing-equation simulation and PMU-like measurements.

The state order is [delta1, omega1, delta2, omega2, delta3, omega3].
``delta`` is an electrical rotor angle in radians and ``omega`` is its
deviation from synchronous speed.  This is intentionally a compact model:
it is a stable demonstrator, rather than a replacement for a full transient
stability package.  The public dynamics and measurement methods are designed
so a fourth-order generator state can be substituted later.
"""

from __future__ import annotations

import numpy as np


class ThreeGeneratorSystem:
    """A coupled nonlinear swing model with nine PMU-like measurements.

    Measurements are ``[delta_1..3, omega_1..3, P_e_1..3]``.  The local
    measurement group for generator *i* is therefore ``[i, 3+i, 6+i]``.
    """

    state_dimension = 6
    measurement_dimension = 9
    state_labels = ("delta1", "omega1", "delta2", "omega2", "delta3", "omega3")
    measurement_labels = (
        "delta1", "delta2", "delta3", "omega1", "omega2", "omega3",
        "electric_power1", "electric_power2", "electric_power3",
    )

    def __init__(self, dt: float = 0.02) -> None:
        self.dt = float(dt)
        self.inertia = np.array([2.7, 3.1, 2.9])
        self.damping = np.array([1.05, 1.15, 1.10])
        # Symmetric transfer susceptances.  The diagonal is unused.
        self.coupling = np.array([[0.0, 1.20, 0.85], [1.20, 0.0, 1.05], [0.85, 1.05, 0.0]])
        self.load_stiffness = np.array([0.62, 0.56, 0.60])
        self.equilibrium_delta = np.array([0.12, -0.07, 0.05])
        self.mechanical_base = self.electrical_power(self.equilibrium_delta)

    @staticmethod
    def partitions() -> list[list[int]]:
        return [[0, 1], [2, 3], [4, 5]]

    @staticmethod
    def partition_measurements(partition_number: int) -> list[int]:
        return [partition_number, 3 + partition_number, 6 + partition_number]

    def electrical_power(self, delta: np.ndarray) -> np.ndarray:
        """Return nonlinear electrical output power for one or many angle vectors."""
        delta = np.asarray(delta, dtype=float)
        if delta.shape[-1] != 3:
            raise ValueError("delta must end in three generator angles")
        differences = delta[..., :, None] - delta[..., None, :]
        transfer = np.sum(self.coupling * np.sin(differences), axis=-1)
        return transfer + self.load_stiffness * np.sin(delta)

    def mechanical_power(self, time: float) -> np.ndarray:
        """Small bounded load/generation changes to excite the nonlinear model."""
        return self.mechanical_base + np.array([
            0.045 * np.sin(0.37 * time),
            0.035 * np.cos(0.29 * time + 0.4),
            0.040 * np.sin(0.33 * time + 1.1),
        ])

    def transition(self, state: np.ndarray, time: float) -> np.ndarray:
        """One explicit-Euler nonlinear transition, supporting shape ``(..., 6)``."""
        state = np.asarray(state, dtype=float)
        if state.shape[-1] != self.state_dimension:
            raise ValueError("state must end in six components")
        delta = state[..., 0::2]
        omega = state[..., 1::2]
        acceleration = (self.mechanical_power(time) - self.damping * omega - self.electrical_power(delta)) / self.inertia
        result = state.copy()
        result[..., 0::2] = delta + self.dt * omega
        result[..., 1::2] = omega + self.dt * acceleration
        return result

    def measurement(self, state: np.ndarray) -> np.ndarray:
        """Map one or many states to PMU-like angle, speed and electric-power data."""
        state = np.asarray(state, dtype=float)
        if state.shape[-1] != self.state_dimension:
            raise ValueError("state must end in six components")
        return np.concatenate((state[..., 0::2], state[..., 1::2], self.electrical_power(state[..., 0::2])), axis=-1)

    def simulate(self, steps: int, initial_state: np.ndarray, process_noise_std: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Produce a finite truth trajectory; process noise is added after transition."""
        if steps < 2:
            raise ValueError("steps must be at least two")
        process_noise_std = np.asarray(process_noise_std, dtype=float)
        states = np.empty((steps, self.state_dimension), dtype=float)
        states[0] = np.asarray(initial_state, dtype=float)
        for k in range(1, steps):
            states[k] = self.transition(states[k - 1], (k - 1) * self.dt)
            states[k] += rng.normal(0.0, process_noise_std, self.state_dimension)
        if not np.all(np.isfinite(states)):
            raise FloatingPointError("non-finite state in simulation")
        return states
