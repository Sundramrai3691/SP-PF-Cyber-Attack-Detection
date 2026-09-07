"""Nonlinear three-generator swing-equation simulation and PMU-like measurements.

The state order is [delta1, omega1, delta2, omega2, delta3, omega3].
``delta`` is an electrical rotor angle in radians and ``omega`` is its
deviation from synchronous speed.  This is intentionally a compact model:
it is a stable demonstrator, rather than a replacement for a full transient
stability package.  The public dynamics and measurement methods are designed
so a fourth-order generator state can be substituted later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


class ThreeGeneratorSystem:
    """A coupled nonlinear swing model with nine PMU-like measurements.

    Measurements are ``[delta_1..3, omega_1..3, P_e_1..3]``.  The local
    measurement group for generator *i* is therefore ``[i, 3+i, 6+i]``.
    """

    model_order = 2
    state_dimension = 6
    measurement_dimension = 9
    state_labels = ("delta1", "omega1", "delta2", "omega2", "delta3", "omega3")
    measurement_labels = (
        "delta1", "delta2", "delta3", "omega1", "omega2", "omega3",
        "electric_power1", "electric_power2", "electric_power3",
    )

    def __init__(self, dt: float = 0.02, mechanical_power_scale: float = 1.0) -> None:
        self.dt = float(dt)
        self.mechanical_power_scale = float(mechanical_power_scale)
        if not np.isfinite(self.mechanical_power_scale) or self.mechanical_power_scale <= 0:
            raise ValueError("mechanical_power_scale must be finite and positive")
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
        return self.mechanical_power_scale * self.mechanical_base + np.array([
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


@dataclass(frozen=True)
class FourthOrderGeneratorParameters:
    """Per-machine prototype parameters for a standard fourth-order model.

    Values are documented engineering assumptions rather than parameters from
    the reference paper. Reactances are in per-unit and time constants are in
    seconds. They can be replaced by validated machine data later.
    """

    H: tuple[float, float, float] = (3.5, 4.0, 3.2)
    D: tuple[float, float, float] = (1.10, 1.20, 1.15)
    Xd: tuple[float, float, float] = (1.80, 1.90, 1.75)
    Xq: tuple[float, float, float] = (1.70, 1.80, 1.65)
    Xd_prime: tuple[float, float, float] = (0.30, 0.32, 0.28)
    Xq_prime: tuple[float, float, float] = (0.55, 0.58, 0.52)
    Td0_prime: tuple[float, float, float] = (5.0, 5.5, 4.8)
    Tq0_prime: tuple[float, float, float] = (0.70, 0.75, 0.65)
    eq_initial: tuple[float, float, float] = (1.05, 1.03, 1.04)


class FourthOrderThreeGeneratorSystem:
    """Three-machine transient-EMF (fourth-order) prototype.

    State order per generator is ``[delta, omega, E'q, E'd]``.  Equations are
    the standard transient-voltage fourth-order form:

    ``delta_dot = omega``
    ``omega_dot = (Pm - Pe - D*omega)/(2H)``
    ``E'q_dot = (Efd - E'q - (Xd-Xd')*Id)/Td0'``
    ``E'd_dot = (-E'd + (Xq-Xq')*Iq)/Tq0'``.

    A lossless coupled internal-EMF network supplies prototype ``Pe, Id, Iq``.
    This is a physically motivated synthetic model, not an exact numerical
    reproduction of a particular paper or a full transient-stability solver.
    """

    model_order = 4
    state_dimension = 12
    measurement_dimension = 15
    state_labels = (
        "delta1", "omega1", "eq1_prime", "ed1_prime",
        "delta2", "omega2", "eq2_prime", "ed2_prime",
        "delta3", "omega3", "eq3_prime", "ed3_prime",
    )
    measurement_labels = (
        "delta1", "delta2", "delta3", "omega1", "omega2", "omega3",
        "eq1_prime", "eq2_prime", "eq3_prime", "ed1_prime", "ed2_prime", "ed3_prime",
        "electric_power1", "electric_power2", "electric_power3",
    )

    def __init__(self, dt: float = 0.02, parameters: FourthOrderGeneratorParameters | None = None) -> None:
        self.dt = float(dt)
        self.parameters = parameters or FourthOrderGeneratorParameters()
        self.H = np.asarray(self.parameters.H, dtype=float)
        self.D = np.asarray(self.parameters.D, dtype=float)
        self.Xd = np.asarray(self.parameters.Xd, dtype=float)
        self.Xq = np.asarray(self.parameters.Xq, dtype=float)
        self.Xd_prime = np.asarray(self.parameters.Xd_prime, dtype=float)
        self.Xq_prime = np.asarray(self.parameters.Xq_prime, dtype=float)
        self.Td0_prime = np.asarray(self.parameters.Td0_prime, dtype=float)
        self.Tq0_prime = np.asarray(self.parameters.Tq0_prime, dtype=float)
        arrays = (self.H, self.D, self.Xd, self.Xq, self.Xd_prime, self.Xq_prime, self.Td0_prime, self.Tq0_prime)
        if any(value.shape != (3,) or not np.all(np.isfinite(value)) for value in arrays):
            raise ValueError("fourth-order parameters must contain three finite values per quantity")
        if np.any(self.H <= 0) or np.any(self.Td0_prime <= 0) or np.any(self.Tq0_prime <= 0):
            raise ValueError("inertia and transient time constants must be positive")
        self.coupling = np.array([[0.0, 1.20, 0.85], [1.20, 0.0, 1.05], [0.85, 1.05, 0.0]])
        self.equilibrium_delta = np.array([0.12, -0.07, 0.05])
        # Determine a stationary prototype operating point for transient EMFs.
        initial = np.zeros(12, dtype=float)
        initial[0::4] = self.equilibrium_delta
        initial[2::4] = np.asarray(self.parameters.eq_initial, dtype=float)
        _, iq = self.dq_currents(initial)
        initial[3::4] = (self.Xq - self.Xq_prime) * iq
        id_current, _ = self.dq_currents(initial)
        self.equilibrium_state = initial
        self.field_voltage = initial[2::4] + (self.Xd - self.Xd_prime) * id_current
        self.mechanical_base = self.electrical_power(initial)

    @staticmethod
    def partitions() -> list[list[int]]:
        return [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]

    @staticmethod
    def partition_measurements(partition_number: int) -> list[int]:
        return [partition_number, 3 + partition_number, 6 + partition_number, 9 + partition_number, 12 + partition_number]

    def _extract(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        state = np.asarray(state, dtype=float)
        if state.shape[-1] != self.state_dimension:
            raise ValueError("fourth-order state must end in 12 components")
        return state[..., 0::4], state[..., 1::4], state[..., 2::4], state[..., 3::4]

    def _emf_polar(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        delta, _, eq, ed = self._extract(state)
        magnitude = np.sqrt(eq ** 2 + ed ** 2)
        angle = delta + np.arctan2(ed, eq)
        return magnitude, angle

    def electrical_power(self, state: np.ndarray) -> np.ndarray:
        """Lossless-network electrical output; transient EMFs affect ``Pe``."""
        magnitude, angle = self._emf_polar(state)
        differences = angle[..., :, None] - angle[..., None, :]
        return np.sum(self.coupling * magnitude[..., :, None] * magnitude[..., None, :] * np.sin(differences), axis=-1)

    def dq_currents(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Prototype d/q currents induced by coupled transient internal EMFs."""
        magnitude, angle = self._emf_polar(state)
        differences = angle[..., :, None] - angle[..., None, :]
        id_current = np.sum(self.coupling * magnitude[..., None, :] * np.sin(differences), axis=-1)
        iq_current = np.sum(self.coupling * (magnitude[..., :, None] - magnitude[..., None, :] * np.cos(differences)), axis=-1)
        return id_current, iq_current

    def mechanical_power(self, time: float) -> np.ndarray:
        return self.mechanical_base + np.array([
            0.045 * np.sin(0.37 * time),
            0.035 * np.cos(0.29 * time + 0.4),
            0.040 * np.sin(0.33 * time + 1.1),
        ])

    def state_derivative(self, state: np.ndarray, time: float, mechanical_power: np.ndarray | None = None) -> np.ndarray:
        """Evaluate standard fourth-order machine derivatives for one/batched states."""
        delta, omega, eq, ed = self._extract(state)
        id_current, iq_current = self.dq_currents(state)
        pm = self.mechanical_power(time) if mechanical_power is None else np.asarray(mechanical_power, dtype=float)
        pe = self.electrical_power(state)
        derivative = np.empty_like(np.asarray(state, dtype=float))
        derivative[..., 0::4] = omega
        derivative[..., 1::4] = (pm - pe - self.D * omega) / (2.0 * self.H)
        derivative[..., 2::4] = (self.field_voltage - eq - (self.Xd - self.Xd_prime) * id_current) / self.Td0_prime
        derivative[..., 3::4] = (-ed + (self.Xq - self.Xq_prime) * iq_current) / self.Tq0_prime
        if not np.all(np.isfinite(derivative)):
            raise FloatingPointError("fourth-order derivative contains NaN/Inf")
        return derivative

    def step(self, state: np.ndarray, time: float) -> np.ndarray:
        """Stable RK4 integration over the configured sample interval."""
        state = np.asarray(state, dtype=float)
        dt = self.dt
        k1 = self.state_derivative(state, time)
        k2 = self.state_derivative(state + 0.5 * dt * k1, time + 0.5 * dt)
        k3 = self.state_derivative(state + 0.5 * dt * k2, time + 0.5 * dt)
        k4 = self.state_derivative(state + dt * k3, time + dt)
        result = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("fourth-order integration contains NaN/Inf")
        return result

    def transition(self, state: np.ndarray, time: float) -> np.ndarray:
        """PF-compatible alias for one RK4 step."""
        return self.step(state, time)

    def measurement(self, state: np.ndarray) -> np.ndarray:
        """Return [delta, omega, E'q, E'd, Pe] for all three generators."""
        delta, omega, eq, ed = self._extract(state)
        return np.concatenate((delta, omega, eq, ed, self.electrical_power(state)), axis=-1)

    def simulate(self, steps: int, initial_state: np.ndarray | None, process_noise_std: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if steps < 2:
            raise ValueError("steps must be at least two")
        noise = np.asarray(process_noise_std, dtype=float)
        if noise.shape != (self.state_dimension,) or np.any(noise < 0):
            raise ValueError("process_noise_std must have 12 non-negative entries")
        states = np.empty((steps, self.state_dimension), dtype=float)
        states[0] = self.equilibrium_state if initial_state is None else np.asarray(initial_state, dtype=float)
        for index in range(1, steps):
            states[index] = self.step(states[index - 1], (index - 1) * self.dt)
            states[index] += rng.normal(0.0, noise, self.state_dimension)
        if not np.all(np.isfinite(states)):
            raise FloatingPointError("fourth-order simulation contains NaN/Inf")
        return states


def make_power_system(model_order: int = 2, dt: float = 0.02) -> ThreeGeneratorSystem | FourthOrderThreeGeneratorSystem:
    """Select the preserved second-order baseline or optional fourth-order model."""
    if model_order == 2:
        return ThreeGeneratorSystem(dt)
    if model_order == 4:
        return FourthOrderThreeGeneratorSystem(dt)
    raise ValueError("model_order must be 2 or 4")
