"""Fixed State Partition-Particle Filter (SP-PF) implementation."""

from __future__ import annotations

from typing import Sequence
import numpy as np

from power_system.network import ThreeGeneratorSystem
from .particle_filter import ParticleFilter
from .partition import StatePartitioner


class StatePartitionParticleFilter:
    """One conditional particle filter per state partition.

    Coupled generator power is evaluated using particle values for the local
    generator and the previous global estimate as context for other machines.
    This is the fixed-partition approximation: every partition owns particles
    and weights while their interactions are exchanged through the combined
    global state after each sample.
    """

    def __init__(
        self,
        system: ThreeGeneratorSystem,
        partitioner: StatePartitioner,
        particle_count: int,
        initial_state: np.ndarray,
        initial_covariance: np.ndarray,
        process_noise_covariance: np.ndarray,
        measurement_noise_covariance: np.ndarray,
        rng: np.random.Generator,
    ) -> None:
        self.system = system
        self.partitioner = partitioner
        self.process_noise_covariance = np.asarray(process_noise_covariance, dtype=float)
        self.measurement_noise_covariance = np.asarray(measurement_noise_covariance, dtype=float)
        self.filters: list[ParticleFilter] = []
        self.global_estimate = np.asarray(initial_state, dtype=float).copy()
        for partition in partitioner.partitions:
            state_indices = np.asarray(partition.state_indices)
            pf = ParticleFilter(particle_count, len(state_indices), rng)
            pf.initialize(self.global_estimate[state_indices], initial_covariance[np.ix_(state_indices, state_indices)])
            self.filters.append(pf)

    def _local_full_states(self, particles: np.ndarray, state_indices: Sequence[int], context: np.ndarray) -> np.ndarray:
        full = np.broadcast_to(context, (particles.shape[0], context.size)).copy()
        full[:, list(state_indices)] = particles
        return full

    def step(self, measurement: np.ndarray, time: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Advance all partitions; return posterior state, prior measurement, Neff."""
        measurement = np.asarray(measurement, dtype=float)
        context = self.global_estimate.copy()
        local_estimates: list[np.ndarray] = []
        prior_measurement = np.empty(self.system.measurement_dimension, dtype=float)
        effective_sizes: list[float] = []
        for partition, pf in zip(self.partitioner.partitions, self.filters):
            state_indices = np.asarray(partition.state_indices)
            measurement_indices = np.asarray(partition.measurement_indices)
            local_q = self.process_noise_covariance[np.ix_(state_indices, state_indices)]
            local_r = self.measurement_noise_covariance[np.ix_(measurement_indices, measurement_indices)]

            def local_transition(local_particles: np.ndarray) -> np.ndarray:
                full = self._local_full_states(local_particles, state_indices, context)
                return self.system.transition(full, time)[:, state_indices]

            pf.predict(local_transition, local_q)
            full_particles = self._local_full_states(pf.particles, state_indices, context)
            particle_measurements = self.system.measurement(full_particles)[:, measurement_indices]
            prior_measurement[measurement_indices] = np.average(particle_measurements, axis=0, weights=pf.weights)
            pf.update(measurement[measurement_indices], particle_measurements, local_r)
            if pf.effective_sample_size() < 0.5 * pf.particle_count:
                pf.systematic_resample()
            local_estimates.append(pf.estimate())
            effective_sizes.append(pf.effective_sample_size())
        self.global_estimate = self.partitioner.combine(local_estimates)
        if not (np.all(np.isfinite(self.global_estimate)) and np.all(np.isfinite(prior_measurement))):
            raise FloatingPointError("SP-PF produced non-finite output")
        return self.global_estimate.copy(), prior_measurement, np.asarray(effective_sizes)
