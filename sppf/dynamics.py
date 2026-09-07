"""Fixed State Partition-Particle Filter (SP-PF) implementation."""

from __future__ import annotations

from typing import Sequence
import numpy as np

from power_system.network import ThreeGeneratorSystem
from .particle_filter import ParticleFilter
from .partition import AdaptiveKLPartitioner, StatePartitioner, weighted_gaussian


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
        partition_mode: str = "fixed",
        adaptive_partitioner: AdaptiveKLPartitioner | None = None,
    ) -> None:
        if partition_mode not in {"fixed", "adaptive_kl"}:
            raise ValueError("partition_mode must be 'fixed' or 'adaptive_kl'")
        if partition_mode == "adaptive_kl" and adaptive_partitioner is None:
            raise ValueError("adaptive_kl mode requires an AdaptiveKLPartitioner")
        if partition_mode == "adaptive_kl" and system.state_dimension != 12:
            raise ValueError("adaptive_kl prototype currently requires the 12-state fourth-order system")
        self.system = system
        self.model_order = getattr(system, "model_order", None)
        self.partitioner = partitioner
        self.partition_mode = partition_mode
        self.adaptive_partitioner = adaptive_partitioner
        self.rng = rng
        self.process_noise_covariance = np.asarray(process_noise_covariance, dtype=float)
        self.measurement_noise_covariance = np.asarray(measurement_noise_covariance, dtype=float)
        self.filters: list[ParticleFilter] = []
        self.global_estimate = np.asarray(initial_state, dtype=float).copy()
        for partition in partitioner.partitions:
            state_indices = np.asarray(partition.state_indices)
            pf = ParticleFilter(particle_count, len(state_indices), rng)
            pf.initialize(self.global_estimate[state_indices], initial_covariance[np.ix_(state_indices, state_indices)])
            self.filters.append(pf)
        self.partition_history: list[dict[str, object]] = []
        self.kl_diagnostics: list[dict[str, object]] = []
        self.partition_count_history: list[int] = []

    def _local_full_states(self, particles: np.ndarray, state_indices: Sequence[int], context: np.ndarray) -> np.ndarray:
        full = np.broadcast_to(context, (particles.shape[0], context.size)).copy()
        full[:, list(state_indices)] = particles
        return full

    def _innovation_signatures(self, measurement: np.ndarray) -> dict[int, object]:
        """Fit one whitened [delta,omega,E'q,E'd,Pe] innovation Gaussian/gen."""
        signatures: dict[int, object] = {}
        context = self.global_estimate.copy()
        noise_std = np.sqrt(np.maximum(np.diag(self.measurement_noise_covariance), 1e-15))
        for partition, pf in zip(self.partitioner.partitions, self.filters):
            full_particles = self._local_full_states(pf.particles, partition.state_indices, context)
            predicted = self.system.measurement(full_particles)
            for generator in partition.generator_indices:
                measurement_indices = np.asarray((generator, 3 + generator, 6 + generator, 9 + generator, 12 + generator))
                innovation = (measurement[measurement_indices][None, :] - predicted[:, measurement_indices]) / noise_std[measurement_indices]
                signatures[generator] = weighted_gaussian(innovation, pf.weights)
        if set(signatures) != {0, 1, 2}:
            raise ValueError("adaptive KL signatures must cover all three generators")
        return signatures

    def _reconfigure_filters(self, new_partitioner: StatePartitioner) -> None:
        """Approximate a joint cloud from old marginals, then project to blocks.

        Each old independent local posterior is sampled according to its
        normalized weights.  Concatenating those samples preserves every old
        marginal and creates dimensionally consistent particles for each new
        merged or split state block. New local filters start uniformly weighted
        because the resampled cloud already carries the posterior weights.
        """
        count = self.filters[0].particle_count
        joint_particles = np.empty((count, self.partitioner.state_dimension), dtype=float)
        for partition, filter_ in zip(self.partitioner.partitions, self.filters):
            selected = self.rng.choice(count, size=count, replace=True, p=filter_.weights)
            joint_particles[:, list(partition.state_indices)] = filter_.particles[selected]
        new_filters: list[ParticleFilter] = []
        for partition in new_partitioner.partitions:
            filter_ = ParticleFilter(count, len(partition.state_indices), self.rng)
            filter_.set_particles(joint_particles[:, list(partition.state_indices)])
            if not np.isclose(filter_.weights.sum(), 1.0) or not np.all(np.isfinite(filter_.particles)):
                raise FloatingPointError("invalid particle population after repartitioning")
            new_filters.append(filter_)
        self.partitioner = new_partitioner
        self.filters = new_filters

    def step(self, measurement: np.ndarray, time: float, index: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        if self.partition_mode == "adaptive_kl":
            if index is None:
                raise ValueError("adaptive_kl mode requires a sample index on every step")
            assert self.adaptive_partitioner is not None
            signatures = self._innovation_signatures(measurement)
            new_partitioner, event = self.adaptive_partitioner.propose(self.partitioner, signatures, index)
            if self.adaptive_partitioner.diagnostics:
                newest = self.adaptive_partitioner.diagnostics[-1]
                if newest.get("index") == index:
                    self.kl_diagnostics.append(newest)
            if event is not None:
                event["time_seconds"] = float(time)
                self._reconfigure_filters(new_partitioner)
                self.partition_history.append(event)
        self.partition_count_history.append(len(self.partitioner.partitions))
        return self.global_estimate.copy(), prior_measurement, np.asarray(effective_sizes)
