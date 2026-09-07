"""Fixed partition metadata; future adaptive KL logic belongs here."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np


@dataclass(frozen=True)
class FixedPartition:
    state_indices: tuple[int, ...]
    measurement_indices: tuple[int, ...]
    name: str
    # Empty retains compatibility with the baseline two-state metadata.
    generator_indices: tuple[int, ...] = ()


class StatePartitioner:
    """Validates and combines a fixed non-overlapping state partition."""

    def __init__(self, partitions: Sequence[FixedPartition], state_dimension: int) -> None:
        self.partitions = tuple(partitions)
        self.state_dimension = int(state_dimension)
        flattened = [index for partition in self.partitions for index in partition.state_indices]
        if sorted(flattened) != list(range(self.state_dimension)):
            raise ValueError("partitions must cover every state index exactly once")
        if any(len(p.state_indices) == 0 or len(p.measurement_indices) == 0 for p in self.partitions):
            raise ValueError("partitions cannot be empty")

    @classmethod
    def three_generator_default(cls) -> "StatePartitioner":
        return cls(
            [
                FixedPartition((0, 1), (0, 3, 6), "generator_1", (0,)),
                FixedPartition((2, 3), (1, 4, 7), "generator_2", (1,)),
                FixedPartition((4, 5), (2, 5, 8), "generator_3", (2,)),
            ],
            state_dimension=6,
        )

    @classmethod
    def fourth_order_default(cls) -> "StatePartitioner":
        """Generator-wise initial blocks for [delta, omega, E'q, E'd]."""
        return cls(
            [
                FixedPartition((0, 1, 2, 3), (0, 3, 6, 9, 12), "generator_1", (0,)),
                FixedPartition((4, 5, 6, 7), (1, 4, 7, 10, 13), "generator_2", (1,)),
                FixedPartition((8, 9, 10, 11), (2, 5, 8, 11, 14), "generator_3", (2,)),
            ],
            state_dimension=12,
        )

    def combine(self, local_estimates: Sequence[np.ndarray]) -> np.ndarray:
        if len(local_estimates) != len(self.partitions):
            raise ValueError("one estimate per partition is required")
        state = np.empty(self.state_dimension, dtype=float)
        for partition, estimate in zip(self.partitions, local_estimates):
            estimate = np.asarray(estimate, dtype=float)
            if estimate.shape != (len(partition.state_indices),):
                raise ValueError("local estimate has incorrect dimension")
            state[list(partition.state_indices)] = estimate
        return state

    # Extension point: a later adaptive implementation can calculate KL
    # divergence and return a new StatePartitioner without changing filters.
    def adaptive_update(self, *_: object, **__: object) -> "StatePartitioner":
        return self


@dataclass(frozen=True)
class GaussianApproximation:
    """Weighted Gaussian approximation of an innovation-particle cloud."""

    mean: np.ndarray
    covariance: np.ndarray


def weighted_gaussian(samples: np.ndarray, weights: np.ndarray, regularization: float = 1e-6) -> GaussianApproximation:
    """Fit a finite weighted Gaussian; used only for equal-coordinate signatures."""
    samples = np.asarray(samples, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if samples.ndim != 2 or weights.shape != (samples.shape[0],) or samples.shape[0] < 2:
        raise ValueError("samples must be (N, dimension) with N matching weights")
    if not np.all(np.isfinite(samples)) or not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("Gaussian inputs must be finite with non-negative weights")
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("Gaussian weights must have positive total")
    normalised = weights / total
    mean = np.average(samples, axis=0, weights=normalised)
    centered = samples - mean
    covariance = (centered * normalised[:, None]).T @ centered
    covariance = 0.5 * (covariance + covariance.T) + np.eye(samples.shape[1]) * regularization
    return GaussianApproximation(mean, covariance)


def gaussian_kl(first: GaussianApproximation, second: GaussianApproximation, regularization: float = 1e-9) -> float:
    """KL(N_first || N_second), regularized for covariance conditioning."""
    mean_a = np.asarray(first.mean, dtype=float)
    mean_b = np.asarray(second.mean, dtype=float)
    covariance_a = np.asarray(first.covariance, dtype=float)
    covariance_b = np.asarray(second.covariance, dtype=float)
    dimension = mean_a.size
    if mean_b.shape != (dimension,) or covariance_a.shape != (dimension, dimension) or covariance_b.shape != (dimension, dimension):
        raise ValueError("Gaussian dimensions must match")
    covariance_a = 0.5 * (covariance_a + covariance_a.T) + np.eye(dimension) * regularization
    covariance_b = 0.5 * (covariance_b + covariance_b.T) + np.eye(dimension) * regularization
    try:
        solve_a = np.linalg.solve(covariance_b, covariance_a)
        difference = mean_b - mean_a
        quadratic = float(difference @ np.linalg.solve(covariance_b, difference))
        sign_a, logdet_a = np.linalg.slogdet(covariance_a)
        sign_b, logdet_b = np.linalg.slogdet(covariance_b)
    except np.linalg.LinAlgError as error:
        raise ValueError("Gaussian covariance is singular after regularization") from error
    if sign_a <= 0 or sign_b <= 0:
        raise ValueError("Gaussian covariance is not positive definite")
    value = 0.5 * (float(np.trace(solve_a)) + quadratic - dimension + logdet_b - logdet_a)
    if not np.isfinite(value):
        raise FloatingPointError("KL divergence is non-finite")
    return float(max(value, 0.0))


def symmetric_gaussian_kl(first: GaussianApproximation, second: GaussianApproximation) -> float:
    """Jeffreys-style symmetric KL of equal-dimensional innovation Gaussians."""
    value = 0.5 * (gaussian_kl(first, second) + gaussian_kl(second, first))
    if not np.isfinite(value):
        raise FloatingPointError("symmetric KL divergence is non-finite")
    return float(value)


@dataclass(frozen=True)
class AdaptiveKLConfig:
    """Hysteretic adaptation settings for generator-block partitioning.

    The compared distributions are weighted Gaussian approximations of each
    generator's *whitened local measurement-innovation particles*, expressed
    in the common [delta, omega, E'q, E'd, Pe] coordinate order. Low symmetric
    KL means two generator innovation distributions are similar enough to
    merge; high KL within an existing merged block means they split back into
    their deterministic generator-wise 4-state blocks.
    """

    # Innovation coordinates are whitened but their fitted covariances can be
    # tight; values are calibrated prototype thresholds, not paper constants.
    merge_threshold: float = 50_000.0
    split_threshold: float = 500_000.0
    adaptation_interval: int = 20
    minimum_partition_lifetime: int = 40


class AdaptiveKLPartitioner:
    """Proposes stable generator-block merge/split operations from KL scores."""

    def __init__(self, config: AdaptiveKLConfig = AdaptiveKLConfig()) -> None:
        if not (0 <= config.merge_threshold < config.split_threshold):
            raise ValueError("merge threshold must be non-negative and below split threshold")
        if config.adaptation_interval < 1 or config.minimum_partition_lifetime < 0:
            raise ValueError("adaptation interval and lifetime must be valid")
        self.config = config
        self.last_change_index = -config.minimum_partition_lifetime
        self.events: list[dict[str, object]] = []
        self.diagnostics: list[dict[str, object]] = []

    @staticmethod
    def _generator_partition(generator_indices: tuple[int, ...]) -> FixedPartition:
        states: list[int] = []
        measurements: list[int] = []
        for generator in generator_indices:
            states.extend(range(4 * generator, 4 * generator + 4))
            measurements.extend((generator, 3 + generator, 6 + generator, 9 + generator, 12 + generator))
        name = "generators_" + "_".join(str(generator + 1) for generator in generator_indices)
        return FixedPartition(tuple(states), tuple(measurements), name, tuple(generator_indices))

    def should_adapt(self, index: int) -> bool:
        return index % self.config.adaptation_interval == 0 and index - self.last_change_index >= self.config.minimum_partition_lifetime

    def propose(self, current: StatePartitioner, signatures: dict[int, GaussianApproximation], index: int) -> tuple[StatePartitioner, dict[str, object] | None]:
        """Return a new valid partitioner plus event when a hysteretic rule fires."""
        if not self.should_adapt(index):
            return current, None
        pairwise = {
            f"{left + 1}-{right + 1}": symmetric_gaussian_kl(signatures[left], signatures[right])
            for left in range(3) for right in range(left + 1, 3)
        }
        old_indices = [list(partition.state_indices) for partition in current.partitions]
        diagnostic = {"index": index, "partitions": old_indices, "pairwise_symmetric_kl": pairwise,
                      "merge_threshold": self.config.merge_threshold, "split_threshold": self.config.split_threshold}
        self.diagnostics.append(diagnostic)

        # Split first: a merged block with unlike generator innovation clouds
        # returns to deterministic [delta, omega, E'q, E'd] generator blocks.
        for position, partition in enumerate(current.partitions):
            generators = partition.generator_indices
            if len(generators) > 1:
                internal = [pairwise[f"{left + 1}-{right + 1}"] for left in generators for right in generators if left < right]
                maximum = max(internal)
                if maximum >= self.config.split_threshold:
                    new_parts = list(current.partitions)
                    new_parts[position:position + 1] = [self._generator_partition((generator,)) for generator in generators]
                    proposed = StatePartitioner(new_parts, state_dimension=12)
                    event = {**diagnostic, "old_partitions": old_indices, "new_partitions": [list(p.state_indices) for p in proposed.partitions],
                             "operation": "split", "kl_value": maximum, "threshold": self.config.split_threshold,
                             "reason": "merged generator innovation distributions became heterogeneous", "number_of_partitions": len(proposed.partitions)}
                    self.last_change_index = index; self.events.append(event)
                    return proposed, event

        # Merge only singleton generator blocks. This keeps the first adaptive
        # prototype interpretable and prevents merging all machines at once.
        candidates: list[tuple[float, int, int]] = []
        for left, first in enumerate(current.partitions):
            if len(first.generator_indices) != 1:
                continue
            for right, second in enumerate(current.partitions[left + 1:], start=left + 1):
                if len(second.generator_indices) == 1:
                    first_generator, second_generator = first.generator_indices[0], second.generator_indices[0]
                    candidates.append((pairwise[f"{min(first_generator, second_generator) + 1}-{max(first_generator, second_generator) + 1}"], left, right))
        if candidates:
            value, left, right = min(candidates)
            if value <= self.config.merge_threshold:
                combined = tuple(sorted(current.partitions[left].generator_indices + current.partitions[right].generator_indices))
                # Candidate generators need not be adjacent in the partition
                # list; retain every intervening partition rather than slicing
                # it away.
                new_parts = [
                    self._generator_partition(combined) if position == left else partition
                    for position, partition in enumerate(current.partitions) if position != right
                ]
                proposed = StatePartitioner(new_parts, state_dimension=12)
                event = {**diagnostic, "old_partitions": old_indices, "new_partitions": [list(p.state_indices) for p in proposed.partitions],
                         "operation": "merge", "kl_value": value, "threshold": self.config.merge_threshold,
                         "reason": "singleton generator innovation distributions were sufficiently similar", "number_of_partitions": len(proposed.partitions)}
                self.last_change_index = index; self.events.append(event)
                return proposed, event
        return current, None
