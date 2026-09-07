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
                FixedPartition((0, 1), (0, 3, 6), "generator_1"),
                FixedPartition((2, 3), (1, 4, 7), "generator_2"),
                FixedPartition((4, 5), (2, 5, 8), "generator_3"),
            ],
            state_dimension=6,
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
