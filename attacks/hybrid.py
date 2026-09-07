"""Composed replay-plus-FDIA attack without modifying either primitive."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .fdia import FDIAConfig, inject_fdia
from .replay import ReplayConfig, inject_replay


@dataclass(frozen=True)
class HybridConfig:
    replay: ReplayConfig
    fdia: FDIAConfig
    # Replay then FDIA means the injected data attack perturbs the replayed
    # signal; this order is explicit and reproducible.
    order: str = "replay_then_fdia"


def inject_hybrid(measurements: np.ndarray, config: HybridConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return composed attacked values, combined attack mask, and net vector."""
    if config.order != "replay_then_fdia":
        raise ValueError("only replay_then_fdia is currently supported")
    replayed, replay_mask, _ = inject_replay(measurements, config.replay)
    attacked, fdia_mask, _ = inject_fdia(replayed, config.fdia)
    return attacked, replay_mask | fdia_mask, attacked - np.asarray(measurements, dtype=float)
