"""Compare baseline and fourth-order fixed/adaptive-KL SP-PF prototypes.

The existing 2nd-order demo remains unchanged.  Fourth-order channels 12 and
13 are the electrical-power measurements of generators 1 and 2; the baseline
uses channels 6 and 7 for the same physical measurement category.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from attacks.fdia import FDIAConfig, inject_fdia
from detection.detector import ResidualDetector
from experiments.run_fdia_sppf import DemoConfig, run_experiment
from power_system.network import FourthOrderThreeGeneratorSystem, make_power_system
from sppf.dynamics import StatePartitionParticleFilter
from sppf.partition import AdaptiveKLConfig, AdaptiveKLPartitioner, StatePartitioner


@dataclass(frozen=True)
class AdaptiveExperimentConfig:
    seed: int = 20260907
    steps: int = 180
    dt: float = 0.02
    particle_count: int = 350
    calibration_start_index: int = 30
    attack_start_index: int = 100
    attack_duration_steps: int = 55
    attack_magnitudes: tuple[float, ...] = (0.0, 0.10, 0.15, 0.20, 0.30)
    fourth_order_attack_channels: tuple[int, ...] = (12, 13)
    second_order_attack_channels: tuple[int, ...] = (6, 7)
    adaptive_kl: AdaptiveKLConfig = AdaptiveKLConfig()
    output_directory: str = "results/adaptive_kl"

    @property
    def attack_end_index(self) -> int:
        return self.attack_start_index + self.attack_duration_steps


def _covariance(std: np.ndarray) -> np.ndarray:
    return np.diag(np.asarray(std, dtype=float) ** 2)


def _metrics(scores: np.ndarray, flags: np.ndarray, attack_mask: np.ndarray, attack_start: int, dt: float) -> dict[str, object]:
    candidates = np.flatnonzero(flags & attack_mask)
    detected = candidates.size > 0
    detection_index = int(candidates[0]) if detected else None
    delay = None if detection_index is None else detection_index - attack_start
    normal = np.arange(flags.size) < attack_start
    return {
        "detected": bool(detected), "detection_index": detection_index,
        "detection_delay_steps": delay, "detection_delay_seconds": None if delay is None else float(delay * dt),
        "false_alarm_rate": float(np.count_nonzero(flags & normal) / np.count_nonzero(normal)),
        "max_score": float(np.max(scores)),
    }


def run_fourth_order_case(config: AdaptiveExperimentConfig, magnitude: float, partition_mode: str) -> dict[str, Any]:
    """One finite fourth-order FDIA case; calibration uses pre-attack samples only."""
    if partition_mode not in {"fixed", "adaptive_kl"}:
        raise ValueError("partition_mode must be fixed or adaptive_kl")
    if config.attack_end_index > config.steps:
        raise ValueError("attack interval exceeds simulation length")
    start_time = perf_counter()
    rng = np.random.default_rng(config.seed)
    system = make_power_system(model_order=4, dt=config.dt)
    assert isinstance(system, FourthOrderThreeGeneratorSystem)
    process_std = np.array([0.00010, 0.00045, 0.00025, 0.00025] * 3)
    filter_process_std = np.array([0.00008, 0.00035, 0.00020, 0.00020] * 3)
    initial_std = np.array([0.025, 0.018, 0.015, 0.015] * 3)
    measurement_std = np.array([0.0035] * 3 + [0.006] * 3 + [0.006] * 3 + [0.006] * 3 + [0.010] * 3)
    measurement_covariance = _covariance(measurement_std)
    truth = system.simulate(config.steps, None, process_std, rng)
    true_measurements = system.measurement(truth)
    clean = true_measurements + rng.multivariate_normal(np.zeros(system.measurement_dimension), measurement_covariance, size=config.steps)
    attack = FDIAConfig(config.attack_start_index, config.attack_end_index, magnitude, config.fourth_order_attack_channels)
    attacked, attack_mask, attack_vector = inject_fdia(clean, attack)
    initial = system.equilibrium_state + rng.multivariate_normal(np.zeros(system.state_dimension), _covariance(initial_std))
    controller = AdaptiveKLPartitioner(config.adaptive_kl) if partition_mode == "adaptive_kl" else None
    sppf = StatePartitionParticleFilter(
        system=system, partitioner=StatePartitioner.fourth_order_default(), particle_count=config.particle_count,
        initial_state=initial, initial_covariance=_covariance(initial_std), process_noise_covariance=_covariance(filter_process_std),
        measurement_noise_covariance=measurement_covariance, rng=rng, partition_mode=partition_mode, adaptive_partitioner=controller,
    )
    estimates = np.empty_like(truth)
    prior = np.empty_like(clean)
    effective_sizes: list[np.ndarray] = []
    partition_counts = np.empty(config.steps, dtype=int)
    estimates[0] = initial; prior[0] = system.measurement(initial); partition_counts[0] = 3
    for index in range(1, config.steps):
        estimates[index], prior[index], effective = sppf.step(attacked[index], (index - 1) * config.dt, index=index if partition_mode == "adaptive_kl" else None)
        effective_sizes.append(effective)
        partition_counts[index] = len(sppf.partitioner.partitions)
    detector = ResidualDetector(measurement_covariance)
    scores = np.array([detector.score(attacked[index], prior[index]) for index in range(config.steps)])
    threshold = detector.calibrate(scores[config.calibration_start_index:config.attack_start_index])
    flags = detector.flags(scores); flags[:config.calibration_start_index] = False
    if not all(np.all(np.isfinite(value)) for value in (truth, attacked, estimates, prior, scores)) or not all(np.all(np.isfinite(value)) for value in effective_sizes):
        raise FloatingPointError("fourth-order case produced NaN/Inf")
    for filter_ in sppf.filters:
        if not np.isclose(filter_.weights.sum(), 1.0) or not (1.0 <= filter_.effective_sample_size() <= config.particle_count):
            raise FloatingPointError("invalid final fourth-order particle filter")
    data = {
        "model_order": 4, "partition_mode": partition_mode, "attack_magnitude": magnitude,
        "attack_channels": list(config.fourth_order_attack_channels), "truth": truth, "estimates": estimates,
        "true_measurements": true_measurements, "attacked_measurements": attacked, "attack_mask": attack_mask,
        "attack_vector": attack_vector, "scores": scores, "flags": flags, "threshold": float(threshold),
        "state_rmse": float(np.sqrt(np.mean((truth - estimates) ** 2))), "partition_counts": partition_counts,
        "events": list(sppf.partition_history), "diagnostics": list(sppf.kl_diagnostics),
        "merge_count": sum(event["operation"] == "merge" for event in sppf.partition_history),
        "split_count": sum(event["operation"] == "split" for event in sppf.partition_history),
        "runtime_seconds": float(perf_counter() - start_time),
        "particle_count": config.particle_count,
    }
    data.update(_metrics(scores, flags, attack_mask, config.attack_start_index, config.dt))
    return data


def run_second_order_case(config: AdaptiveExperimentConfig, magnitude: float) -> dict[str, Any]:
    """Run unchanged baseline code with matching timing; only its size differs."""
    start_time = perf_counter()
    attack = FDIAConfig(config.attack_start_index, config.attack_end_index, magnitude, config.second_order_attack_channels)
    baseline_config = DemoConfig(seed=config.seed, steps=config.steps, dt=config.dt, particle_count=config.particle_count,
                                 calibration_start_index=config.calibration_start_index, attack=attack)
    result = run_experiment(baseline_config, output_dir=None, make_plots=False)
    data = {
        "model_order": 2, "partition_mode": "fixed", "attack_magnitude": magnitude,
        "attack_channels": list(config.second_order_attack_channels), "truth": result["truth"], "estimates": result["estimates"],
        "true_measurements": result["true_measurements"], "attacked_measurements": result["attacked_measurements"],
        "attack_mask": result["attack_mask"], "scores": result["scores"], "flags": result["flags"],
        "threshold": float(result["threshold"]), "state_rmse": float(result["state_rmse"]),
        "partition_counts": np.full(config.steps, 3), "events": [], "diagnostics": [], "merge_count": 0, "split_count": 0,
        "runtime_seconds": float(perf_counter() - start_time),
        "particle_count": config.particle_count,
    }
    data.update(_metrics(np.asarray(result["scores"]), np.asarray(result["flags"]), np.asarray(result["attack_mask"]), config.attack_start_index, config.dt))
    return data


def _row(data: dict[str, Any]) -> dict[str, object]:
    return {
        "model_order": data["model_order"], "partition_mode": data["partition_mode"], "attack_magnitude": data["attack_magnitude"],
        "attack_channels": ",".join(map(str, data["attack_channels"])), "detected": data["detected"],
        "detection_delay": data["detection_delay_steps"], "detection_delay_seconds": data["detection_delay_seconds"],
        "false_alarm_rate": data["false_alarm_rate"], "state_rmse": data["state_rmse"], "max_score": data["max_score"],
        "num_partitions_initial": 3, "num_partitions_final": int(data["partition_counts"][-1]),
        "merge_count": data["merge_count"], "split_count": data["split_count"],
        "average_particles_per_partition": float(data["particle_count"] / np.mean(data["partition_counts"])),
        "runtime": data["runtime_seconds"],
    }


def _plot(all_data: list[dict[str, Any]], config: AdaptiveExperimentConfig, output: Path) -> None:
    cache = output / ".matplotlib"; cache.mkdir(parents=True, exist_ok=True); os.environ["MPLCONFIGDIR"] = str(cache.resolve())
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output.mkdir(parents=True, exist_ok=True)
    fourth_fixed = [data for data in all_data if data["model_order"] == 4 and data["partition_mode"] == "fixed"]
    fourth_adaptive = [data for data in all_data if data["model_order"] == 4 and data["partition_mode"] == "adaptive_kl"]
    magnitudes = [data["attack_magnitude"] * 100 for data in fourth_fixed]
    for filename, key, ylabel, title in [
        ("fixed_vs_adaptive_detection_delay.png", "detection_delay_steps", "Detection delay (samples)", "Fourth-order fixed vs adaptive-KL delay"),
        ("fixed_vs_adaptive_state_rmse.png", "state_rmse", "State RMSE", "Fourth-order fixed vs adaptive-KL RMSE"),
    ]:
        fig, axis = plt.subplots(figsize=(8, 4.5))
        axis.plot(magnitudes, [np.nan if data[key] is None else data[key] for data in fourth_fixed], marker="o", label="fourth-order fixed")
        axis.plot(magnitudes, [np.nan if data[key] is None else data[key] for data in fourth_adaptive], marker="s", label="fourth-order adaptive KL")
        axis.set(xlabel="FDIA magnitude (%)", ylabel=ylabel, title=title); axis.legend(); axis.grid(alpha=0.25); fig.tight_layout(); fig.savefig(output / filename, dpi=160); plt.close(fig)
    representative = next(data for data in fourth_adaptive if data["attack_magnitude"] == 0.20)
    time = np.arange(config.steps) * config.dt
    fig, axis = plt.subplots(figsize=(9, 4))
    axis.step(time, representative["partition_counts"], where="post")
    axis.set(xlabel="Time (s)", ylabel="Number of partitions", ylim=(0.5, 3.5), title="Adaptive-KL partition count (20% FDIA)"); axis.grid(alpha=0.25); fig.tight_layout(); fig.savefig(output / "partitions_vs_time.png", dpi=160); plt.close(fig)
    fig, axis = plt.subplots(figsize=(9, 4))
    diagnostics = representative["diagnostics"]
    if diagnostics:
        indices = [entry["index"] for entry in diagnostics]
        for pair in ("1-2", "1-3", "2-3"):
            axis.plot(indices, [entry["pairwise_symmetric_kl"][pair] for entry in diagnostics], marker="o", label=f"generators {pair}")
        axis.axhline(config.adaptive_kl.merge_threshold, color="tab:green", linestyle="--", label="merge threshold")
        axis.axhline(config.adaptive_kl.split_threshold, color="tab:red", linestyle="--", label="split threshold")
    # Symmetric KL spans several orders of magnitude in this prototype. A
    # symmetric log scale keeps both measured values and hysteresis thresholds
    # visible without changing any decision calculation.
    axis.set_yscale("symlog", linthresh=config.adaptive_kl.merge_threshold)
    axis.set(xlabel="Sample index", ylabel="Symmetric KL (symlog)", title="Whitened innovation Gaussian KL (20% FDIA)"); axis.legend(fontsize=8); axis.grid(alpha=0.25); fig.tight_layout(); fig.savefig(output / "kl_divergence_thresholds.png", dpi=160); plt.close(fig)
    fig, axis = plt.subplots(figsize=(9, 4))
    axis.plot(time, representative["scores"], label="innovation J-score")
    axis.axhline(representative["threshold"], color="tab:red", linestyle="--", label="threshold")
    axis.axvspan(time[config.attack_start_index], time[config.attack_end_index - 1], color="tab:red", alpha=0.12, label="FDIA interval")
    axis.set(xlabel="Time (s)", ylabel="J-score", title="Adaptive fourth-order detection score (20% FDIA)"); axis.legend(); axis.grid(alpha=0.25); fig.tight_layout(); fig.savefig(output / "adaptive_detection_score.png", dpi=160); plt.close(fig)
    fig, axes = plt.subplots(3, 4, figsize=(13, 8), sharex=True)
    for index, axis in enumerate(axes.flat):
        axis.plot(time, representative["truth"][:, index], linewidth=1.2, label="reference")
        axis.plot(time, representative["estimates"][:, index], linewidth=1.0, label="adaptive SP-PF")
        axis.axvspan(time[config.attack_start_index], time[config.attack_end_index - 1], color="tab:red", alpha=0.08)
        axis.set_ylabel(FourthOrderThreeGeneratorSystem.state_labels[index]); axis.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=7); axes[-1, 0].set_xlabel("Time (s)"); axes[-1, 1].set_xlabel("Time (s)"); axes[-1, 2].set_xlabel("Time (s)"); axes[-1, 3].set_xlabel("Time (s)")
    fig.suptitle("Fourth-order adaptive-KL SP-PF state estimates (20% FDIA)"); fig.tight_layout(); fig.savefig(output / "fourth_order_state_estimation.png", dpi=160); plt.close(fig)


def run_comparison(config: AdaptiveExperimentConfig = AdaptiveExperimentConfig()) -> dict[str, Any]:
    """Run the requested compact comparison and save all reproducible artefacts."""
    if config.attack_end_index > config.steps:
        raise ValueError("attack end exceeds total steps")
    outputs: list[dict[str, Any]] = []
    for magnitude in config.attack_magnitudes:
        outputs.append(run_second_order_case(config, magnitude))
        outputs.append(run_fourth_order_case(config, magnitude, "fixed"))
        outputs.append(run_fourth_order_case(config, magnitude, "adaptive_kl"))
    rows = [_row(data) for data in outputs]
    directory = Path(config.output_directory); directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(directory / "comparison.csv", index=False)
    payload = {"configuration": asdict(config), "comparisons": rows}
    (directory / "comparison.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    representative = next(data for data in outputs if data["model_order"] == 4 and data["partition_mode"] == "adaptive_kl" and data["attack_magnitude"] == 0.20)
    history = {"events": representative["events"], "diagnostics": representative["diagnostics"], "description": "Events and periodic KL diagnostics for fourth-order adaptive-KL 20% FDIA."}
    (directory / "partition_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    _plot(outputs, config, directory)
    return {"rows": rows, "data": outputs, "history": history}


def print_report(result: dict[str, Any]) -> None:
    frame = pd.DataFrame(result["rows"])
    print("=" * 56); print("FOURTH-ORDER FIXED VS ADAPTIVE-KL SP-PF") ; print("=" * 56)
    print(f"Scenarios: {len(frame)}; finite completed rows: {len(frame)}")
    print(frame[["model_order", "partition_mode", "attack_magnitude", "detected", "detection_delay", "state_rmse", "merge_count", "split_count", "runtime"]].to_string(index=False))


if __name__ == "__main__":
    print_report(run_comparison())
