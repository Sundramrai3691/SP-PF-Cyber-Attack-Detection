"""End-to-end fixed SP-PF FDIA experiment.

Run from the repository root with ``python run_demo.py`` or directly with
``python -m experiments.run_fdia_sppf``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import os
import numpy as np

from attacks.fdia import FDIAConfig, inject_fdia
from detection.detector import ResidualDetector
from power_system.network import ThreeGeneratorSystem
from sppf.dynamics import StatePartitionParticleFilter
from sppf.partition import StatePartitioner


@dataclass(frozen=True)
class DemoConfig:
    seed: int = 20260907
    steps: int = 240
    dt: float = 0.02
    particle_count: int = 600
    process_noise_std: tuple[float, ...] = (0.00010, 0.00045, 0.00010, 0.00045, 0.00010, 0.00045)
    filter_process_noise_std: tuple[float, ...] = (0.00008, 0.00035, 0.00008, 0.00035, 0.00008, 0.00035)
    measurement_noise_std: tuple[float, ...] = (0.0035, 0.0035, 0.0035, 0.006, 0.006, 0.006, 0.010, 0.010, 0.010)
    initial_std: tuple[float, ...] = (0.025, 0.018, 0.025, 0.018, 0.025, 0.018)
    threshold_multiplier: float = 4.0
    calibration_start_index: int = 30
    attack: FDIAConfig = field(default_factory=FDIAConfig)


def _covariance(std: tuple[float, ...] | np.ndarray) -> np.ndarray:
    return np.diag(np.asarray(std, dtype=float) ** 2)


def run_experiment(config: DemoConfig = DemoConfig(), output_dir: str | Path | None = "results", make_plots: bool = True) -> dict[str, object]:
    """Run simulation, FDIA, SP-PF estimation, calibration and detection."""
    if not (5 <= config.calibration_start_index < config.attack.start_index):
        raise ValueError("calibration_start_index must leave an attack-free calibration interval")
    rng = np.random.default_rng(config.seed)
    system = ThreeGeneratorSystem(config.dt)
    partitioner = StatePartitioner.three_generator_default()
    measurement_covariance = _covariance(config.measurement_noise_std)
    initial_covariance = _covariance(config.initial_std)
    filter_q = _covariance(config.filter_process_noise_std)

    true_initial = np.array([0.125, 0.0, -0.065, 0.0, 0.045, 0.0])
    truth = system.simulate(config.steps, true_initial, np.asarray(config.process_noise_std), rng)
    true_measurements = system.measurement(truth)
    clean_measurements = true_measurements + rng.multivariate_normal(np.zeros(system.measurement_dimension), measurement_covariance, size=config.steps)
    attacked_measurements, attack_mask, attack_vector = inject_fdia(clean_measurements, config.attack)

    initial_estimate = true_initial + rng.multivariate_normal(np.zeros(system.state_dimension), initial_covariance)
    sppf = StatePartitionParticleFilter(
        system, partitioner, config.particle_count, initial_estimate, initial_covariance,
        filter_q, measurement_covariance, rng,
    )
    estimates = np.empty_like(truth)
    prior_measurements = np.empty_like(clean_measurements)
    effective_sizes = np.empty((config.steps, len(partitioner.partitions)))
    estimates[0] = initial_estimate
    prior_measurements[0] = system.measurement(initial_estimate)
    effective_sizes[0] = config.particle_count
    for k in range(1, config.steps):
        estimates[k], prior_measurements[k], effective_sizes[k] = sppf.step(attacked_measurements[k], (k - 1) * config.dt)

    detector = ResidualDetector(measurement_covariance, config.threshold_multiplier)
    scores = np.array([detector.score(attacked_measurements[k], prior_measurements[k]) for k in range(config.steps)])
    # Exclude PF start-up transient from attack-free threshold calibration.
    threshold = detector.calibrate(scores[config.calibration_start_index:config.attack.start_index])
    flags = detector.flags(scores)
    flags[:config.calibration_start_index] = False

    # A deliberately simple, separate baseline: open-loop nonlinear dynamics
    # from the same initial estimate, calibrated over the same normal window.
    baseline_states = np.empty_like(truth)
    baseline_predictions = np.empty_like(clean_measurements)
    baseline_states[0] = initial_estimate
    baseline_predictions[0] = system.measurement(initial_estimate)
    for k in range(1, config.steps):
        baseline_states[k] = system.transition(baseline_states[k - 1], (k - 1) * config.dt)
        baseline_predictions[k] = system.measurement(baseline_states[k])
    baseline_detector = ResidualDetector(measurement_covariance, config.threshold_multiplier)
    baseline_scores = np.array([baseline_detector.score(attacked_measurements[k], baseline_predictions[k]) for k in range(config.steps)])
    baseline_threshold = baseline_detector.calibrate(baseline_scores[config.calibration_start_index:config.attack.start_index])
    baseline_flags = baseline_detector.flags(baseline_scores)
    baseline_flags[:config.calibration_start_index] = False

    after_attack = np.flatnonzero(flags & (np.arange(config.steps) >= config.attack.start_index))
    detection_index = int(after_attack[0]) if after_attack.size else None
    baseline_after_attack = np.flatnonzero(baseline_flags & (np.arange(config.steps) >= config.attack.start_index))
    baseline_detection_index = int(baseline_after_attack[0]) if baseline_after_attack.size else None
    rmse = float(np.sqrt(np.mean((truth - estimates) ** 2)))
    _assert_valid(truth, true_measurements, clean_measurements, attacked_measurements, estimates, scores, effective_sizes)
    result: dict[str, object] = {
        "config": config,
        "truth": truth,
        "true_measurements": true_measurements,
        "clean_measurements": clean_measurements,
        "attacked_measurements": attacked_measurements,
        "attack_vector": attack_vector,
        "attack_mask": attack_mask,
        "estimates": estimates,
        "prior_measurements": prior_measurements,
        "scores": scores,
        "threshold": threshold,
        "flags": flags,
        "effective_sizes": effective_sizes,
        "baseline_scores": baseline_scores,
        "baseline_threshold": baseline_threshold,
        "baseline_flags": baseline_flags,
        "detection_index": detection_index,
        "detection_delay_steps": None if detection_index is None else detection_index - config.attack.start_index,
        "baseline_detection_index": baseline_detection_index,
        "baseline_detection_delay_steps": None if baseline_detection_index is None else baseline_detection_index - config.attack.start_index,
        "state_rmse": rmse,
    }
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        _write_summary(result, destination)
        if make_plots:
            _plot(result, destination)
    return result


def _assert_valid(*arrays: np.ndarray) -> None:
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise FloatingPointError("experiment produced NaN or Inf")


def _write_summary(result: dict[str, object], output_dir: Path) -> None:
    config = result["config"]
    assert isinstance(config, DemoConfig)
    summary = {
        "particle_count_per_partition": config.particle_count,
        "number_of_partitions": 3,
        "attack_start_index": config.attack.start_index,
        "attack_end_index_exclusive": config.attack.end_index,
        "attacked_measurement_indices": config.attack.measurement_indices,
        "attack_magnitude": config.attack.magnitude,
        "detection_index": result["detection_index"],
        "detection_delay_steps": result["detection_delay_steps"],
        "detection_delay_seconds": None if result["detection_delay_steps"] is None else result["detection_delay_steps"] * config.dt,
        "threshold": result["threshold"],
        "maximum_detection_score": float(np.max(result["scores"])),
        "state_rmse": result["state_rmse"],
        "baseline_detection_index": result["baseline_detection_index"],
        "baseline_threshold": result["baseline_threshold"],
        "seed": config.seed,
        "calibration_start_index": config.calibration_start_index,
    }
    (output_dir / "fdia_sppf_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _plot(result: dict[str, object], output_dir: Path) -> None:
    # The workspace may not permit Matplotlib's default user-profile cache.
    # Keep its cache local and writable before importing Matplotlib.
    cache_dir = output_dir / ".matplotlib"
    cache_dir.mkdir(exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_dir.resolve())
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config = result["config"]
    assert isinstance(config, DemoConfig)
    time = np.arange(config.steps) * config.dt
    start, end = config.attack.start_index, config.attack.end_index
    attacked_index = config.attack.measurement_indices[0]
    labels = ThreeGeneratorSystem.measurement_labels

    fig, axis = plt.subplots(figsize=(10, 4))
    axis.plot(time, result["true_measurements"][:, attacked_index], label="true measurement", linewidth=1.5)
    axis.plot(time, result["attacked_measurements"][:, attacked_index], label="FDIA measurement", linewidth=1.1)
    axis.axvspan(time[start], time[end - 1], color="tab:red", alpha=0.12, label="attack interval")
    axis.set(xlabel="Time (s)", ylabel=labels[attacked_index], title="True/noisy vs FDIA measurement")
    axis.legend(); axis.grid(alpha=0.25); fig.tight_layout(); fig.savefig(output_dir / "01_measurement_fdia.png", dpi=160); plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(11, 8), sharex=True)
    for state_index, axis in enumerate(axes.flat):
        axis.plot(time, result["truth"][:, state_index], label="true", linewidth=1.6)
        axis.plot(time, result["estimates"][:, state_index], label="SP-PF", linewidth=1.1)
        axis.axvspan(time[start], time[end - 1], color="tab:red", alpha=0.10)
        axis.set_ylabel(ThreeGeneratorSystem.state_labels[state_index]); axis.grid(alpha=0.25)
    axes[0, 0].legend(); axes[-1, 0].set_xlabel("Time (s)"); axes[-1, 1].set_xlabel("Time (s)")
    fig.suptitle("True state vs fixed-partition particle-filter estimate"); fig.tight_layout(); fig.savefig(output_dir / "02_state_estimation.png", dpi=160); plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 4))
    axis.plot(time, result["scores"], label="SP-PF innovation J-score")
    axis.axhline(float(result["threshold"]), color="tab:red", linestyle="--", label="SP-PF threshold")
    axis.plot(time, result["baseline_scores"], color="0.45", alpha=0.65, label="open-loop baseline")
    axis.axhline(float(result["baseline_threshold"]), color="0.2", linestyle=":", label="baseline threshold")
    axis.axvspan(time[start], time[end - 1], color="tab:red", alpha=0.10)
    axis.set(xlabel="Time (s)", ylabel="J", title="Residual detection statistic"); axis.legend(ncol=2); axis.grid(alpha=0.25); fig.tight_layout(); fig.savefig(output_dir / "03_detection_score.png", dpi=160); plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 3.5))
    axis.step(time, result["attack_mask"].astype(int), where="post", label="injected attack", linewidth=1.6)
    axis.step(time, result["flags"].astype(int), where="post", label="SP-PF alarm", linewidth=1.2)
    axis.step(time, result["baseline_flags"].astype(int), where="post", label="baseline alarm", linewidth=1.0, alpha=0.75)
    axis.set(xlabel="Time (s)", ylabel="Binary flag", ylim=(-0.08, 1.15), title="Attack and detection flags")
    axis.legend(ncol=3); axis.grid(alpha=0.25); fig.tight_layout(); fig.savefig(output_dir / "04_attack_flags.png", dpi=160); plt.close(fig)


def print_report(result: dict[str, object]) -> None:
    config = result["config"]
    assert isinstance(config, DemoConfig)
    print("=" * 48)
    print("SP-PF FDIA DETECTION RESULTS")
    print("=" * 48)
    print(f"Particles per partition: {config.particle_count}")
    print("Number of partitions: 3")
    print(f"Attack start: index {config.attack.start_index} ({config.attack.start_index * config.dt:.3f} s)")
    print(f"Attacked measurement indices: {config.attack.measurement_indices}; magnitude: {config.attack.magnitude:.3f}")
    print(f"Detection time: {result['detection_index']}")
    delay_seconds = None if result["detection_delay_steps"] is None else result["detection_delay_steps"] * config.dt
    delay_text = "None" if delay_seconds is None else f"{delay_seconds:.3f} s"
    print(f"Detection delay: {result['detection_delay_steps']} steps ({delay_text})")
    print(f"Threshold: {float(result['threshold']):.3f}")
    print(f"Maximum detection score: {float(np.max(result['scores'])):.3f}")
    print(f"State RMSE: {float(result['state_rmse']):.6f}")
    print(f"Baseline detection time: {result['baseline_detection_index']}")


if __name__ == "__main__":
    print_report(run_experiment())
