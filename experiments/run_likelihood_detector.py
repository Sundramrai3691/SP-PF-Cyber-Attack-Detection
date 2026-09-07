"""Compare residual and particle-likelihood SP-PF detectors on the 4th-order model.

Run with ``python -m experiments.run_likelihood_detector``.  This experiment
does not alter the original residual-demo scripts; it is an explicitly
selected detector comparison using identical fourth-order simulation settings.
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
from detection.detector import (
    LikelihoodRatioDetector,
    ResidualDetector,
    compute_global_log_likelihood,
    compute_particle_log_likelihood,
    compute_weighted_particle_log_likelihood,
)
from experiments.run_adaptive_kl_experiment import AdaptiveExperimentConfig, _covariance
from power_system.network import FourthOrderThreeGeneratorSystem, make_power_system
from sppf.dynamics import StatePartitionParticleFilter
from sppf.partition import AdaptiveKLConfig, AdaptiveKLPartitioner, StatePartitioner


@dataclass(frozen=True)
class LikelihoodExperimentConfig:
    """Compact, reproducible matrix; matches the fourth-order comparison setup."""

    seed: int = 20260907
    steps: int = 180
    dt: float = 0.02
    particle_count: int = 350
    calibration_start_index: int = 30
    attack_start_index: int = 100
    attack_duration_steps: int = 55
    attack_magnitudes: tuple[float, ...] = (0.0, 0.10, 0.15, 0.20, 0.30)
    attack_channels: tuple[int, ...] = (12, 13)
    threshold_multiplier: float = 4.0
    adaptive_kl: AdaptiveKLConfig = AdaptiveKLConfig()
    output_directory: str = "results/likelihood_detector"

    @property
    def attack_end_index(self) -> int:
        return self.attack_start_index + self.attack_duration_steps


def compute_sppf_global_log_likelihood(likelihood_inputs: list[dict[str, object]]) -> tuple[float, np.ndarray]:
    """Evaluate the SP-PF predictive likelihood from captured particle clouds.

    For partition ``p`` this computes
    ``ell_p = log(sum_i w_prior,p,i * p(y_p | x_p,i))``.  The SP-PF currently
    factorises its local measurement updates conditionally on the global
    context, so the prototype global likelihood is ``ell = sum_p ell_p``.
    Summing logs avoids a product of tiny Gaussian densities.
    """
    values: list[float] = []
    for item in likelihood_inputs:
        particle_logs = compute_particle_log_likelihood(
            np.asarray(item["measurement"]),
            np.asarray(item["predicted_measurements"]),
            np.asarray(item["measurement_covariance"]),
        )
        values.append(compute_weighted_particle_log_likelihood(particle_logs, np.asarray(item["weights"])))
    partition_values = np.asarray(values, dtype=float)
    return compute_global_log_likelihood(partition_values), partition_values


def _detector_outputs(
    detector_type: str,
    measurements: np.ndarray,
    prior_measurements: np.ndarray,
    measurement_covariance: np.ndarray,
    log_likelihoods: np.ndarray,
    calibration: slice,
    threshold_multiplier: float,
) -> tuple[np.ndarray, np.ndarray, float, dict[str, float]]:
    """Return scores/flags/threshold without conflating residual and likelihood units."""
    if detector_type == "residual":
        detector = ResidualDetector(measurement_covariance, threshold_multiplier)
        scores = np.asarray([detector.score(y, predicted) for y, predicted in zip(measurements, prior_measurements)])
        threshold = detector.calibrate(scores[calibration])
        flags = detector.flags(scores)
        details: dict[str, float] = {}
    elif detector_type == "likelihood":
        detector = LikelihoodRatioDetector(threshold_multiplier)
        threshold = detector.calibrate(log_likelihoods[calibration])
        scores = detector.scores(log_likelihoods)
        flags = detector.flags(scores)
        details = {
            "reference_log_likelihood": float(detector.reference_log_likelihood),
            "reference_log_likelihood_std": float(detector.reference_std),
        }
    else:
        raise ValueError("detector_type must be 'residual' or 'likelihood'")
    if not (np.all(np.isfinite(scores)) and np.isfinite(threshold)):
        raise FloatingPointError("detector outputs are non-finite")
    return scores, flags, float(threshold), details


def run_fourth_order_detector_case(
    config: LikelihoodExperimentConfig,
    attack_magnitude: float,
    partition_mode: str,
    detector_type: str,
) -> dict[str, Any]:
    """Run one detector/mode pair with exactly one attack-free calibration window."""
    if partition_mode not in {"fixed", "adaptive_kl"}:
        raise ValueError("partition_mode must be fixed or adaptive_kl")
    if not (5 <= config.calibration_start_index < config.attack_start_index < config.attack_end_index <= config.steps):
        raise ValueError("invalid calibration or attack interval")
    started = perf_counter()
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
    clean = true_measurements + rng.multivariate_normal(
        np.zeros(system.measurement_dimension), measurement_covariance, size=config.steps
    )
    attack = FDIAConfig(config.attack_start_index, config.attack_end_index, attack_magnitude, config.attack_channels)
    attacked, attack_mask, attack_vector = inject_fdia(clean, attack)
    initial = system.equilibrium_state + rng.multivariate_normal(np.zeros(system.state_dimension), _covariance(initial_std))
    adaptive = AdaptiveKLPartitioner(config.adaptive_kl) if partition_mode == "adaptive_kl" else None
    sppf = StatePartitionParticleFilter(
        system=system,
        partitioner=StatePartitioner.fourth_order_default(),
        particle_count=config.particle_count,
        initial_state=initial,
        initial_covariance=_covariance(initial_std),
        process_noise_covariance=_covariance(filter_process_std),
        measurement_noise_covariance=measurement_covariance,
        rng=rng,
        partition_mode=partition_mode,
        adaptive_partitioner=adaptive,
    )
    estimates = np.empty_like(truth)
    prior = np.empty_like(clean)
    log_likelihoods = np.empty(config.steps, dtype=float)
    partition_log_likelihoods: list[np.ndarray] = []
    partition_counts = np.empty(config.steps, dtype=int)
    effective_sizes: list[np.ndarray] = []

    estimates[0] = initial
    prior[0] = system.measurement(initial)
    initial_inputs = sppf.capture_likelihood_inputs(attacked[0])
    log_likelihoods[0], first_partition_logs = compute_sppf_global_log_likelihood(initial_inputs)
    partition_log_likelihoods.append(first_partition_logs)
    partition_counts[0] = len(sppf.partitioner.partitions)
    for index in range(1, config.steps):
        estimates[index], prior[index], effective = sppf.step(
            attacked[index], (index - 1) * config.dt, index=index if partition_mode == "adaptive_kl" else None
        )
        log_likelihoods[index], local_logs = compute_sppf_global_log_likelihood(sppf.last_likelihood_inputs)
        partition_log_likelihoods.append(local_logs)
        partition_counts[index] = len(sppf.partitioner.partitions)
        effective_sizes.append(effective)

    calibration = slice(config.calibration_start_index, config.attack_start_index)
    scores, flags, threshold, detector_details = _detector_outputs(
        detector_type, attacked, prior, measurement_covariance, log_likelihoods, calibration, config.threshold_multiplier
    )
    # The PF initialisation transient is not an alarm decision period.
    flags[:config.calibration_start_index] = False
    candidates = np.flatnonzero(flags & attack_mask)
    detection_index = int(candidates[0]) if candidates.size else None
    delay = None if detection_index is None else detection_index - config.attack_start_index
    normal_indices = np.arange(config.steps) < config.attack_start_index
    delta_indices = np.array([0, 4, 8])
    omega_indices = np.array([1, 5, 9])

    finite_arrays = (truth, attacked, estimates, prior, log_likelihoods, scores)
    if not all(np.all(np.isfinite(value)) for value in finite_arrays):
        raise FloatingPointError("likelihood detector case produced NaN/Inf")
    for effective in effective_sizes:
        if not np.all(np.isfinite(effective)) or np.any(effective < 1.0) or np.any(effective > config.particle_count):
            raise FloatingPointError("invalid effective sample size")
    for filter_ in sppf.filters:
        if not np.all(np.isfinite(filter_.particles)) or not np.all(np.isfinite(filter_.weights)) or not np.isclose(filter_.weights.sum(), 1.0):
            raise FloatingPointError("invalid final particle filter state")

    data: dict[str, Any] = {
        "model_order": 4,
        "partition_mode": partition_mode,
        "detector_type": detector_type,
        "attack_magnitude": float(attack_magnitude),
        "attack_channels": list(config.attack_channels),
        "attack_start": config.attack_start_index,
        "attack_end": config.attack_end_index,
        "truth": truth,
        "estimates": estimates,
        "true_measurements": true_measurements,
        "attacked_measurements": attacked,
        "attack_mask": attack_mask,
        "attack_vector": attack_vector,
        "prior_measurements": prior,
        "log_likelihoods": log_likelihoods,
        "partition_log_likelihoods": partition_log_likelihoods,
        "scores": scores,
        "flags": flags,
        "threshold": threshold,
        "detected": bool(detection_index is not None),
        "detection_index": detection_index,
        "detection_delay": delay,
        "detection_delay_seconds": None if delay is None else float(delay * config.dt),
        "false_alarm_rate": float(np.count_nonzero(flags & normal_indices) / np.count_nonzero(normal_indices)),
        "max_score": float(np.max(scores)),
        "state_rmse": float(np.sqrt(np.mean((truth - estimates) ** 2))),
        "delta_mse": float(np.mean((truth[:, delta_indices] - estimates[:, delta_indices]) ** 2)),
        "omega_mse": float(np.mean((truth[:, omega_indices] - estimates[:, omega_indices]) ** 2)),
        "mean_log_likelihood": float(np.mean(log_likelihoods)),
        "min_log_likelihood": float(np.min(log_likelihoods)),
        "max_log_likelihood": float(np.max(log_likelihoods)),
        "mean_log_likelihood_ratio": float(np.mean(scores)) if detector_type == "likelihood" else None,
        "min_log_likelihood_ratio": float(np.min(scores)) if detector_type == "likelihood" else None,
        "max_log_likelihood_ratio": float(np.max(scores)) if detector_type == "likelihood" else None,
        "partition_counts": partition_counts,
        "merge_count": sum(event["operation"] == "merge" for event in sppf.partition_history),
        "split_count": sum(event["operation"] == "split" for event in sppf.partition_history),
        "runtime": float(perf_counter() - started),
        "particle_count": config.particle_count,
        **detector_details,
    }
    return data


def _row(data: dict[str, Any]) -> dict[str, object]:
    """Machine-readable row retaining likelihood diagnostics where applicable."""
    return {
        key: data.get(key)
        for key in (
            "model_order", "partition_mode", "detector_type", "attack_magnitude", "attack_channels",
            "attack_start", "attack_end", "detected", "detection_index", "detection_delay",
            "detection_delay_seconds", "false_alarm_rate", "threshold", "max_score", "state_rmse",
            "delta_mse", "omega_mse", "mean_log_likelihood", "min_log_likelihood", "max_log_likelihood",
            "mean_log_likelihood_ratio", "min_log_likelihood_ratio", "max_log_likelihood_ratio",
            "reference_log_likelihood", "reference_log_likelihood_std", "merge_count", "split_count", "runtime",
        )
    } | {"attack_channels": ",".join(map(str, data["attack_channels"]))}


def _select(data: list[dict[str, Any]], partition_mode: str, detector_type: str, magnitude: float) -> dict[str, Any]:
    return next(item for item in data if item["partition_mode"] == partition_mode and item["detector_type"] == detector_type and item["attack_magnitude"] == magnitude)


def _plot(data: list[dict[str, Any]], config: LikelihoodExperimentConfig, directory: Path) -> None:
    cache = directory / ".matplotlib"; cache.mkdir(parents=True, exist_ok=True); os.environ["MPLCONFIGDIR"] = str(cache.resolve())
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    representative_residual = _select(data, "fixed", "residual", 0.20)
    representative_likelihood = _select(data, "fixed", "likelihood", 0.20)
    adaptive_likelihood = _select(data, "adaptive_kl", "likelihood", 0.20)
    time = np.arange(config.steps) * config.dt
    attack_span = (time[config.attack_start_index], time[config.attack_end_index - 1])

    for filename, result, ylabel, title in (
        ("residual_score_vs_threshold.png", representative_residual, "J-statistic", "Fourth-order fixed residual detector (20% FDIA)"),
        ("likelihood_ratio_vs_threshold.png", representative_likelihood, "log(Lref / Lk)", "Fourth-order fixed likelihood detector (20% FDIA)"),
    ):
        fig, axis = plt.subplots(figsize=(9, 4))
        axis.plot(time, result["scores"], label=result["detector_type"])
        axis.axhline(result["threshold"], color="tab:red", linestyle="--", label="calibrated threshold")
        axis.axvspan(*attack_span, color="tab:red", alpha=0.12, label="FDIA interval")
        axis.set(xlabel="Time (s)", ylabel=ylabel, title=title); axis.legend(); axis.grid(alpha=0.25)
        fig.tight_layout(); fig.savefig(directory / filename, dpi=160); plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 4))
    axis.plot(time, representative_residual["scores"] / representative_residual["threshold"], label="residual score / threshold")
    axis.plot(time, representative_likelihood["scores"] / representative_likelihood["threshold"], label="likelihood score / threshold")
    axis.axhline(1.0, color="tab:red", linestyle="--", label="alarm boundary")
    axis.axvspan(*attack_span, color="tab:red", alpha=0.12)
    axis.set(xlabel="Time (s)", ylabel="Normalised detection statistic", title="Residual vs likelihood detector (same 20% FDIA)")
    axis.legend(); axis.grid(alpha=0.25); fig.tight_layout(); fig.savefig(directory / "residual_vs_likelihood.png", dpi=160); plt.close(fig)

    magnitudes = [value * 100.0 for value in config.attack_magnitudes]
    for filename, key, ylabel, title in (
        ("detection_delay_comparison.png", "detection_delay", "Delay (samples)", "Detection-delay comparison"),
        ("state_rmse_comparison.png", "state_rmse", "State RMSE", "State-estimation comparison"),
    ):
        fig, axis = plt.subplots(figsize=(8, 4.5))
        for mode in ("fixed", "adaptive_kl"):
            for detector in ("residual", "likelihood"):
                selected = [_select(data, mode, detector, magnitude) for magnitude in config.attack_magnitudes]
                values = [np.nan if result[key] is None else result[key] for result in selected]
                axis.plot(magnitudes, values, marker="o", label=f"{mode}: {detector}")
        axis.set(xlabel="FDIA magnitude (%)", ylabel=ylabel, title=title); axis.legend(fontsize=8); axis.grid(alpha=0.25)
        fig.tight_layout(); fig.savefig(directory / filename, dpi=160); plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 4))
    axis.plot(time, representative_likelihood["scores"], label="fixed likelihood")
    axis.plot(time, adaptive_likelihood["scores"], label="adaptive-KL likelihood", alpha=0.85)
    axis.axhline(representative_likelihood["threshold"], color="tab:blue", linestyle="--", alpha=0.6, label="fixed threshold")
    axis.axhline(adaptive_likelihood["threshold"], color="tab:orange", linestyle="--", alpha=0.7, label="adaptive threshold")
    axis.axvspan(*attack_span, color="tab:red", alpha=0.12)
    axis.set(xlabel="Time (s)", ylabel="log(Lref / Lk)", title="Fixed vs adaptive-KL likelihood detector (20% FDIA)")
    axis.legend(fontsize=8); axis.grid(alpha=0.25); fig.tight_layout(); fig.savefig(directory / "fixed_vs_adaptive_likelihood.png", dpi=160); plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 4))
    axis.plot(time, representative_likelihood["log_likelihoods"], label="global predictive log likelihood")
    axis.axhline(representative_likelihood["reference_log_likelihood"], color="tab:red", linestyle="--", label="normal reference mean")
    axis.axvspan(*attack_span, color="tab:red", alpha=0.12, label="FDIA interval")
    axis.set(xlabel="Time (s)", ylabel="log L", title="SP-PF global predictive likelihood (20% FDIA)")
    axis.legend(); axis.grid(alpha=0.25); fig.tight_layout(); fig.savefig(directory / "log_likelihood_over_time.png", dpi=160); plt.close(fig)


def _write_paper_comparison(directory: Path) -> None:
    text = """# Likelihood Detector: Paper Context vs This Prototype

## Paper direction

The reference work motivates likelihood or likelihood-ratio behaviour from
particle-filter state estimation. Its equations, system parameters, and attack
model are not numerically reproduced here.

## Implemented prototype

For each SP-PF partition and sample, the detector evaluates the full Gaussian
measurement log-density for every predicted particle:

`log p(y|x_i) = -0.5[(y-h_i)^T R^-1(y-h_i) + log|R| + m log(2π)]`.

The partition predictive likelihood is
`log L_p = log Σ_i w_prior,i exp(log p(y_p|x_i))`, evaluated with log-sum-exp.
Because the current SP-PF update is conditionally partitioned, the prototype
global likelihood is `log L = Σ_p log L_p`.

Attack evidence is a **normal-reference likelihood degradation**,
`g_k = log(L_ref / L_k) = μ0 - log L_k`, where `μ0` is the mean global
log-likelihood over attack-free calibration samples. The threshold is
`4σ0`, where `σ0` is the calibration standard deviation. This is a calibrated
one-sided likelihood-ratio-style test under H0, not the paper's exact H1
likelihood-ratio equation because this prototype has no parameterised attack
measurement distribution.

## Important limitations

- Thresholds are calibrated from this prototype's attack-free data and are
  not copied from the paper.
- The model is a fourth-order, three-generator synthetic prototype with
  approximate parameters and a simplified observation model.
- Adaptive-KL is an optional partitioning mechanism; it is not claimed to
  improve either RMSE or detection.
- Results are experimental comparisons, not an exact reproduction of paper
  figures or numerical thresholds.
"""
    (directory / "paper_comparison.md").write_text(text, encoding="utf-8")


def run_comparison(config: LikelihoodExperimentConfig = LikelihoodExperimentConfig()) -> dict[str, Any]:
    """Run 5 FDIA magnitudes × 2 modes × 2 detectors and persist results."""
    outputs: list[dict[str, Any]] = []
    for magnitude in config.attack_magnitudes:
        for partition_mode in ("fixed", "adaptive_kl"):
            for detector_type in ("residual", "likelihood"):
                outputs.append(run_fourth_order_detector_case(config, magnitude, partition_mode, detector_type))
    rows = [_row(item) for item in outputs]
    directory = Path(config.output_directory); directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(directory / "comparison.csv", index=False)
    (directory / "comparison.json").write_text(json.dumps({"configuration": asdict(config), "comparisons": rows}, indent=2), encoding="utf-8")
    _write_paper_comparison(directory)
    _plot(outputs, config, directory)
    return {"rows": rows, "data": outputs}


def print_report(result: dict[str, Any]) -> None:
    frame = pd.DataFrame(result["rows"])
    print("=" * 68)
    print("FOURTH-ORDER SP-PF: RESIDUAL VS LIKELIHOOD DETECTOR")
    print("=" * 68)
    print(f"Scenarios: {len(frame)}; completed rows: {len(frame)}")
    print(frame[["partition_mode", "detector_type", "attack_magnitude", "detected", "detection_delay", "false_alarm_rate", "state_rmse", "runtime"]].to_string(index=False))


if __name__ == "__main__":
    print_report(run_comparison())
