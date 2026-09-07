"""Shared paper-inspired test-case framework for the preserved 2nd-order model.

Terminology is deliberately conservative: active power is represented by the
model's synthetic electrical-power channels / mechanical-power input. The model
has no reactive-power measurement, so reactive paper cases are recorded as not
directly reproducible rather than simulated under a false label.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any
import json
import os

import numpy as np
import pandas as pd

from attacks.fdia import FDIAConfig, inject_fdia
from attacks.hybrid import HybridConfig, inject_hybrid
from attacks.replay import ReplayConfig, inject_replay
from detection.detector import ResidualDetector
from power_system.network import ThreeGeneratorSystem
from sppf.dynamics import StatePartitionParticleFilter
from sppf.partition import StatePartitioner


PAPER_INSPIRED_INTENSITIES = (0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.04, 0.045)
RESULT_COLUMNS = [
    "test_case", "attack_type", "attack_intensity", "operating_condition", "attack_start", "attack_end",
    "attacked_channels", "detected", "detection_delay", "detection_delay_seconds", "threshold", "max_score",
    "delta_mse", "omega_mse", "overall_mse", "state_rmse", "false_alarm_rate", "runtime_seconds",
]


@dataclass(frozen=True)
class PaperCaseConfig:
    seed: int = 20260907
    dt: float = 0.001
    steps: int = 200
    particle_count: int = 220
    attack_start_index: int = 60
    attack_duration_steps: int = 100
    replay_delay_steps: int = 5
    replay_reference_start_index: int | None = None
    replay_channels: tuple[int, ...] = tuple(range(9))
    # Calibration comes from an entirely attack-free run, allowing an early
    # replay start experiment without using attacked observations.
    calibration_start_index: int = 30
    calibration_end_index: int = 90
    threshold_multiplier: float = 4.0
    output_directory: str = "results/paper_test_cases"

    @property
    def attack_end_index(self) -> int:
        return self.attack_start_index + self.attack_duration_steps


def _covariance(values: tuple[float, ...] | np.ndarray) -> np.ndarray:
    return np.diag(np.asarray(values, dtype=float) ** 2)


def _condition_scale(condition: str) -> float:
    if condition == "nominal":
        return 1.0
    if condition == "active_mechanical_power_plus_50pct":
        # The current swing model supports mechanical active-power input but
        # not an AC power-flow operating-point solve.
        return 1.5
    raise ValueError(f"unsupported operating condition: {condition}")


def _generate_clean(config: PaperCaseConfig, operating_condition: str) -> tuple[ThreeGeneratorSystem, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    system = ThreeGeneratorSystem(config.dt, mechanical_power_scale=_condition_scale(operating_condition))
    rng = np.random.default_rng(config.seed)
    process_std = np.array([0.00010, 0.00045] * 3)
    measurement_std = np.array([0.0035] * 3 + [0.006] * 3 + [0.010] * 3)
    initial_truth = np.array([0.125, 0.0, -0.065, 0.0, 0.045, 0.0])
    truth = system.simulate(config.steps, initial_truth, process_std, rng)
    measurement_covariance = _covariance(measurement_std)
    clean = system.measurement(truth) + rng.multivariate_normal(np.zeros(system.measurement_dimension), measurement_covariance, size=config.steps)
    return system, truth, clean, measurement_covariance, _covariance(np.array([0.00008, 0.00035] * 3))


def _run_filter(
    system: ThreeGeneratorSystem, measurements: np.ndarray, measurement_covariance: np.ndarray,
    process_covariance: np.ndarray, config: PaperCaseConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    initial_covariance = _covariance(np.array([0.008, 0.006] * 3))
    initial_truth = np.array([0.125, 0.0, -0.065, 0.0, 0.045, 0.0])
    rng = np.random.default_rng(config.seed + 1001)
    initial = initial_truth + rng.multivariate_normal(np.zeros(system.state_dimension), initial_covariance)
    filter_ = StatePartitionParticleFilter(system, StatePartitioner.three_generator_default(), config.particle_count,
                                           initial, initial_covariance, process_covariance, measurement_covariance, rng)
    estimates = np.empty((config.steps, system.state_dimension), dtype=float)
    prior = np.empty_like(measurements)
    neff = np.empty((config.steps, 3), dtype=float)
    estimates[0] = initial; prior[0] = system.measurement(initial); neff[0] = config.particle_count
    for index in range(1, config.steps):
        estimates[index], prior[index], neff[index] = filter_.step(measurements[index], (index - 1) * config.dt)
    if not (np.all(np.isfinite(estimates)) and np.all(np.isfinite(prior)) and np.all(np.isfinite(neff))):
        raise FloatingPointError("paper-case SP-PF output contains NaN/Inf")
    for local_filter in filter_.filters:
        if not np.isclose(local_filter.weights.sum(), 1.0, atol=1e-10) or not (1 <= local_filter.effective_sample_size() <= config.particle_count):
            raise FloatingPointError("paper-case particle filter invariant failed")
    return estimates, prior, neff


def _apply_attack(clean: np.ndarray, config: PaperCaseConfig, attack_type: str, intensity: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, ...]]:
    channels = (6, 7)  # Synthetic electrical-power measurements for G1/G2.
    fdia = FDIAConfig(config.attack_start_index, config.attack_end_index, intensity, channels)
    replay = ReplayConfig(config.attack_start_index, config.attack_end_index, delay_steps=config.replay_delay_steps,
                          measurement_indices=config.replay_channels, blend=intensity, reference_start_index=config.replay_reference_start_index)
    if attack_type == "none":
        return clean.copy(), np.zeros(config.steps, dtype=bool), np.zeros_like(clean), channels
    if attack_type == "fdi":
        return (*inject_fdia(clean, fdia), channels)
    if attack_type == "replay":
        return (*inject_replay(clean, replay), config.replay_channels)
    if attack_type == "hybrid":
        return (*inject_hybrid(clean, HybridConfig(replay=replay, fdia=fdia)), tuple(sorted(set(channels + config.replay_channels))))
    raise ValueError(f"unsupported attack type: {attack_type}")


def run_paper_case(
    test_case: str, attack_type: str, intensity: float, operating_condition: str = "nominal", config: PaperCaseConfig = PaperCaseConfig(),
) -> dict[str, Any]:
    """Run one attack-free calibrated prototype test case."""
    if config.attack_end_index > config.steps or not (0 <= intensity <= 1):
        raise ValueError("invalid paper-case interval or intensity")
    started = perf_counter()
    system, truth, clean, measurement_covariance, process_covariance = _generate_clean(config, operating_condition)
    normal_estimates, normal_prior, _ = _run_filter(system, clean, measurement_covariance, process_covariance, config)
    calibration_detector = ResidualDetector(measurement_covariance, config.threshold_multiplier)
    normal_scores = np.array([calibration_detector.score(clean[index], normal_prior[index]) for index in range(config.steps)])
    threshold = calibration_detector.calibrate(normal_scores[config.calibration_start_index:config.calibration_end_index])
    attacked, attack_mask, attack_vector, channels = _apply_attack(clean, config, attack_type, intensity)
    estimates, prior, neff = _run_filter(system, attacked, measurement_covariance, process_covariance, config)
    detector = ResidualDetector(measurement_covariance, config.threshold_multiplier)
    scores = np.array([detector.score(attacked[index], prior[index]) for index in range(config.steps)])
    flags = scores > threshold
    flags[:config.calibration_start_index] = False
    candidates = np.flatnonzero(flags & attack_mask)
    detected = bool(candidates.size)
    detection_index = int(candidates[0]) if detected else None
    delta_indices, omega_indices = np.array([0, 2, 4]), np.array([1, 3, 5])
    error = estimates - truth
    normal_window = np.arange(config.steps) < config.attack_start_index
    row = {
        "test_case": test_case, "attack_type": attack_type, "attack_intensity": intensity,
        "operating_condition": operating_condition, "attack_start": config.attack_start_index,
        "attack_end": config.attack_end_index, "attacked_channels": ",".join(map(str, channels)),
        "detected": detected, "detection_delay": None if detection_index is None else detection_index - config.attack_start_index,
        "detection_delay_seconds": None if detection_index is None else (detection_index - config.attack_start_index) * config.dt,
        "threshold": float(threshold), "max_score": float(np.max(scores)),
        "delta_mse": float(np.mean(error[:, delta_indices] ** 2)), "omega_mse": float(np.mean(error[:, omega_indices] ** 2)),
        "overall_mse": float(np.mean(error ** 2)), "state_rmse": float(np.sqrt(np.mean(error ** 2))),
        "false_alarm_rate": float(np.count_nonzero(flags & normal_window) / np.count_nonzero(normal_window)),
        "runtime_seconds": float(perf_counter() - started),
    }
    numeric = (truth, attacked, estimates, scores, prior, neff, normal_scores)
    if not all(np.all(np.isfinite(value)) for value in numeric):
        raise FloatingPointError("non-finite paper-case metric input")
    return {"row": row, "truth": truth, "clean_measurements": clean, "attacked_measurements": attacked,
            "attack_vector": attack_vector, "attack_mask": attack_mask, "estimates": estimates, "scores": scores,
            "threshold": threshold, "flags": flags, "normal_estimates": normal_estimates, "normal_scores": normal_scores,
            "config": config}


def _plot_detection(data: dict[str, Any], filename: Path, title: str) -> None:
    import matplotlib.pyplot as plt
    config = data["config"]; time = np.arange(config.steps) * config.dt
    figure, axis = plt.subplots(figsize=(9, 4))
    axis.plot(time, data["scores"], label="innovation J-score")
    axis.axhline(data["threshold"], color="tab:red", linestyle="--", label="attack-free threshold")
    mask = data["attack_mask"]
    if np.any(mask):
        indices = np.flatnonzero(mask); axis.axvspan(time[indices[0]], time[indices[-1]], color="tab:red", alpha=0.12, label="attack interval")
    axis.set(title=title, xlabel="Time (s)", ylabel="J-score"); axis.legend(); axis.grid(alpha=0.25); figure.tight_layout(); figure.savefig(filename, dpi=160); plt.close(figure)


def _plot_states(main: dict[str, dict[str, Any]], index_set: tuple[int, int, int], label: str, filename: Path) -> None:
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(3, 4, figsize=(14, 8), sharex=True)
    for column, (case_name, data) in enumerate(main.items()):
        time = np.arange(data["config"].steps) * data["config"].dt
        for row, state_index in enumerate(index_set):
            axis = axes[row, column]
            axis.plot(time, data["truth"][:, state_index], label="true", linewidth=1.15)
            axis.plot(time, data["estimates"][:, state_index], label="SP-PF", linewidth=1.0)
            axis.set_title(case_name); axis.set_ylabel(f"{label}{row + 1}"); axis.grid(alpha=0.2)
            if row == 0 and column == 0: axis.legend(fontsize=7)
    for axis in axes[-1]: axis.set_xlabel("Time (s)")
    figure.suptitle(f"Prototype rotor-{label} estimation by attack type"); figure.tight_layout(); figure.savefig(filename, dpi=160); plt.close(figure)


def _configure_matplotlib(output: Path) -> None:
    cache = output / ".matplotlib"; cache.mkdir(parents=True, exist_ok=True); os.environ["MPLCONFIGDIR"] = str(cache.resolve())
    import matplotlib
    matplotlib.use("Agg")


def _matrix(main: dict[str, dict[str, Any]]) -> pd.DataFrame:
    fdi = main["FDI-1 prototype strong electrical-power"] ["row"]
    replay = main["Replay nominal"] ["row"]
    hybrid = main["Hybrid nominal"] ["row"]
    return pd.DataFrame([
        {"paper_case": "FDI-1", "paper_description": "50% active-power perturbation", "implemented": "prototype equivalent",
         "our_test_name": "FDI-1 prototype strong electrical-power", "our_result": f"detected={fdi['detected']}; delay={fdi['detection_delay']}",
         "limitation": "Current model has synthetic electrical-power channels, not separately validated active-power/RTU measurements."},
        {"paper_case": "FDI-2", "paper_description": "50% reactive-power perturbation", "implemented": "not directly reproducible",
         "our_test_name": "none", "our_result": "not run", "limitation": "No reactive-power measurement channels in the current second-order synthetic model."},
        {"paper_case": "FDI-3", "paper_description": "30% reactive-power perturbation", "implemented": "not directly reproducible",
         "our_test_name": "none", "our_result": "not run", "limitation": "No reactive-power measurement channels in the current second-order synthetic model."},
        {"paper_case": "Operating conditions", "paper_description": "+50% active and reactive conditions", "implemented": "active-input approximation only",
         "our_test_name": "active_mechanical_power_plus_50pct", "our_result": "mechanical input scaled by 1.5", "limitation": "No AC power-flow/reaction-power operating point in the current model."},
        {"paper_case": "Replay", "paper_description": "Delayed measurement replay", "implemented": "prototype implementation",
         "our_test_name": "Replay nominal / active", "our_result": f"nominal detected={replay['detected']}", "limitation": "Replay uses synthetic measurement history and configurable blend, not paper data."},
        {"paper_case": "Hybrid", "paper_description": "FDIA plus replay", "implemented": "prototype implementation",
         "our_test_name": "Hybrid nominal", "our_result": f"detected={hybrid['detected']}", "limitation": "Composition order is replay then FDIA; detector remains residual/J-statistic."},
        {"paper_case": "Intensity analysis", "paper_description": "Multiple attack intensities", "implemented": "paper-inspired prototype sweep",
         "our_test_name": "FDI/replay/hybrid intensity sweep", "our_result": "0 to 0.045 levels", "limitation": "Intensity meanings are not numerically comparable to the paper."},
        {"paper_case": "State estimation/MSE", "paper_description": "Rotor angle and speed validation", "implemented": "prototype MSE",
         "our_test_name": "delta/omega plots and MSE", "our_result": "reported in paper-case results", "limitation": "Do not compare directly with paper MSE due to different model/normalization."},
    ])


def run_paper_suite(config: PaperCaseConfig = PaperCaseConfig()) -> dict[str, Any]:
    """Run the paper-inspired, clearly delimited prototype cases and plots."""
    if config.attack_end_index > config.steps or config.calibration_end_index > config.steps:
        raise ValueError("paper-case time windows exceed step count")
    output = Path(config.output_directory); output.mkdir(parents=True, exist_ok=True); _configure_matplotlib(output)
    main = {
        "Nominal": run_paper_case("Nominal control", "none", 0.0, "nominal", config),
        "FDI-1 prototype strong electrical-power": run_paper_case("FDI-1 prototype strong electrical-power", "fdi", 0.50, "nominal", config),
        "Replay nominal": run_paper_case("Replay nominal", "replay", 1.0, "nominal", config),
        # 15% is a moderate FDIA component under the existing sweep's scale;
        # the replay component remains a full selected-channel replay.
        "Hybrid nominal": run_paper_case("Hybrid nominal", "hybrid", 0.15, "nominal", config),
        "FDI-1 active +50%": run_paper_case("FDI-1 active +50%", "fdi", 0.50, "active_mechanical_power_plus_50pct", config),
        "Replay active +50%": run_paper_case("Replay active +50%", "replay", 1.0, "active_mechanical_power_plus_50pct", config),
    }
    intensity_results: list[dict[str, Any]] = []
    for attack_type in ("fdi", "replay", "hybrid"):
        for intensity in PAPER_INSPIRED_INTENSITIES:
            intensity_results.append(run_paper_case(f"Intensity {attack_type} {intensity:g}", attack_type, intensity, "nominal", config))
    all_results = list(main.values()) + intensity_results
    rows = [result["row"] for result in all_results]
    pd.DataFrame(rows, columns=RESULT_COLUMNS).to_csv(output / "results.csv", index=False)
    (output / "results.json").write_text(json.dumps({"configuration": asdict(config), "results": rows}, indent=2), encoding="utf-8")
    matrix = _matrix(main); matrix.to_csv(Path("results") / "paper_testcase_matrix.csv", index=False)
    _plot_detection(main["FDI-1 prototype strong electrical-power"], output / "fdia_detection.png", "FDI-1 prototype electrical-power perturbation")
    _plot_detection(main["Replay nominal"], output / "replay_detection.png", "Replay detection: nominal condition")
    _plot_detection(main["Hybrid nominal"], output / "hybrid_detection.png", "Hybrid detection: nominal condition")
    _plot_states({key: main[key] for key in ("Nominal", "FDI-1 prototype strong electrical-power", "Replay nominal", "Hybrid nominal")}, (0, 2, 4), "delta", output / "rotor_angle_estimation.png")
    _plot_states({key: main[key] for key in ("Nominal", "FDI-1 prototype strong electrical-power", "Replay nominal", "Hybrid nominal")}, (1, 3, 5), "omega", output / "rotor_speed_estimation.png")
    import matplotlib.pyplot as plt
    frame = pd.DataFrame([result["row"] for result in intensity_results])
    figure, axis = plt.subplots(figsize=(8, 4.5))
    for attack_type, group in frame.groupby("attack_type"):
        axis.plot(group["attack_intensity"], group["detected"].astype(float), marker="o", label=attack_type)
    axis.set(title="Paper-inspired prototype attack-intensity sweep", xlabel="Intensity (prototype scale)", ylabel="Detected (0/1)", ylim=(-0.05, 1.05)); axis.legend(); axis.grid(alpha=0.25); figure.tight_layout(); figure.savefig(output / "attack_intensity_sweep.png", dpi=160); plt.close(figure)
    main_frame = pd.DataFrame([result["row"] for result in main.values()])
    figure, axis = plt.subplots(figsize=(9, 4.5)); positions = np.arange(len(main_frame)); width = 0.25
    axis.bar(positions - width, main_frame["delta_mse"], width, label="delta prototype MSE")
    axis.bar(positions, main_frame["omega_mse"], width, label="omega prototype MSE")
    axis.bar(positions + width, main_frame["overall_mse"], width, label="overall prototype MSE")
    axis.set(title="Prototype MSE by paper-inspired test case", ylabel="MSE", xticks=positions, xticklabels=["nominal", "FDI", "replay", "hybrid", "FDI +50% Pm", "replay +50% Pm"]); axis.legend(fontsize=8); axis.grid(axis="y", alpha=0.25); figure.tight_layout(); figure.savefig(output / "mse_comparison.png", dpi=160); plt.close(figure)
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    for axis, key in zip(axes.flat[:3], ("FDI-1 prototype strong electrical-power", "Replay nominal", "Hybrid nominal")):
        data = main[key]; time = np.arange(config.steps) * config.dt
        axis.plot(time, data["scores"]); axis.axhline(data["threshold"], color="tab:red", linestyle="--")
        axis.set(title=key, xlabel="Time (s)", ylabel="J-score"); axis.grid(alpha=0.25)
    for attack_type, group in frame.groupby("attack_type"):
        axes[1, 1].plot(group["attack_intensity"], group["detected"].astype(float), marker="o", label=attack_type)
    axes[1, 1].set(title="Intensity detection", xlabel="Intensity", ylabel="Detected (0/1)", ylim=(-0.05, 1.05)); axes[1, 1].legend(); axes[1, 1].grid(alpha=0.25)
    figure.suptitle("Paper-inspired test cases: simplified SP-PF prototype"); figure.tight_layout(); figure.savefig(output / "summary.png", dpi=180); plt.close(figure)
    return {"main": main, "intensities": intensity_results, "rows": rows, "matrix": matrix}
