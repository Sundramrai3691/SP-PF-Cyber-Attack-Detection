"""Presentation-focused progress demonstration for the current SP-PF prototype.

Run with ``python -m experiments.mentor_demo``.  This module intentionally
orchestrates existing validated components and does not change their algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from attacks.fdia import FDIAConfig
from experiments.run_fdia_sppf import DemoConfig, run_experiment
from experiments.run_fdia_sweep import run_sweep


@dataclass(frozen=True)
class MentorDemoConfig:
    representative_magnitude: float = 0.10
    representative_channels: tuple[int, ...] = (6, 7)
    output_directory: str = "results"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def _ensure_sweep_summary(output_directory: Path) -> dict[str, Any]:
    path = output_directory / "fdia_sweep_results.json"
    if not path.is_file():
        return run_sweep()
    return _load_json(path)


def _ensure_faculty_summary(output_directory: Path) -> dict[str, Any]:
    path = output_directory / "faculty_dataset_summary.json"
    if not path.is_file():
        from experiments.inspect_faculty_dataset import inspect_faculty_dataset
        return inspect_faculty_dataset()
    return _load_json(path)


def _threshold_flags(result: dict[str, Any], threshold: float) -> np.ndarray:
    flags = np.asarray(result["scores"], dtype=float) > threshold
    flags[:DemoConfig().calibration_start_index] = False
    return flags


def _representative_metrics(result: dict[str, Any], threshold: float) -> dict[str, object]:
    flags = _threshold_flags(result, threshold)
    attack_mask = np.asarray(result["attack_mask"], dtype=bool)
    candidates = np.flatnonzero(flags & attack_mask)
    detection_index = int(candidates[0]) if candidates.size else None
    delay_steps = None if detection_index is None else detection_index - int(np.flatnonzero(attack_mask)[0])
    return {
        "detected": detection_index is not None,
        "detection_index": detection_index,
        "detection_delay_steps": delay_steps,
        "detection_delay_seconds": None if delay_steps is None else float(delay_steps * DemoConfig().dt),
        "threshold": float(threshold), "maximum_score": float(np.max(result["scores"])),
        "state_rmse": float(result["state_rmse"]),
    }


def _sweep_rates(sweep: dict[str, Any]) -> dict[float, float]:
    grouped: dict[float, list[bool]] = {}
    for row in sweep["scenarios"]:
        if row["scenario_status"] == "success":
            grouped.setdefault(float(row["attack_magnitude"]), []).append(bool(row["detected"]))
    return {magnitude: float(np.mean(values)) for magnitude, values in sorted(grouped.items())}


def _create_summary_figure(result: dict[str, Any], representative: dict[str, object], rates: dict[float, float], output: Path) -> None:
    cache_dir = output.parent / ".matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_dir.resolve())
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config = result["config"]
    time = np.arange(config.steps) * config.dt
    start, end = config.attack.start_index, config.attack.end_index
    figure, axes = plt.subplots(2, 2, figsize=(14, 9))

    measurement_axis = axes[0, 0]
    channel = config.attack.measurement_indices[0]
    measurement_axis.plot(time, result["true_measurements"][:, channel], label="reference measurement", linewidth=1.5)
    measurement_axis.plot(time, result["attacked_measurements"][:, channel], label="FDIA measurement", linewidth=1.0)
    measurement_axis.axvspan(time[start], time[end - 1], color="tab:red", alpha=0.12, label="FDIA interval")
    measurement_axis.set(title="A. Representative measurement under FDIA", xlabel="Time (s)", ylabel="Electrical power measurement")
    measurement_axis.legend(fontsize=8); measurement_axis.grid(alpha=0.25)

    state_axis = axes[0, 1]
    state_indices = (0, 1, 2, 3, 4, 5)
    labels = ("delta1", "omega1", "delta2", "omega2", "delta3", "omega3")
    colours = ("tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown")
    for index, label, colour in zip(state_indices, labels, colours):
        state_axis.plot(time, result["truth"][:, index], color=colour, linewidth=1.25, label=f"{label} reference")
        state_axis.plot(time, result["estimates"][:, index], color=colour, linestyle="--", linewidth=1.0, label=f"{label} SP-PF")
    state_axis.axvspan(time[start], time[end - 1], color="tab:red", alpha=0.08)
    state_axis.set(title="B. SP-PF state estimate vs reference", xlabel="Time (s)", ylabel="State value")
    state_axis.legend(fontsize=6.5, ncol=2); state_axis.grid(alpha=0.25)

    score_axis = axes[1, 0]
    score_axis.plot(time, result["scores"], label="innovation J-score")
    score_axis.axhline(float(representative["threshold"]), color="tab:red", linestyle="--", label="fixed normal threshold")
    score_axis.axvspan(time[start], time[end - 1], color="tab:red", alpha=0.12, label="FDIA interval")
    score_axis.set(title="C. Residual-based detection", xlabel="Time (s)", ylabel="J-score")
    score_axis.legend(fontsize=8); score_axis.grid(alpha=0.25)

    rate_axis = axes[1, 1]
    magnitudes = sorted(rates)
    rate_axis.plot([magnitude * 100.0 for magnitude in magnitudes], [rates[magnitude] for magnitude in magnitudes], marker="o")
    rate_axis.set(title="D. Sweep detection rate", xlabel="FDIA magnitude (%)", ylabel="Detection rate", ylim=(-0.05, 1.05))
    rate_axis.grid(alpha=0.25)

    figure.suptitle("Working initial SP-PF prototype: mentor progress demonstration", fontsize=15, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _write_progress_summary(
    output: Path, representative: dict[str, object], rates: dict[float, float], faculty: dict[str, Any], config: MentorDemoConfig,
) -> None:
    workbook = faculty["workbooks"][0]
    sheets = ", ".join(workbook["sheet_names"])
    sampling = workbook["sheets"][0]["time"]
    rate_lines = "\n".join(f"- {magnitude * 100:.0f}%: {rate * 100:.0f}% detection across configured channels" for magnitude, rate in rates.items())
    delay = "not detected" if representative["detection_delay_steps"] is None else f"{representative['detection_delay_steps']} samples ({representative['detection_delay_seconds']:.3f} s)"
    text = f"""# SP-PF Cyber-Physical Attack Detection: Mentor Progress Summary

## PROJECT OBJECTIVE

Build and validate a working initial State Partition-Particle Filter (SP-PF) prototype for FDIA detection in a controlled nonlinear power-system simulation.

## CURRENT IMPLEMENTATION

- SP-PF means **State Partition-Particle Filter**: the six-dimensional state is split into three fixed two-state generator partitions, and each partition uses its own particle population before local estimates are recombined.
- Three-generator, second-order nonlinear swing-equation prototype.
- State: `[delta1, omega1, delta2, omega2, delta3, omega3]`.
- Fixed partitions: `[0,1]`, `[2,3]`, `[4,5]`; 600 particles per partition.
- FDIA injection and an innovation residual/J-statistic detector with a threshold calibrated from attack-free data only.
- Reproducible 21-scenario FDIA sweep and separate faculty-workbook inspection/adapter.

## CURRENT RESULTS

Representative FDIA: {config.representative_magnitude * 100:.0f}% on channels {list(config.representative_channels)}.

- Detected: {'YES' if representative['detected'] else 'NO'}.
- Detection delay: {delay}.
- Fixed threshold: {representative['threshold']:.3f}; maximum score: {representative['maximum_score']:.3f}.
- State RMSE: {representative['state_rmse']:.6f}.

Sweep detection rates:
{rate_lines}

## FACULTY DATASET FINDINGS

- Workbook: `data_statedata.xlsx`; sheets: {sheets}.
- Sampling is uniform at {sampling['mean_dt']} ms ({sampling['mean_dt'] * 1000:.0f} microsecond).
- Numerical evidence supports sampled instantaneous three-phase voltage waveforms in labelled kV units, not PMU phasors and not direct generator states.
- Direct compatibility with the current estimator: **No**. A validated waveform-to-measurement/state observation model is required first.

## CURRENT LIMITATIONS

- Working prototype only: not an exact reproduction of the paper or a production detector.
- Second-order generator dynamics, fixed partitions, residual/J-statistic detector, synthetic nonlinear simulation, and FDIA only.
- No adaptive KL partitioning, replay/hybrid attacks, fourth-order machine model, or paper-exact likelihood-ratio detector.
- Faculty voltage waveforms are not currently mapped to the prototype state-space model.

## NEXT TECHNICAL OPTIONS

### Option A - Continue entirely in Python (default recommendation)

The current implementation already works, is reproducible with NumPy/SciPy, and processes Excel data directly. It avoids an environment switch while the algorithm and its validation mature.

### Option B - MATLAB/MATPOWER simulation with Python SP-PF

This can provide a more established power-system simulation source while retaining the working Python estimator. It requires a defined MATLAB/Python interface, conversion of measurements and states, plus setup and debugging of the cross-tool workflow.

### Option C - Move the entire implementation to MATLAB

This entails substantial reimplementation. It should be chosen only if MATLAB is specifically required by the project or mentor, not merely for presentation.

## QUESTIONS FOR MENTOR

1. Should we continue with the current Python-based implementation, or is MATLAB/MATPOWER expected for the final project?
2. Does the faculty dataset need to become the primary experimental dataset, or can it remain a separate validation/input dataset while the algorithm is developed on a controlled simulation?
3. Should we target the paper's full fourth-order generator model?
4. Should adaptive KL-based partitioning be the next implementation milestone?
5. Does Ma'am expect an exact reproduction of the paper's likelihood-ratio detection method, or is a validated prototype acceptable initially?
6. Are there additional system measurements/model parameters available for the faculty dataset so that a physically meaningful state/measurement model can be constructed?
"""
    output.write_text(text, encoding="utf-8")


def run_mentor_demo(config: MentorDemoConfig = MentorDemoConfig()) -> dict[str, object]:
    """Run the concise progress demo and create mentor-facing artefacts."""
    output_directory = Path(config.output_directory)
    control_attack = FDIAConfig(120, 190, 0.0, config.representative_channels)
    control = run_experiment(replace(DemoConfig(), attack=control_attack), output_dir=None, make_plots=False)
    control_threshold = float(control["threshold"])
    control_false_alarms = int(np.count_nonzero(_threshold_flags(control, control_threshold)))

    representative_attack = FDIAConfig(120, 190, config.representative_magnitude, config.representative_channels)
    representative_result = run_experiment(replace(DemoConfig(), attack=representative_attack), output_dir=None, make_plots=False)
    if not all(np.all(np.isfinite(np.asarray(representative_result[key], dtype=float))) for key in ("truth", "estimates", "scores", "effective_sizes")):
        raise FloatingPointError("mentor demonstration produced a non-finite quantity")
    representative = _representative_metrics(representative_result, control_threshold)
    sweep = _ensure_sweep_summary(output_directory)
    faculty = _ensure_faculty_summary(output_directory)
    rates = _sweep_rates(sweep)
    output_directory.mkdir(parents=True, exist_ok=True)
    figure = output_directory / "mentor_demo_summary.png"
    summary = output_directory / "mentor_progress_summary.md"
    _create_summary_figure(representative_result, representative, rates, figure)
    _write_progress_summary(summary, representative, rates, faculty, config)
    return {
        "control_false_alarm_count": control_false_alarms,
        "representative": representative,
        "sweep_rates": rates,
        "figure": str(figure), "summary": str(summary),
        "faculty_sheet_names": faculty["workbooks"][0]["sheet_names"],
    }


def print_report(result: dict[str, object], config: MentorDemoConfig = MentorDemoConfig()) -> None:
    representative = result["representative"]
    print("=" * 52)
    print("SP-PF CYBER-PHYSICAL ATTACK DETECTION")
    print("MENTOR PROGRESS DEMO")
    print("=" * 52)
    print("System: 3-generator nonlinear working prototype")
    print("State: [delta1, omega1, delta2, omega2, delta3, omega3]")
    print("Partitions: [0,1], [2,3], [4,5]")
    print("Particles / partition: 600")
    print("Detection: residual / J-statistic; fixed attack-free threshold")
    print("SP-PF: State Partition-Particle Filter; one particle filter per fixed generator partition")
    print(f"Attack-free control: completed; false alarms after warm-up = {result['control_false_alarm_count']}")
    print("\n--- REPRESENTATIVE FDIA RESULT ---")
    print(f"Attack magnitude: {config.representative_magnitude * 100:.0f}%")
    print(f"Attacked channels: {list(config.representative_channels)}")
    print("Detected:", "YES" if representative["detected"] else "NO")
    print("Detection delay:", representative["detection_delay_steps"], "samples /", representative["detection_delay_seconds"], "s")
    print(f"Threshold: {representative['threshold']:.3f}")
    print(f"Maximum score: {representative['maximum_score']:.3f}")
    print(f"State RMSE: {representative['state_rmse']:.6f}")
    print("\n--- MULTI-SCENARIO SUMMARY ---")
    for magnitude, rate in result["sweep_rates"].items():
        print(f"{magnitude * 100:.0f}% -> {rate * 100:.0f}% detection")
    print("\n--- FACULTY DATASET ---")
    print("Sheets:", ", ".join(result["faculty_sheet_names"]))
    print("Sampling: 0.001 ms (1 microsecond), uniform")
    print("Observed representation: sampled three-phase voltage waveforms")
    print("Direct compatibility with current SP-PF: NO")
    print("Reason: current estimator requires a validated state/measurement observation model")
    print("\nOutputs:", result["figure"], "and", result["summary"])


if __name__ == "__main__":
    print_report(run_mentor_demo())
