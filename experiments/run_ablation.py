"""Multi-seed fourth-order PF/SP-PF ablation and low-intensity FDIA study.

This is a deliberately separate experiment: it leaves the original demo,
sweeps, adaptive-KL comparison and likelihood comparison unchanged.  One
identical simulation/attack realisation is used for each estimator mode/seed;
both detectors are then evaluated from that same estimator output.
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
from power_system.network import FourthOrderThreeGeneratorSystem, make_power_system
from sppf.dynamics import StatePartitionParticleFilter
from sppf.particle_filter import ParticleFilter
from sppf.partition import AdaptiveKLConfig, AdaptiveKLPartitioner, StatePartitioner


ESTIMATOR_MODES = ("full_pf", "fixed_sppf", "adaptive_kl")
DETECTOR_TYPES = ("residual", "likelihood")


@dataclass(frozen=True)
class AblationConfig:
    """Fair default configuration for the compact five-seed ablation matrix."""

    seeds: tuple[int, ...] = (20260907, 20260908, 20260909, 20260910, 20260911)
    attack_magnitudes: tuple[float, ...] = (0.0, 0.01, 0.02, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30)
    steps: int = 180
    dt: float = 0.02
    particle_count: int = 350
    calibration_start_index: int = 30
    attack_start_index: int = 100
    attack_duration_steps: int = 55
    attack_channels: tuple[int, ...] = (12, 13)
    threshold_multiplier: float = 4.0
    adaptive_kl: AdaptiveKLConfig = AdaptiveKLConfig()
    output_directory: str = "results/ablation"
    make_plots: bool = True

    @property
    def attack_end_index(self) -> int:
        return self.attack_start_index + self.attack_duration_steps


def _covariance(std: np.ndarray) -> np.ndarray:
    return np.diag(np.asarray(std, dtype=float) ** 2)


def _likelihood_from_inputs(inputs: list[dict[str, object]]) -> tuple[float, np.ndarray]:
    """Predictive particle likelihood, summed in log form across partitions."""
    local_values: list[float] = []
    for item in inputs:
        particle_logs = compute_particle_log_likelihood(
            np.asarray(item["measurement"]), np.asarray(item["predicted_measurements"]),
            np.asarray(item["measurement_covariance"]),
        )
        local_values.append(compute_weighted_particle_log_likelihood(particle_logs, np.asarray(item["weights"])))
    partition_logs = np.asarray(local_values, dtype=float)
    return compute_global_log_likelihood(partition_logs), partition_logs


def classification_metrics(flags: np.ndarray, attack_mask: np.ndarray, calibration_start: int, attack_end: int) -> dict[str, float | int]:
    """Sample-level metrics over [calibration_start, attack_end), excluding recovery.

    Positive samples are only the injected attack interval. Negative samples
    are the post-start-up, pre-attack samples. Post-attack recovery is not
    counted as a false alarm, as requested.
    """
    flags = np.asarray(flags, dtype=bool)
    attack_mask = np.asarray(attack_mask, dtype=bool)
    if flags.shape != attack_mask.shape or not (0 <= calibration_start <= attack_end <= flags.size):
        raise ValueError("invalid evaluation window")
    selected = np.arange(flags.size)
    evaluation = (selected >= calibration_start) & (selected < attack_end)
    positive = attack_mask & evaluation
    negative = ~attack_mask & evaluation
    tp = int(np.count_nonzero(flags & positive))
    fn = int(np.count_nonzero(~flags & positive))
    fp = int(np.count_nonzero(flags & negative))
    tn = int(np.count_nonzero(~flags & negative))
    precision = float(tp / (tp + fp)) if tp + fp else 0.0
    recall = float(tp / (tp + fn)) if tp + fn else 0.0
    f1 = float(2.0 * precision * recall / (precision + recall)) if precision + recall else 0.0
    false_alarm_rate = float(fp / (fp + tn)) if fp + tn else 0.0
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1, "false_alarm_rate": false_alarm_rate}


def _validate_filter(filter_: ParticleFilter, particle_count: int) -> None:
    if filter_.particles.shape != (particle_count, filter_.state_dimension):
        raise FloatingPointError("particle array shape changed unexpectedly")
    if not (np.all(np.isfinite(filter_.particles)) and np.all(np.isfinite(filter_.weights))):
        raise FloatingPointError("particles or weights contain NaN/Inf")
    if not np.isclose(float(filter_.weights.sum()), 1.0) or not (1.0 <= filter_.effective_sample_size() <= particle_count):
        raise FloatingPointError("invalid weights or effective sample size")


def _shared_case(config: AblationConfig, seed: int, magnitude: float) -> dict[str, Any]:
    """Construct a deterministic common truth, noise and FDIA realisation."""
    if not (5 <= config.calibration_start_index < config.attack_start_index < config.attack_end_index <= config.steps):
        raise ValueError("invalid calibration or attack interval")
    rng = np.random.default_rng(seed)
    system = make_power_system(model_order=4, dt=config.dt)
    assert isinstance(system, FourthOrderThreeGeneratorSystem)
    process_std = np.array([0.00010, 0.00045, 0.00025, 0.00025] * 3)
    filter_process_std = np.array([0.00008, 0.00035, 0.00020, 0.00020] * 3)
    initial_std = np.array([0.025, 0.018, 0.015, 0.015] * 3)
    measurement_std = np.array([0.0035] * 3 + [0.006] * 3 + [0.006] * 3 + [0.006] * 3 + [0.010] * 3)
    r = _covariance(measurement_std)
    truth = system.simulate(config.steps, None, process_std, rng)
    clean = system.measurement(truth) + rng.multivariate_normal(np.zeros(system.measurement_dimension), r, size=config.steps)
    attack = FDIAConfig(config.attack_start_index, config.attack_end_index, magnitude, config.attack_channels)
    attacked, attack_mask, _ = inject_fdia(clean, attack)
    initial = system.equilibrium_state + rng.multivariate_normal(np.zeros(system.state_dimension), _covariance(initial_std))
    # A separate deterministic PF generator prevents estimator construction
    # from modifying the common physical/noise/attack realisation.
    pf_seed = int(rng.integers(0, np.iinfo(np.int64).max))
    return {
        "system": system, "truth": truth, "attacked": attacked, "attack_mask": attack_mask,
        "initial": initial, "r": r, "q": _covariance(filter_process_std),
        "initial_covariance": _covariance(initial_std), "pf_seed": pf_seed,
    }


def _run_full_pf(case: dict[str, Any], config: AblationConfig) -> dict[str, Any]:
    """Genuine one-filter baseline with particles of shape (N, 12)."""
    system: FourthOrderThreeGeneratorSystem = case["system"]
    rng = np.random.default_rng(case["pf_seed"])
    pf = ParticleFilter(config.particle_count, system.state_dimension, rng)
    pf.initialize(case["initial"], case["initial_covariance"])
    steps = config.steps
    estimates = np.empty_like(case["truth"])
    prior = np.empty_like(case["attacked"])
    log_likelihoods = np.empty(steps)
    partition_counts = np.ones(steps, dtype=int)
    effective_sizes: list[float] = []

    full_indices = np.arange(system.measurement_dimension)
    for index in range(steps):
        if index > 0:
            pf.predict(lambda particles, time=(index - 1) * config.dt: system.transition(particles, time), case["q"])
        predictions = system.measurement(pf.particles)
        prior[index] = np.average(predictions, axis=0, weights=pf.weights)
        inputs = [{"measurement": case["attacked"][index, full_indices], "predicted_measurements": predictions,
                   "weights": pf.weights.copy(), "measurement_covariance": case["r"]}]
        log_likelihoods[index], _ = _likelihood_from_inputs(inputs)
        pf.update(case["attacked"][index], predictions, case["r"])
        if pf.effective_sample_size() < 0.5 * config.particle_count:
            pf.systematic_resample()
        estimates[index] = pf.estimate()
        effective_sizes.append(pf.effective_sample_size())
    _validate_filter(pf, config.particle_count)
    return {"estimates": estimates, "prior": prior, "log_likelihoods": log_likelihoods,
            "partition_counts": partition_counts, "effective_sizes": np.asarray(effective_sizes),
            "merge_count": 0, "split_count": 0, "particles_total_final": config.particle_count}


def _run_sppf(case: dict[str, Any], config: AblationConfig, adaptive: bool) -> dict[str, Any]:
    """Run an existing fixed/adaptive SP-PF path without altering its logic."""
    system: FourthOrderThreeGeneratorSystem = case["system"]
    rng = np.random.default_rng(case["pf_seed"])
    controller = AdaptiveKLPartitioner(config.adaptive_kl) if adaptive else None
    sppf = StatePartitionParticleFilter(
        system, StatePartitioner.fourth_order_default(), config.particle_count, case["initial"], case["initial_covariance"],
        case["q"], case["r"], rng, partition_mode="adaptive_kl" if adaptive else "fixed", adaptive_partitioner=controller,
    )
    estimates = np.empty_like(case["truth"])
    prior = np.empty_like(case["attacked"])
    log_likelihoods = np.empty(config.steps)
    partition_counts = np.empty(config.steps, dtype=int)
    effective_sizes: list[np.ndarray] = []
    estimates[0] = case["initial"]
    prior[0] = system.measurement(case["initial"])
    log_likelihoods[0], _ = _likelihood_from_inputs(sppf.capture_likelihood_inputs(case["attacked"][0]))
    partition_counts[0] = len(sppf.partitioner.partitions)
    for index in range(1, config.steps):
        estimates[index], prior[index], effective = sppf.step(
            case["attacked"][index], (index - 1) * config.dt, index=index if adaptive else None
        )
        log_likelihoods[index], _ = _likelihood_from_inputs(sppf.last_likelihood_inputs)
        partition_counts[index] = len(sppf.partitioner.partitions)
        effective_sizes.append(effective)
    for filter_ in sppf.filters:
        _validate_filter(filter_, config.particle_count)
    if not all(np.all(np.isfinite(values)) for values in (estimates, prior, log_likelihoods, partition_counts)):
        raise FloatingPointError("SP-PF ablation produced NaN/Inf")
    return {
        "estimates": estimates, "prior": prior, "log_likelihoods": log_likelihoods,
        "partition_counts": partition_counts, "effective_sizes": effective_sizes,
        "merge_count": sum(event["operation"] == "merge" for event in sppf.partition_history),
        "split_count": sum(event["operation"] == "split" for event in sppf.partition_history),
        "particles_total_final": config.particle_count * len(sppf.filters),
    }


def _detector_data(detector_type: str, estimates: dict[str, Any], case: dict[str, Any], config: AblationConfig) -> tuple[np.ndarray, np.ndarray, float, dict[str, float]]:
    calibration = slice(config.calibration_start_index, config.attack_start_index)
    if detector_type == "residual":
        detector = ResidualDetector(case["r"], config.threshold_multiplier)
        scores = np.asarray([detector.score(y, predicted) for y, predicted in zip(case["attacked"], estimates["prior"])])
        threshold = detector.calibrate(scores[calibration])
        flags = detector.flags(scores)
        details: dict[str, float] = {}
    elif detector_type == "likelihood":
        detector = LikelihoodRatioDetector(config.threshold_multiplier)
        threshold = detector.calibrate(estimates["log_likelihoods"][calibration])
        scores = detector.scores(estimates["log_likelihoods"])
        flags = detector.flags(scores)
        details = {"reference_log_likelihood": float(detector.reference_log_likelihood), "reference_log_likelihood_std": float(detector.reference_std)}
    else:
        raise ValueError("unknown detector type")
    flags[:config.calibration_start_index] = False
    if not (np.all(np.isfinite(scores)) and np.isfinite(threshold)):
        raise FloatingPointError("non-finite detector metric")
    return scores, flags, float(threshold), details


def run_estimator_case(config: AblationConfig, seed: int, magnitude: float, estimator_mode: str) -> list[dict[str, Any]]:
    """Run one estimator once, then evaluate both detectors on identical outputs."""
    if estimator_mode not in ESTIMATOR_MODES:
        raise ValueError(f"estimator_mode must be one of {ESTIMATOR_MODES}")
    started = perf_counter()
    case = _shared_case(config, seed, magnitude)
    if estimator_mode == "full_pf":
        estimator = _run_full_pf(case, config)
    else:
        estimator = _run_sppf(case, config, adaptive=estimator_mode == "adaptive_kl")
    runtime = float(perf_counter() - started)
    truth = case["truth"]; state_error = truth - estimator["estimates"]
    delta_indices = np.array([0, 4, 8]); omega_indices = np.array([1, 5, 9])
    outputs: list[dict[str, Any]] = []
    for detector_type in DETECTOR_TYPES:
        scores, flags, threshold, details = _detector_data(detector_type, estimator, case, config)
        candidates = np.flatnonzero(flags & case["attack_mask"])
        detection_index = int(candidates[0]) if candidates.size else None
        metrics = classification_metrics(flags, case["attack_mask"], config.calibration_start_index, config.attack_end_index)
        outputs.append({
            "seed": seed, "model_order": 4, "estimator_mode": estimator_mode, "detector_type": detector_type,
            "attack_magnitude": float(magnitude), "attack_channels": list(config.attack_channels),
            "attack_start": config.attack_start_index, "attack_end": config.attack_end_index,
            "detected": bool(detection_index is not None), "detection_index": detection_index,
            "detection_delay": None if detection_index is None else int(detection_index - config.attack_start_index),
            "detection_delay_seconds": None if detection_index is None else float((detection_index - config.attack_start_index) * config.dt),
            "threshold": threshold, "max_score": float(np.max(scores)), "scores": scores,
            "log_likelihoods": estimator["log_likelihoods"], "state_rmse": float(np.sqrt(np.mean(state_error ** 2))),
            "delta_mse": float(np.mean(state_error[:, delta_indices] ** 2)), "omega_mse": float(np.mean(state_error[:, omega_indices] ** 2)),
            "runtime_seconds": runtime, "particles_per_filter": config.particle_count,
            "particles_total_final": estimator["particles_total_final"], "num_partitions_initial": int(estimator["partition_counts"][0]),
            "num_partitions_final": int(estimator["partition_counts"][-1]), "mean_num_partitions": float(np.mean(estimator["partition_counts"])),
            "merge_count": estimator["merge_count"], "split_count": estimator["split_count"],
            "partition_counts": estimator["partition_counts"], **metrics, **details,
        })
    return outputs


def _mean_std(values: pd.Series) -> tuple[float, float]:
    return float(values.mean()), float(values.std(ddof=0))


def aggregate_metrics(raw: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-seed rows; missed delays remain visible via a count column."""
    grouped: list[dict[str, object]] = []
    for key, group in raw.groupby(["estimator_mode", "detector_type", "attack_magnitude"], sort=False):
        delays = group["detection_delay"].dropna()
        row: dict[str, object] = {"estimator_mode": key[0], "detector_type": key[1], "attack_magnitude": key[2],
                                  "seed_count": int(len(group)), "detection_rate": float(group["detected"].mean()),
                                  "detected_runs": int(group["detected"].sum()), "missed_runs": int((~group["detected"]).sum())}
        for source, prefix in (("state_rmse", "state_rmse"), ("delta_mse", "delta_mse"), ("omega_mse", "omega_mse"),
                               ("false_alarm_rate", "false_alarm_rate"), ("runtime_seconds", "runtime_seconds"),
                               ("precision", "precision"), ("recall", "recall"), ("f1", "f1"), ("mean_num_partitions", "num_partitions")):
            mean, std = _mean_std(group[source]); row[f"mean_{prefix}"] = mean; row[f"std_{prefix}"] = std
        row["mean_detection_delay"] = float(delays.mean()) if not delays.empty else None
        row["std_detection_delay"] = float(delays.std(ddof=0)) if not delays.empty else None
        for name in ("tp", "tn", "fp", "fn", "merge_count", "split_count", "particles_per_filter", "particles_total_final"):
            row[name] = float(group[name].mean())
        grouped.append(row)
    return pd.DataFrame(grouped)


def _plot(raw: pd.DataFrame, summary: pd.DataFrame, config: AblationConfig, directory: Path) -> None:
    cache = directory / ".matplotlib"; cache.mkdir(parents=True, exist_ok=True); os.environ["MPLCONFIGDIR"] = str(cache.resolve())
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    x = np.asarray(config.attack_magnitudes) * 100.0
    def values(mode: str, detector: str, column: str) -> list[float]:
        selected = summary[(summary.estimator_mode == mode) & (summary.detector_type == detector)].sort_values("attack_magnitude")
        return [np.nan if pd.isna(value) else float(value) for value in selected[column]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for axis, detector in zip(axes, DETECTOR_TYPES):
        for mode in ESTIMATOR_MODES:
            axis.plot(x, values(mode, detector, "detection_rate"), marker="o", label=mode)
        axis.set(title=f"{detector}: detection rate", xlabel="FDIA magnitude (%)", ylabel="Detection rate", ylim=(-0.05, 1.05)); axis.grid(alpha=0.25); axis.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(directory / "detection_rate_by_estimator.png", dpi=160); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    for axis, mode in zip(axes, ESTIMATOR_MODES):
        for detector in DETECTOR_TYPES:
            axis.plot(x, values(mode, detector, "detection_rate"), marker="o", label=detector)
        axis.set(title=mode, xlabel="FDIA magnitude (%)", ylabel="Detection rate", ylim=(-0.05, 1.05)); axis.grid(alpha=0.25); axis.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(directory / "detection_rate_residual_vs_likelihood.png", dpi=160); plt.close(fig)

    for filename, column, ylabel, title in (
        ("detection_delay_vs_magnitude.png", "mean_detection_delay", "Mean delay (samples)", "Detection delay across seeds"),
        ("state_rmse_vs_magnitude.png", "mean_state_rmse", "Mean state RMSE", "State RMSE across seeds"),
        ("false_alarm_rate.png", "mean_false_alarm_rate", "Mean false-alarm rate", "Pre-attack false-alarm rate"),
    ):
        fig, axis = plt.subplots(figsize=(9, 4.5))
        for mode in ESTIMATOR_MODES:
            for detector in DETECTOR_TYPES:
                axis.plot(x, values(mode, detector, column), marker="o", label=f"{mode}: {detector}")
        axis.set(xlabel="FDIA magnitude (%)", ylabel=ylabel, title=title); axis.grid(alpha=0.25); axis.legend(fontsize=8, ncol=2)
        fig.tight_layout(); fig.savefig(directory / filename, dpi=160); plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 4.5))
    runtime_summary = summary.groupby("estimator_mode")["mean_runtime_seconds"].mean().reindex(ESTIMATOR_MODES)
    axis.bar(runtime_summary.index, runtime_summary.values, color=["tab:gray", "tab:blue", "tab:orange"])
    axis.set(ylabel="Mean estimator runtime (s)", title="Runtime by estimator mode"); axis.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(directory / "runtime_by_estimator.png", dpi=160); plt.close(fig)

    representative = raw[(raw.estimator_mode == "adaptive_kl") & (raw.detector_type == "likelihood") & (raw.attack_magnitude == 0.05)].iloc[0]
    counts = np.asarray(representative["partition_counts"])
    fig, axis = plt.subplots(figsize=(9, 4))
    axis.step(np.arange(counts.size) * config.dt, counts, where="post")
    axis.set(xlabel="Time (s)", ylabel="Number of partitions", title="Adaptive-KL partition count (5% FDIA, one seed)", ylim=(0.5, 3.5)); axis.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(directory / "adaptive_partition_count_vs_time.png", dpi=160); plt.close(fig)

    representative = raw[(raw.estimator_mode == "fixed_sppf") & (raw.detector_type == "likelihood") & (raw.attack_magnitude == 0.05)].iloc[0]
    time = np.arange(config.steps) * config.dt
    fig, axis = plt.subplots(figsize=(9, 4))
    axis.plot(time, representative["scores"], label="log likelihood degradation")
    axis.axhline(representative["threshold"], color="tab:red", linestyle="--", label="calibrated threshold")
    axis.axvspan(time[config.attack_start_index], time[config.attack_end_index - 1], color="tab:red", alpha=0.12, label="FDIA interval")
    axis.set(xlabel="Time (s)", ylabel="log(Lref / Lk)", title="Example 5% likelihood score: fixed SP-PF",); axis.legend(); axis.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(directory / "low_intensity_likelihood_score.png", dpi=160); plt.close(fig)


def _write_summary(summary: pd.DataFrame, config: AblationConfig, directory: Path) -> None:
    # Avoid requiring the optional ``tabulate`` package merely to create a
    # presentable Markdown artifact.
    selected_columns = [
        "estimator_mode", "detector_type", "attack_magnitude", "seed_count",
        "detection_rate", "mean_detection_delay", "mean_false_alarm_rate",
        "mean_state_rmse", "std_state_rmse", "mean_runtime_seconds",
    ]
    header = "| " + " | ".join(selected_columns) + " |"
    separator = "| " + " | ".join(["---"] * len(selected_columns)) + " |"
    table_rows = [header, separator]
    for _, result in summary[selected_columns].iterrows():
        cells: list[str] = []
        for column in selected_columns:
            value = result[column]
            if pd.isna(value):
                cells.append("N/A")
            elif isinstance(value, (float, np.floating)):
                cells.append(f"{float(value):.6g}")
            else:
                cells.append(str(value))
        table_rows.append("| " + " | ".join(cells) + " |")
    lines = [
        "# Baseline and Ablation Study", "",
        "## Scope", "",
        "This multi-seed study compares a genuine 12-state full PF, fixed generator-wise SP-PF, and optional adaptive-KL SP-PF on the same fourth-order synthetic FDIA setup. Both residual/J-statistic and particle-likelihood detectors are applied to the same estimator output.", "",
        "## Fairness and particle budget", "",
        f"All runs use seeds {list(config.seeds)}, identical FDIA timing/channels, process/measurement covariances, calibration indices {config.calibration_start_index}:{config.attack_start_index}, and {config.particle_count} particles per filter. Full PF therefore has {config.particle_count} total particles; fixed SP-PF has {config.particle_count} particles in each of three filters (1050 total). Adaptive-KL has {config.particle_count} particles per active partition. Runtime and accuracy conclusions must be read with this explicitly different total particle budget.", "",
        "## Evaluation window", "",
        "Sample-level TP/FN are counted only during the attack interval. TN/FP are counted from the end of PF warm-up through the sample immediately before attack onset. Post-attack recovery samples are excluded from false-alarm statistics.", "",
        "## Aggregate results", "", *table_rows, "",
        "## Interpretation rule", "",
        "These are prototype measurements. A mode is not described as superior unless detection, RMSE, runtime, and multi-seed variation support that claim. Adaptive-KL is an optional prototype and is not assumed to improve the fixed partition.",
    ]
    (directory / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _serialisable_raw(raw: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in raw.to_dict(orient="records"):
        # Arrays are retained in-memory for plotting but result JSON remains a
        # compact metric file rather than an unnecessary particle trace dump.
        row.pop("scores", None); row.pop("log_likelihoods", None); row.pop("partition_counts", None)
        rows.append(row)
    return rows


def run_ablation(config: AblationConfig = AblationConfig()) -> dict[str, Any]:
    """Run all seeds/magnitudes/estimator modes and both detector analyses."""
    rows: list[dict[str, Any]] = []
    for magnitude in config.attack_magnitudes:
        for seed in config.seeds:
            for mode in ESTIMATOR_MODES:
                rows.extend(run_estimator_case(config, seed, magnitude, mode))
    raw = pd.DataFrame(rows)
    summary = aggregate_metrics(raw)
    directory = Path(config.output_directory); directory.mkdir(parents=True, exist_ok=True)
    raw_csv = raw.copy()
    for column in ("scores", "log_likelihoods", "partition_counts"):
        raw_csv[column] = raw_csv[column].apply(lambda value: json.dumps(np.asarray(value).tolist()))
    raw_csv["attack_channels"] = raw_csv["attack_channels"].apply(lambda value: ",".join(map(str, value)))
    raw_csv.to_csv(directory / "ablation_results.csv", index=False)
    payload = {"configuration": asdict(config), "per_seed_results": _serialisable_raw(raw), "aggregate_results": summary.to_dict(orient="records")}
    (directory / "ablation_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_summary(summary, config, directory)
    if config.make_plots:
        _plot(raw, summary, config, directory)
    return {"raw": raw, "summary": summary}


def print_report(result: dict[str, Any]) -> None:
    summary: pd.DataFrame = result["summary"]
    print("=" * 72)
    print("FOURTH-ORDER PF / SP-PF LOW-INTENSITY ABLATION")
    print("=" * 72)
    print(f"Aggregate scenarios: {len(summary)}")
    print(summary[["estimator_mode", "detector_type", "attack_magnitude", "seed_count", "detection_rate", "mean_detection_delay", "mean_false_alarm_rate", "mean_state_rmse", "mean_runtime_seconds"]].to_string(index=False))


if __name__ == "__main__":
    print_report(run_ablation())
