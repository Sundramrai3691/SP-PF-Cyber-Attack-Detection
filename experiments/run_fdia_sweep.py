"""Deterministic multi-scenario FDIA evaluation for the existing SP-PF demo.

This evaluates the current simplified fixed-partition/residual prototype; it
does not change the single-scenario ``run_demo.py`` smoke test.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from attacks.fdia import FDIAConfig
from experiments.run_fdia_sppf import DemoConfig, run_experiment


@dataclass(frozen=True)
class SweepConfig:
    """Configuration for a quick, reproducible single-seed FDIA sweep."""

    seed: int = 20260907
    particle_count: int = 600
    steps: int = 240
    attack_start_index: int = 120
    attack_duration_steps: int = 70
    magnitudes: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.30)
    # 3--5 are omega1--3; 6--8 are electrical_power1--3 in network.py.
    channel_configurations: tuple[tuple[int, ...], ...] = ((6, 7), (6,), (7,), (3, 4))
    output_directory: str = "results"

    @property
    def attack_end_index(self) -> int:
        return self.attack_start_index + self.attack_duration_steps


METRIC_COLUMNS = [
    "scenario_id", "scenario_status", "error", "attack_magnitude", "attack_channels",
    "attack_start", "attack_end", "detected", "detection_index",
    "detection_delay_steps", "detection_delay_seconds", "threshold", "max_score",
    "state_rmse", "false_alarm_count", "false_alarm_rate", "random_seed",
    "particle_count_per_partition", "number_of_partitions", "threshold_source",
]


def calculate_false_alarm_metrics(
    flags: np.ndarray, attack_mask: np.ndarray, normal_evaluation_mask: np.ndarray | None = None,
) -> tuple[int, float]:
    """Return false alarms on a declared attack-free evaluation interval.

    Sweep scenarios use the pre-attack interval.  This avoids labelling
    residual recovery after a real attack as an attack-free false alarm.
    """
    flags = np.asarray(flags, dtype=bool)
    attack_mask = np.asarray(attack_mask, dtype=bool)
    normal_mask = ~attack_mask if normal_evaluation_mask is None else np.asarray(normal_evaluation_mask, dtype=bool) & ~attack_mask
    normal_count = int(np.count_nonzero(normal_mask))
    if flags.shape != normal_mask.shape or normal_count == 0:
        raise ValueError("flags and attack mask must align and contain normal samples")
    count = int(np.count_nonzero(flags & normal_mask))
    return count, float(count / normal_count)


def calculate_detection_metrics(
    flags: np.ndarray, attack_mask: np.ndarray, attack_start: int, dt_seconds: float, intentional_attack: bool,
) -> dict[str, object]:
    """Detection is an alarm during an intentionally injected attack interval.

    For the no-attack control, `detected` is deliberately false even if a false
    alarm occurs; that alarm is captured by the false-alarm metrics instead.
    """
    flags = np.asarray(flags, dtype=bool)
    attack_mask = np.asarray(attack_mask, dtype=bool)
    if flags.shape != attack_mask.shape:
        raise ValueError("flags and attack mask must have the same shape")
    candidates = np.flatnonzero(flags & attack_mask) if intentional_attack else np.array([], dtype=int)
    if candidates.size == 0:
        return {"detected": False, "detection_index": None, "detection_delay_steps": None, "detection_delay_seconds": None}
    index = int(candidates[0])
    delay = index - int(attack_start)
    return {"detected": True, "detection_index": index, "detection_delay_steps": delay, "detection_delay_seconds": float(delay * dt_seconds)}


def _validate_result(result: dict[str, object], particle_count: int) -> None:
    for key in ("truth", "attacked_measurements", "estimates", "scores", "effective_sizes", "partition_weight_sums"):
        if not np.all(np.isfinite(np.asarray(result[key], dtype=float))):
            raise FloatingPointError(f"{key} contains NaN/Inf")
    effective_sizes = np.asarray(result["effective_sizes"], dtype=float)
    if np.any(effective_sizes < 1.0) or np.any(effective_sizes > particle_count):
        raise ValueError("effective sample size is outside valid bounds")
    if not np.allclose(np.asarray(result["partition_weight_sums"], dtype=float), 1.0, rtol=0.0, atol=1e-10):
        raise ValueError("partition particle weights do not sum to one")


def _base_demo_config(config: SweepConfig, attack: FDIAConfig) -> DemoConfig:
    return replace(DemoConfig(), seed=config.seed, steps=config.steps, particle_count=config.particle_count, attack=attack)


def _scenario_id(magnitude: float, channels: tuple[int, ...]) -> str:
    magnitude_text = "control" if magnitude == 0.0 else f"fdia_{int(round(magnitude * 100)):02d}pct"
    return f"{magnitude_text}_channels_{'_'.join(str(channel) for channel in channels)}"


def _metrics_row(
    scenario_id: str, magnitude: float, channels: tuple[int, ...], config: SweepConfig,
    threshold: float, result: dict[str, object], flags: np.ndarray,
) -> dict[str, object]:
    _validate_result(result, config.particle_count)
    attack_mask = np.asarray(result["attack_mask"], dtype=bool)
    # The attack-free pre-attack interval is consistent across magnitudes and
    # cannot be contaminated by detector recovery after the injected attack.
    normal_evaluation_mask = np.ones_like(attack_mask, dtype=bool) if magnitude == 0.0 else np.arange(attack_mask.size) < config.attack_start_index
    false_alarm_count, false_alarm_rate = calculate_false_alarm_metrics(flags, attack_mask, normal_evaluation_mask)
    detected_metrics = calculate_detection_metrics(
        flags, attack_mask, config.attack_start_index, DemoConfig().dt, intentional_attack=magnitude > 0.0,
    )
    return {
        "scenario_id": scenario_id, "scenario_status": "success", "error": None,
        "attack_magnitude": magnitude, "attack_channels": ",".join(str(channel) for channel in channels),
        "attack_start": config.attack_start_index, "attack_end": config.attack_end_index,
        **detected_metrics, "threshold": float(threshold), "max_score": float(np.max(result["scores"])),
        "state_rmse": float(result["state_rmse"]), "false_alarm_count": false_alarm_count,
        "false_alarm_rate": false_alarm_rate, "random_seed": config.seed,
        "particle_count_per_partition": config.particle_count, "number_of_partitions": 3,
        "threshold_source": "fixed_attack_free_control",
    }


def _failed_row(scenario_id: str, magnitude: float, channels: tuple[int, ...], config: SweepConfig, error: Exception) -> dict[str, object]:
    row: dict[str, object] = {column: None for column in METRIC_COLUMNS}
    row.update({
        "scenario_id": scenario_id, "scenario_status": "failed", "error": f"{type(error).__name__}: {error}",
        "attack_magnitude": magnitude, "attack_channels": ",".join(str(channel) for channel in channels),
        "attack_start": config.attack_start_index, "attack_end": config.attack_end_index,
        "random_seed": config.seed, "particle_count_per_partition": config.particle_count,
        "number_of_partitions": 3, "threshold_source": "fixed_attack_free_control",
    })
    return row


def _plot_summary(rows: list[dict[str, object]], representative: dict[str, object], config: SweepConfig, output_dir: Path) -> None:
    cache_dir = output_dir / ".matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_dir.resolve())
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    successful = pd.DataFrame([row for row in rows if row["scenario_status"] == "success"])
    grouped = successful.groupby("attack_magnitude", as_index=False).agg(
        detection_rate=("detected", "mean"),
        mean_detection_delay_steps=("detection_delay_steps", "mean"),
        mean_max_score=("max_score", "mean"), mean_state_rmse=("state_rmse", "mean"),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    plots = [
        ("detection_rate_vs_magnitude.png", "detection_rate", "Detection rate", "Detection rate vs FDIA magnitude"),
        ("detection_delay_vs_magnitude.png", "mean_detection_delay_steps", "Mean detection delay (samples)", "Detection delay vs FDIA magnitude"),
        ("maximum_score_vs_magnitude.png", "mean_max_score", "Mean maximum J-score", "Maximum score vs FDIA magnitude"),
        ("state_rmse_vs_magnitude.png", "mean_state_rmse", "Mean state RMSE", "State RMSE vs FDIA magnitude"),
    ]
    for filename, column, ylabel, title in plots:
        fig, axis = plt.subplots(figsize=(7.5, 4.2))
        axis.plot(grouped["attack_magnitude"] * 100.0, grouped[column], marker="o")
        axis.set(xlabel="FDIA magnitude (%)", ylabel=ylabel, title=title)
        axis.grid(alpha=0.25); fig.tight_layout(); fig.savefig(output_dir / filename, dpi=160); plt.close(fig)

    result = representative["result"]
    time = np.arange(config.steps) * DemoConfig().dt
    fig, axis = plt.subplots(figsize=(10, 4))
    axis.plot(time, result["scores"], label="SP-PF innovation J-score")
    axis.axhline(float(representative["threshold"]), color="tab:red", linestyle="--", label="fixed normal threshold")
    start, end = config.attack_start_index, config.attack_end_index
    axis.axvspan(time[start], time[end - 1], color="tab:red", alpha=0.12, label="attack interval")
    axis.set(xlabel="Time (s)", ylabel="J", title=f"Representative scenario: {representative['row']['scenario_id']}")
    axis.legend(); axis.grid(alpha=0.25); fig.tight_layout()
    fig.savefig(output_dir / "representative_detection_score.png", dpi=160); plt.close(fig)


def run_sweep(config: SweepConfig = SweepConfig()) -> dict[str, object]:
    """Run all scenarios, persist metrics, and return them for testing/use."""
    if config.attack_end_index > config.steps:
        raise ValueError("attack interval must fit within configured step count")
    output_root = Path(config.output_directory)
    control_channels = config.channel_configurations[0]
    control_attack = FDIAConfig(config.attack_start_index, config.attack_end_index, 0.0, control_channels)
    control_result = run_experiment(_base_demo_config(config, control_attack), output_dir=None, make_plots=False)
    _validate_result(control_result, config.particle_count)
    fixed_threshold = float(control_result["threshold"])

    scenarios = [(0.0, control_channels)] + [(magnitude, channels) for magnitude in config.magnitudes for channels in config.channel_configurations]
    rows: list[dict[str, object]] = []
    representative: dict[str, object] | None = None
    for magnitude, channels in scenarios:
        scenario_id = _scenario_id(magnitude, channels)
        try:
            attack = FDIAConfig(config.attack_start_index, config.attack_end_index, magnitude, channels)
            result = control_result if magnitude == 0.0 else run_experiment(_base_demo_config(config, attack), output_dir=None, make_plots=False)
            flags = np.asarray(result["scores"], dtype=float) > fixed_threshold
            flags[:DemoConfig().calibration_start_index] = False
            row = _metrics_row(scenario_id, magnitude, channels, config, fixed_threshold, result, flags)
            rows.append(row)
            if magnitude == max(config.magnitudes) and channels == control_channels:
                representative = {"row": row, "result": result, "threshold": fixed_threshold}
        except Exception as error:  # Preserve a failed scenario in output rather than silently omitting it.
            rows.append(_failed_row(scenario_id, magnitude, channels, config, error))
    if representative is None:
        raise RuntimeError("Representative scenario failed; inspect recorded scenario errors")

    output_root.mkdir(parents=True, exist_ok=True)
    dataframe = pd.DataFrame(rows, columns=METRIC_COLUMNS)
    dataframe.to_csv(output_root / "fdia_sweep_results.csv", index=False)
    payload = {
        "configuration": asdict(config),
        "calibration": {"threshold": fixed_threshold, "source": "attack-free control scenario only", "attacked_samples_used": False},
        "scenario_count": len(rows), "successful_scenarios": int((dataframe["scenario_status"] == "success").sum()),
        "failed_scenarios": int((dataframe["scenario_status"] == "failed").sum()), "scenarios": rows,
    }
    (output_root / "fdia_sweep_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _plot_summary(rows, representative, config, output_root / "sweep")
    return payload


def print_report(payload: dict[str, object]) -> None:
    print("=" * 52)
    print("SP-PF MULTI-SCENARIO FDIA SWEEP")
    print("=" * 52)
    print(f"Scenarios: {payload['scenario_count']}; successful: {payload['successful_scenarios']}; failed: {payload['failed_scenarios']}")
    print(f"Fixed attack-free threshold: {payload['calibration']['threshold']:.3f}")
    for row in payload["scenarios"]:
        print(f"{row['scenario_id']}: status={row['scenario_status']}, detected={row['detected']}, "
              f"delay={row['detection_delay_steps']}, false_alarm_rate={row['false_alarm_rate']}, rmse={row['state_rmse']}")


if __name__ == "__main__":
    print_report(run_sweep())
