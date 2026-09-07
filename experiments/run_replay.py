"""Run replay tests, including the requested approximately 0.01 s start."""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
from experiments.paper_cases import PaperCaseConfig, _configure_matplotlib, _plot_detection, run_paper_case


if __name__ == "__main__":
    config = PaperCaseConfig(attack_start_index=10, attack_duration_steps=100, output_directory="results/paper_test_cases/replay")
    output = Path(config.output_directory); output.mkdir(parents=True, exist_ok=True); _configure_matplotlib(output)
    nominal = run_paper_case("Replay nominal start 0.01s", "replay", 1.0, "nominal", config)
    active = run_paper_case("Replay active +50% start 0.01s", "replay", 1.0, "active_mechanical_power_plus_50pct", config)
    _plot_detection(nominal, output / "replay_nominal.png", "Replay nominal, start = 0.010 s")
    _plot_detection(active, output / "replay_active_plus50.png", "Replay active mechanical power +50%, start = 0.010 s")
    print("Replay nominal:", nominal["row"])
    print("Replay active +50%:", active["row"])
