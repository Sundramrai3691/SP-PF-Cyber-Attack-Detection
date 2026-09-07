"""Run one moderate hybrid replay-plus-FDIA prototype case."""
from __future__ import annotations
from pathlib import Path
from experiments.paper_cases import PaperCaseConfig, _configure_matplotlib, _plot_detection, run_paper_case


if __name__ == "__main__":
    config = PaperCaseConfig(output_directory="results/paper_test_cases/hybrid")
    output = Path(config.output_directory); output.mkdir(parents=True, exist_ok=True); _configure_matplotlib(output)
    result = run_paper_case("Hybrid moderate nominal", "hybrid", 0.15, "nominal", config)
    _plot_detection(result, output / "hybrid_moderate.png", "Hybrid replay + FDIA, intensity 0.15")
    print("Hybrid moderate:", result["row"])
