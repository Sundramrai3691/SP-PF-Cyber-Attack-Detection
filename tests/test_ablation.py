"""Coverage for the full-PF/SP-PF ablation framework."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.run_ablation import AblationConfig, aggregate_metrics, classification_metrics, run_ablation, run_estimator_case


class AblationTests(unittest.TestCase):
    @staticmethod
    def config(output: str = "results/ablation_test") -> AblationConfig:
        return AblationConfig(
            seeds=(20260907,), attack_magnitudes=(0.0, 0.05), steps=70,
            particle_count=60, calibration_start_index=10, attack_start_index=35,
            attack_duration_steps=20, output_directory=output, make_plots=False,
        )

    def test_full_pf_is_one_genuine_twelve_state_filter(self) -> None:
        rows = run_estimator_case(self.config(), 20260907, 0.05, "full_pf")
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["particles_total_final"] == 60 for row in rows))
        self.assertTrue(all(row["num_partitions_initial"] == 1 for row in rows))
        self.assertTrue(all(np.isfinite(row["state_rmse"]) for row in rows))

    def test_confusion_metrics_exclude_post_attack_recovery(self) -> None:
        flags = np.array([False, False, True, True, False, True, True])
        attack = np.array([False, False, False, True, True, False, False])
        metrics = classification_metrics(flags, attack, calibration_start=2, attack_end=5)
        self.assertEqual((metrics["tp"], metrics["tn"], metrics["fp"], metrics["fn"]), (1, 0, 1, 1))
        self.assertAlmostEqual(float(metrics["precision"]), 0.5)
        self.assertAlmostEqual(float(metrics["recall"]), 0.5)

    def test_multi_seed_aggregate_reports_variation(self) -> None:
        first = run_estimator_case(self.config(), 20260907, 0.05, "fixed_sppf")
        second = run_estimator_case(self.config(), 20260908, 0.05, "fixed_sppf")
        summary = aggregate_metrics(__import__("pandas").DataFrame(first + second))
        self.assertEqual(len(summary), 2)
        self.assertTrue((summary["seed_count"] == 2).all())
        self.assertTrue(np.all(np.isfinite(summary["mean_state_rmse"])))

    def test_result_serialization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = str(Path(temporary) / "ablation")
            result = run_ablation(self.config(output))
            self.assertEqual(len(result["raw"]), 12)
            directory = Path(output)
            self.assertTrue((directory / "ablation_results.csv").is_file())
            payload = json.loads((directory / "ablation_results.json").read_text(encoding="utf-8"))
            self.assertEqual(len(payload["per_seed_results"]), 12)
            self.assertTrue((directory / "summary.md").is_file())


if __name__ == "__main__":
    unittest.main()
