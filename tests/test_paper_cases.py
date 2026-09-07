import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from attacks.fdia import FDIAConfig
from attacks.hybrid import HybridConfig, inject_hybrid
from attacks.replay import ReplayConfig, inject_replay
from experiments.paper_cases import PAPER_INSPIRED_INTENSITIES, PaperCaseConfig, run_paper_case, run_paper_suite


class PaperCaseAttackTests(unittest.TestCase):
    def test_replay_delays_only_requested_channels(self) -> None:
        measurements = np.arange(60, dtype=float).reshape(10, 6)
        config = ReplayConfig(4, 8, delay_steps=2, measurement_indices=(1, 4), blend=1.0)
        attacked, mask, vector = inject_replay(measurements, config)
        self.assertTrue(mask[4:8].all())
        self.assertTrue(np.array_equal(attacked[4:8, [1, 4]], measurements[2:6, [1, 4]]))
        self.assertTrue(np.allclose(vector[:, [0, 2, 3, 5]], 0.0))

    def test_hybrid_is_replay_then_fdia(self) -> None:
        measurements = np.ones((12, 9))
        replay = ReplayConfig(4, 8, delay_steps=2, measurement_indices=(6,), blend=1.0)
        fdia = FDIAConfig(4, 8, 0.2, (6,))
        attacked, mask, _ = inject_hybrid(measurements, HybridConfig(replay, fdia))
        self.assertTrue(mask[4:8].all())
        self.assertTrue(np.allclose(attacked[4:8, 6], 1.2))


class PaperCaseExperimentTests(unittest.TestCase):
    def test_mse_and_common_result_schema(self) -> None:
        config = PaperCaseConfig(steps=100, attack_start_index=40, attack_duration_steps=40, calibration_end_index=80, particle_count=100)
        result = run_paper_case("test", "fdi", 0.05, config=config)
        row = result["row"]
        self.assertTrue(np.isfinite(row["delta_mse"]))
        self.assertTrue(np.isfinite(row["omega_mse"]))
        self.assertAlmostEqual(row["overall_mse"], float(np.mean((result["estimates"] - result["truth"]) ** 2)))
        self.assertIn("runtime_seconds", row)

    def test_intensity_sweep_outputs_serialize(self) -> None:
        self.assertEqual(PAPER_INSPIRED_INTENSITIES[0], 0.0)
        self.assertEqual(PAPER_INSPIRED_INTENSITIES[-1], 0.045)
        with tempfile.TemporaryDirectory() as temporary_directory:
            config = PaperCaseConfig(steps=100, attack_start_index=40, attack_duration_steps=40, calibration_end_index=80,
                                     particle_count=60, output_directory=temporary_directory)
            result = run_paper_suite(config)
            output = Path(temporary_directory)
            self.assertEqual(len(result["intensities"]), 24)
            self.assertTrue((output / "results.csv").is_file())
            self.assertTrue((output / "results.json").is_file())
            self.assertEqual(len(pd.read_csv(output / "results.csv")), 30)
            self.assertEqual(len(json.loads((output / "results.json").read_text())["results"]), 30)
