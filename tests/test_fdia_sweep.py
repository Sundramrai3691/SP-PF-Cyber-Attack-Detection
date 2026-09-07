import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from attacks.fdia import FDIAConfig, inject_fdia
from experiments.run_fdia_sweep import (
    SweepConfig,
    calculate_detection_metrics,
    calculate_false_alarm_metrics,
    run_sweep,
)


class FDIASweepTests(unittest.TestCase):
    def test_attack_magnitude_and_channels_are_applied(self) -> None:
        measurements = np.full((10, 9), 2.0)
        attacked, mask, vector = inject_fdia(measurements, FDIAConfig(3, 7, 0.05, (6, 7)))
        self.assertTrue(mask[3:7].all())
        self.assertTrue(np.allclose(vector[3:7, [6, 7]], 0.1))
        self.assertTrue(np.allclose(vector[:, [0, 1, 2, 3, 4, 5, 8]], 0.0))

    def test_attack_free_case_has_no_intentional_detection(self) -> None:
        flags = np.array([False, True, False])
        attack_mask = np.zeros(3, dtype=bool)
        metrics = calculate_detection_metrics(flags, attack_mask, attack_start=1, dt_seconds=0.02, intentional_attack=False)
        self.assertFalse(metrics["detected"])
        self.assertIsNone(metrics["detection_index"])

    def test_detection_delay_and_false_alarm_metrics(self) -> None:
        flags = np.array([True, False, False, True, False, True])
        attack_mask = np.array([False, False, False, True, True, False])
        detection = calculate_detection_metrics(flags, attack_mask, attack_start=3, dt_seconds=0.02, intentional_attack=True)
        self.assertTrue(detection["detected"])
        self.assertEqual(detection["detection_index"], 3)
        self.assertEqual(detection["detection_delay_steps"], 0)
        self.assertAlmostEqual(detection["detection_delay_seconds"], 0.0)
        count, rate = calculate_false_alarm_metrics(flags, attack_mask)
        self.assertEqual(count, 2)
        self.assertAlmostEqual(rate, 0.5)

    def test_small_sweep_writes_valid_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config = SweepConfig(
                particle_count=120, steps=100, attack_start_index=55, attack_duration_steps=20,
                magnitudes=(0.05, 0.10), channel_configurations=((6,),), output_directory=temporary_directory,
            )
            payload = run_sweep(config)
            self.assertEqual(payload["scenario_count"], 3)
            self.assertEqual(payload["successful_scenarios"], 3)
            control = payload["scenarios"][0]
            self.assertEqual(control["attack_magnitude"], 0.0)
            self.assertFalse(control["detected"])
            csv_path = Path(temporary_directory) / "fdia_sweep_results.csv"
            json_path = Path(temporary_directory) / "fdia_sweep_results.json"
            self.assertTrue(csv_path.is_file() and json_path.is_file())
            self.assertEqual(len(pd.read_csv(csv_path)), 3)
            self.assertEqual(json.loads(json_path.read_text(encoding="utf-8"))["scenario_count"], 3)
