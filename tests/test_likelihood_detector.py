"""Tests for the particle-measurement likelihood detector extension."""

from __future__ import annotations

import unittest

import numpy as np

from detection.detector import (
    LikelihoodRatioDetector,
    compute_global_log_likelihood,
    compute_log_likelihood_ratio,
    compute_particle_log_likelihood,
    compute_weighted_particle_log_likelihood,
    logsumexp,
)
from experiments.run_likelihood_detector import LikelihoodExperimentConfig, run_fourth_order_detector_case


class LikelihoodMathTests(unittest.TestCase):
    def test_single_particle_likelihood_matches_normal_density(self) -> None:
        value = compute_particle_log_likelihood(np.array([0.0]), np.array([[0.0]]), np.array([[1.0]]))
        self.assertAlmostEqual(float(value[0]), -0.5 * np.log(2.0 * np.pi), places=11)

    def test_weighted_particle_likelihood(self) -> None:
        particle_logs = np.log(np.array([0.25, 0.75]))
        value = compute_weighted_particle_log_likelihood(particle_logs, np.array([0.5, 0.5]))
        self.assertAlmostEqual(value, np.log(0.5), places=12)

    def test_log_sum_exp_is_stable_for_extreme_underflow_values(self) -> None:
        self.assertTrue(np.isfinite(logsumexp(np.array([-1200.0, -1201.0]))))
        self.assertAlmostEqual(logsumexp(np.array([1000.0, 999.0])) - 1000.0, np.log1p(np.exp(-1.0)), places=12)

    def test_partition_aggregation_and_log_ratio(self) -> None:
        self.assertAlmostEqual(compute_global_log_likelihood(np.array([-2.0, -3.0])), -5.0)
        self.assertAlmostEqual(compute_log_likelihood_ratio(-8.0, -5.0), 3.0)

    def test_normal_calibration_and_decision(self) -> None:
        detector = LikelihoodRatioDetector(threshold_multiplier=2.0)
        threshold = detector.calibrate(np.array([-10.0, -9.0, -11.0, -10.0]))
        scores = detector.scores(np.array([-10.0, -15.0]))
        self.assertGreater(threshold, 0.0)
        self.assertFalse(detector.flags(scores)[0])
        self.assertTrue(detector.flags(scores)[1])

    def test_invalid_covariance_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            compute_particle_log_likelihood(np.array([0.0]), np.array([[0.0]]), np.array([[-1.0]]))


class LikelihoodIntegrationTests(unittest.TestCase):
    @staticmethod
    def _config() -> LikelihoodExperimentConfig:
        return LikelihoodExperimentConfig(
            steps=75, particle_count=70, calibration_start_index=10,
            attack_start_index=40, attack_duration_steps=20,
        )

    def test_attack_free_likelihood_case_is_finite_and_not_an_attack(self) -> None:
        data = run_fourth_order_detector_case(self._config(), 0.0, "fixed", "likelihood")
        self.assertFalse(data["detected"])
        self.assertTrue(np.all(np.isfinite(data["log_likelihoods"])))
        self.assertTrue(np.all(np.isfinite(data["scores"])))

    def test_fdia_likelihood_case_produces_likelihood_metrics(self) -> None:
        data = run_fourth_order_detector_case(self._config(), 0.20, "fixed", "likelihood")
        self.assertIn("mean_log_likelihood", data)
        self.assertTrue(np.isfinite(data["max_log_likelihood_ratio"]))
        self.assertEqual(data["attack_channels"], [12, 13])

    def test_adaptive_partition_compatibility(self) -> None:
        data = run_fourth_order_detector_case(self._config(), 0.20, "adaptive_kl", "likelihood")
        self.assertEqual(data["partition_mode"], "adaptive_kl")
        self.assertTrue(np.all(np.isfinite(data["log_likelihoods"])))
        self.assertTrue(np.all(data["partition_counts"] >= 1))


if __name__ == "__main__":
    unittest.main()
