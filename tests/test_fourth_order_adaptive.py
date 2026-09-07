import unittest

import numpy as np

from experiments.run_adaptive_kl_experiment import AdaptiveExperimentConfig, run_fourth_order_case
from power_system.network import FourthOrderThreeGeneratorSystem, ThreeGeneratorSystem, make_power_system
from sppf.dynamics import StatePartitionParticleFilter
from sppf.partition import (
    AdaptiveKLConfig,
    AdaptiveKLPartitioner,
    GaussianApproximation,
    StatePartitioner,
    symmetric_gaussian_kl,
    weighted_gaussian,
)


class FourthOrderSystemTests(unittest.TestCase):
    def setUp(self) -> None:
        self.system = FourthOrderThreeGeneratorSystem()

    def test_fourth_order_state_dimension_and_dynamics(self) -> None:
        self.assertEqual(self.system.state_dimension, 12)
        state = self.system.equilibrium_state
        derivative = self.system.state_derivative(state, 0.0)
        stepped = self.system.step(state, 0.0)
        self.assertEqual(derivative.shape, (12,))
        self.assertEqual(stepped.shape, (12,))
        self.assertTrue(np.all(np.isfinite(derivative)))
        self.assertTrue(np.all(np.isfinite(stepped)))

    def test_model_order_factory_preserves_both_modes(self) -> None:
        self.assertIsInstance(make_power_system(2), ThreeGeneratorSystem)
        self.assertIsInstance(make_power_system(4), FourthOrderThreeGeneratorSystem)
        with self.assertRaises(ValueError):
            make_power_system(3)

    def test_fourth_order_measurement_uses_transient_states(self) -> None:
        state = self.system.equilibrium_state.copy()
        altered = state.copy(); altered[2] += 0.10; altered[3] += 0.05
        measurement = self.system.measurement(state)
        altered_measurement = self.system.measurement(altered)
        self.assertEqual(measurement.shape, (15,))
        self.assertTrue(np.allclose(measurement[6:12], state[[2, 6, 10, 3, 7, 11]]))
        self.assertFalse(np.allclose(measurement[12:15], altered_measurement[12:15]))


class AdaptivePartitionTests(unittest.TestCase):
    @staticmethod
    def _gaussian(mean: float) -> GaussianApproximation:
        return GaussianApproximation(np.full(5, mean), np.eye(5))

    def test_kl_is_finite_and_symmetric(self) -> None:
        samples = np.array([[0., 0.], [1., 1.], [2., 2.]])
        gaussian = weighted_gaussian(samples, np.array([0.2, 0.3, 0.5]))
        self.assertAlmostEqual(symmetric_gaussian_kl(gaussian, gaussian), 0.0, places=8)
        other = weighted_gaussian(samples + 1.0, np.array([0.2, 0.3, 0.5]))
        self.assertTrue(np.isfinite(symmetric_gaussian_kl(gaussian, other)))

    def test_merge_split_and_partition_validity(self) -> None:
        controller = AdaptiveKLPartitioner(AdaptiveKLConfig(merge_threshold=0.1, split_threshold=1.0, adaptation_interval=1, minimum_partition_lifetime=0))
        initial = StatePartitioner.fourth_order_default()
        merged, merge_event = controller.propose(initial, {0: self._gaussian(0), 1: self._gaussian(0), 2: self._gaussian(5)}, 1)
        self.assertEqual(merge_event["operation"], "merge")
        self.assertEqual(len(merged.partitions), 2)
        split, split_event = controller.propose(merged, {0: self._gaussian(0), 1: self._gaussian(10), 2: self._gaussian(5)}, 2)
        self.assertEqual(split_event["operation"], "split")
        self.assertEqual(len(split.partitions), 3)
        self.assertEqual(sorted(index for p in split.partitions for index in p.state_indices), list(range(12)))

    def test_particle_filter_repartition_and_adaptive_step(self) -> None:
        system = FourthOrderThreeGeneratorSystem()
        rng = np.random.default_rng(11)
        covariance = np.eye(12) * 1e-4
        measurement_covariance = np.eye(15) * 1e-4
        controller = AdaptiveKLPartitioner(AdaptiveKLConfig(adaptation_interval=1, minimum_partition_lifetime=0))
        filter_ = StatePartitionParticleFilter(system, StatePartitioner.fourth_order_default(), 80, system.equilibrium_state,
                                                covariance, covariance, measurement_covariance, rng, "adaptive_kl", controller)
        measurement = system.measurement(system.equilibrium_state)
        estimate, prediction, neff = filter_.step(measurement, 0.0, index=1)
        self.assertTrue(np.all(np.isfinite(estimate)))
        self.assertTrue(np.all(np.isfinite(prediction)))
        self.assertTrue(np.all((neff >= 1.0) & (neff <= 80.0)))
        for local_filter in filter_.filters:
            self.assertAlmostEqual(float(local_filter.weights.sum()), 1.0, places=10)
            self.assertTrue(np.all(np.isfinite(local_filter.particles)))

    def test_adaptive_fourth_order_fdia_end_to_end(self) -> None:
        config = AdaptiveExperimentConfig(steps=100, attack_start_index=60, attack_duration_steps=20, particle_count=100,
                                          adaptive_kl=AdaptiveKLConfig(adaptation_interval=10, minimum_partition_lifetime=20))
        result = run_fourth_order_case(config, 0.15, "adaptive_kl")
        self.assertTrue(np.all(np.isfinite(result["estimates"])))
        self.assertTrue(np.all(np.isfinite(result["scores"])))
        self.assertTrue(np.isfinite(result["threshold"]))
        self.assertIn(result["detected"], (True, False))
