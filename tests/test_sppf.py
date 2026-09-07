import tempfile
import unittest
from pathlib import Path
import numpy as np

from attacks.fdia import FDIAConfig, inject_fdia
from detection.detector import ResidualDetector
from experiments.run_fdia_sppf import DemoConfig, run_experiment
from power_system.network import ThreeGeneratorSystem
from sppf.particle_filter import ParticleFilter
from sppf.partition import StatePartitioner


class ParticleFilterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.filter = ParticleFilter(100, 2, np.random.default_rng(7))
        self.filter.initialize(np.array([0.1, -0.2]), np.diag([0.01, 0.02]))

    def test_initialization_and_propagation(self) -> None:
        self.assertEqual(self.filter.particles.shape, (100, 2))
        self.assertAlmostEqual(float(self.filter.weights.sum()), 1.0, places=12)
        self.filter.predict(lambda particles: particles + 0.01, np.diag([1e-4, 1e-4]))
        self.assertTrue(np.all(np.isfinite(self.filter.particles)))

    def test_weight_normalization_neff_and_resampling(self) -> None:
        observation = np.array([0.0])
        predicted = self.filter.particles[:, :1]
        self.filter.update(observation, predicted, np.array([[0.02]]))
        self.assertAlmostEqual(float(self.filter.weights.sum()), 1.0, places=12)
        self.assertGreaterEqual(self.filter.effective_sample_size(), 1.0)
        self.assertLessEqual(self.filter.effective_sample_size(), 100.0)
        self.filter.systematic_resample()
        self.assertTrue(np.allclose(self.filter.weights, 0.01))


class ComponentTests(unittest.TestCase):
    def test_state_partitioning(self) -> None:
        partitioner = StatePartitioner.three_generator_default()
        state = partitioner.combine([np.array([1., 2.]), np.array([3., 4.]), np.array([5., 6.])])
        self.assertTrue(np.array_equal(state, np.arange(1., 7.)))

    def test_fdia_generation(self) -> None:
        values = np.ones((12, 3))
        attacked, mask, vector = inject_fdia(values, FDIAConfig(3, 8, 0.2, (1,)))
        self.assertTrue(mask[3:8].all())
        self.assertFalse(mask[:3].any())
        self.assertTrue(np.allclose(attacked[3:8, 1], 1.2))
        self.assertTrue(np.allclose(vector[:3], 0.0))

    def test_detection_statistic(self) -> None:
        detector = ResidualDetector(np.eye(2), 2.0)
        self.assertAlmostEqual(detector.score(np.array([2., 1.]), np.zeros(2)), 5.0)
        detector.calibrate(np.array([1., 2., 3.]))
        self.assertGreater(detector.threshold, 3.0)

    def test_network_dimensions(self) -> None:
        system = ThreeGeneratorSystem()
        state = np.zeros(6)
        self.assertEqual(system.transition(state, 0.0).shape, (6,))
        self.assertEqual(system.measurement(state).shape, (9,))


class EndToEndTests(unittest.TestCase):
    def test_end_to_end_demo(self) -> None:
        config = DemoConfig(steps=100, particle_count=180, attack=FDIAConfig(55, 85, 0.30, (6, 7)))
        with tempfile.TemporaryDirectory() as temporary_directory:
            result = run_experiment(config, temporary_directory, make_plots=False)
            self.assertIsNotNone(result["detection_index"])
            self.assertGreaterEqual(result["detection_index"], config.attack.start_index)
            self.assertTrue(np.all(np.isfinite(result["estimates"])))
            self.assertTrue(np.all(np.isfinite(result["scores"])))
            self.assertTrue((Path(temporary_directory) / "fdia_sppf_summary.json").is_file())
