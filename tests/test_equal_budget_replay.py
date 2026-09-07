"""Equal-budget accounting and replay detector-evaluation tests."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
import numpy as np

from attacks.replay import ReplayConfig, inject_replay
from experiments.run_equal_budget import EqualBudgetConfig, run_equal_budget
from experiments.run_replay_comparison import ReplayComparisonConfig, run_replay_case


class EqualBudgetReplayTests(unittest.TestCase):
    def test_replay_blend_intensity(self) -> None:
        values=np.arange(30.0).reshape(10,3)
        cfg=ReplayConfig(4,7,2,(0,1),blend=.5)
        attacked,mask,vector=inject_replay(values,cfg)
        self.assertTrue(mask[4:7].all()); self.assertFalse(mask[:4].any())
        self.assertTrue(np.allclose(attacked[4,0],.5*values[4,0]+.5*values[2,0]))
        zero,zero_mask,_=inject_replay(values,ReplayConfig(4,7,2,(0,1),blend=0.0))
        self.assertTrue(np.array_equal(zero,values)); self.assertFalse(zero_mask.any())

    def test_equal_budget_and_adaptive_history(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            cfg=EqualBudgetConfig(seeds=(1,),attack_magnitudes=(0.0,),total_particles=90,sppf_particles_per_partition=30,output_directory=str(Path(temp)/"equal"))
            result=run_equal_budget(cfg); raw=result["raw"]
            full=raw[raw.estimator_mode=="full_pf"].iloc[0]
            fixed=raw[raw.estimator_mode=="fixed_sppf"].iloc[0]
            adaptive=raw[raw.estimator_mode=="adaptive_kl"].iloc[0]
            self.assertEqual(full.particles_total_final,90); self.assertEqual(fixed.particles_total_final,90)
            self.assertEqual(np.asarray(adaptive.particle_count_history)[0],90)
            self.assertTrue((Path(cfg.output_directory)/"equal_budget_results.json").is_file())

    def test_replay_detector_comparison_rows(self) -> None:
        cfg=ReplayComparisonConfig(seeds=(1,),intensities=(0.0,0.3),output_directory="results/replay_test")
        rows=run_replay_case(cfg,1,.3)
        self.assertEqual({row["detector_type"] for row in rows},{"residual","likelihood"})
        self.assertTrue(all(np.isfinite(row["state_rmse"]) for row in rows))
        self.assertTrue(all(row["replay_delay_steps"]==20 for row in rows))


if __name__=="__main__": unittest.main()
