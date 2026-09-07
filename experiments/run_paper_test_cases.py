"""Run structured paper-inspired cases for the simplified prototype."""
from __future__ import annotations
import pandas as pd
from experiments.paper_cases import run_paper_suite


if __name__ == "__main__":
    result = run_paper_suite()
    frame = pd.DataFrame(result["rows"])
    print("PAPER-INSPIRED PROTOTYPE TEST CASES")
    print(f"Completed rows: {len(frame)}")
    print(frame[["test_case", "attack_type", "attack_intensity", "operating_condition", "detected", "detection_delay", "delta_mse", "omega_mse", "overall_mse"]].to_string(index=False))
