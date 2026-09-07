"""Main entry point for the reproducible SP-PF FDIA demonstration."""

from experiments.run_fdia_sppf import print_report, run_experiment


if __name__ == "__main__":
    print_report(run_experiment())
