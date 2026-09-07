# Exact numerical values for presentation claims

All values below are five-seed means aggregated from `equal_budget_results.json` and `replay_results.json`.

## Equal-budget FDIA (likelihood detector)

| Estimator | Control RMSE | 15% FDIA det. | 20% FDIA det. | 20% FDIA RMSE | Runtime ctrl (s) | Mean partitions
| --- | --- | --- | --- | --- | --- | ---
| Full PF | 0.00690 | 40% | 80% | 0.00731 | 1.281 | 1.00
| Fixed SP-PF | 0.00449 | 100% | 100% | 0.00495 | 1.724 | 3.00
| Adaptive-KL | 0.00450 | 100% | 100% | 0.00505 | 1.438 | 2.42

## Replay validation (negative result)

| Detector | Max detection rate any blend | 30% blend detection | Mean false-alarm rate
| --- | --- | --- | ---
| Residual | 0% | 0% | 0.0000
| Likelihood | 0% | 0% | 0.0000

## Configuration

- Equal-budget seeds: 20260907, 20260908, 20260909, 20260910, 20260911
- Equal-budget attack magnitudes: [0.0, 0.01, 0.02, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3]
- Total initial particles Full PF: 1050
- SP-PF particles per partition: 350
- Replay delay steps: 20 ( = 0.4 seconds at 0.02 s per sample)
