# Results at a glance

## Equal-budget FDIA (five seeds)
- Fixed SP-PF lowest observed control RMSE: 0.00449 vs Full PF 0.00690 vs Adaptive-KL 0.00450.
- 15% FDIA crossover: Fixed and adaptive SP-PF 100% detection; Full PF only 40%.
- 20% FDIA: Fixed/adaptive 100% detection, Full PF 80%.
- Adaptive-KL similar detection but no observed RMSE benefit over fixed SP-PF.

## Adaptive-KL behaviour
- Mean partitions ~2.42 per run; ~0.8 merges/run; splits mostly zero.
- Starts 1050 initial particles; after a merge total falls to ~700 active particles.

## Detectors
- Residual and particle-likelihood detectors made identical FDIA detection decisions across the equal-budget matrix.
- Likelihood uses weighted particle-observation densities with log-sum-exp aggregation.
- Paper-aligned direction only; not the paper's exact H1 likelihood-ratio formulation.

## Replay validation (negative result)
- Five seeds. Blends 0-30%. 20-step delayed history.
- Residual 0% / Likelihood 0% detection across all blends. False alarms 0%.
- State RMSE unchanged.

## Ablation (different particle budgets)
- Ablation confirms the SP-PF advantage robust across setups but not the fair comparison.
- Fair comparison is always the 1050 equal-budget matrix only.

## Faculty dataset
- Workbook sheets: Stable swing / Voltage stress / load encroachement.
- 1 microsecond uniform sampling (0.001 ms). 6001-7001 samples each.
- Sampled instantaneous three-phase voltage waveforms (labelled kV). NOT PMU phasors, not generator states.
- Direct SP-PF integration requires a waveform- observation model first.
