# Baseline and Ablation Study

## Scope

This multi-seed study compares a genuine 12-state full PF, fixed generator-wise SP-PF, and optional adaptive-KL SP-PF on the same fourth-order synthetic FDIA setup. Both residual/J-statistic and particle-likelihood detectors are applied to the same estimator output.

## Fairness and particle budget

All runs use seeds [20260907, 20260908, 20260909, 20260910, 20260911], identical FDIA timing/channels, process/measurement covariances, calibration indices 30:100, and 350 particles per filter. Full PF therefore has 350 total particles; fixed SP-PF has 350 particles in each of three filters (1050 total). Adaptive-KL has 350 particles per active partition. Runtime and accuracy conclusions must be read with this explicitly different total particle budget.

## Evaluation window

Sample-level TP/FN are counted only during the attack interval. TN/FP are counted from the end of PF warm-up through the sample immediately before attack onset. Post-attack recovery samples are excluded from false-alarm statistics.

## Aggregate results

| estimator_mode | detector_type | attack_magnitude | seed_count | detection_rate | mean_detection_delay | mean_false_alarm_rate | mean_state_rmse | std_state_rmse | mean_runtime_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full_pf | residual | 0 | 5 | 0 | N/A | 0 | 0.0112396 | 0.00465871 | 0.676961 |
| full_pf | likelihood | 0 | 5 | 0 | N/A | 0 | 0.0112396 | 0.00465871 | 0.676961 |
| fixed_sppf | residual | 0 | 5 | 0 | N/A | 0 | 0.00448536 | 0.00118107 | 1.57038 |
| fixed_sppf | likelihood | 0 | 5 | 0 | N/A | 0 | 0.00448536 | 0.00118107 | 1.57038 |
| adaptive_kl | residual | 0 | 5 | 0 | N/A | 0 | 0.00450443 | 0.00116457 | 1.37333 |
| adaptive_kl | likelihood | 0 | 5 | 0 | N/A | 0 | 0.00450443 | 0.00116457 | 1.37333 |
| full_pf | residual | 0.01 | 5 | 0 | N/A | 0 | 0.0112087 | 0.00467279 | 0.639808 |
| full_pf | likelihood | 0.01 | 5 | 0 | N/A | 0 | 0.0112087 | 0.00467279 | 0.639808 |
| fixed_sppf | residual | 0.01 | 5 | 0 | N/A | 0 | 0.00448768 | 0.00117591 | 1.44464 |
| fixed_sppf | likelihood | 0.01 | 5 | 0 | N/A | 0 | 0.00448768 | 0.00117591 | 1.44464 |
| adaptive_kl | residual | 0.01 | 5 | 0 | N/A | 0 | 0.00451774 | 0.00115951 | 1.30359 |
| adaptive_kl | likelihood | 0.01 | 5 | 0 | N/A | 0 | 0.00451774 | 0.00115951 | 1.30359 |
| full_pf | residual | 0.02 | 5 | 0 | N/A | 0 | 0.0111644 | 0.00465067 | 0.657349 |
| full_pf | likelihood | 0.02 | 5 | 0 | N/A | 0 | 0.0111644 | 0.00465067 | 0.657349 |
| fixed_sppf | residual | 0.02 | 5 | 0 | N/A | 0 | 0.00449875 | 0.00117554 | 1.37864 |
| fixed_sppf | likelihood | 0.02 | 5 | 0 | N/A | 0 | 0.00449875 | 0.00117554 | 1.37864 |
| adaptive_kl | residual | 0.02 | 5 | 0 | N/A | 0 | 0.00453362 | 0.00115971 | 1.28527 |
| adaptive_kl | likelihood | 0.02 | 5 | 0 | N/A | 0 | 0.00453362 | 0.00115971 | 1.28527 |
| full_pf | residual | 0.05 | 5 | 0 | N/A | 0 | 0.0111741 | 0.00464512 | 0.672219 |
| full_pf | likelihood | 0.05 | 5 | 0 | N/A | 0 | 0.0111741 | 0.00464512 | 0.672219 |
| fixed_sppf | residual | 0.05 | 5 | 0 | N/A | 0 | 0.0045521 | 0.00115112 | 1.52796 |
| fixed_sppf | likelihood | 0.05 | 5 | 0 | N/A | 0 | 0.0045521 | 0.00115112 | 1.52796 |
| adaptive_kl | residual | 0.05 | 5 | 0 | N/A | 0 | 0.00459264 | 0.00113761 | 1.44974 |
| adaptive_kl | likelihood | 0.05 | 5 | 0 | N/A | 0 | 0.00459264 | 0.00113761 | 1.44974 |
| full_pf | residual | 0.075 | 5 | 0 | N/A | 0 | 0.0113603 | 0.00463848 | 0.641623 |
| full_pf | likelihood | 0.075 | 5 | 0 | N/A | 0 | 0.0113603 | 0.00463848 | 0.641623 |
| fixed_sppf | residual | 0.075 | 5 | 0.2 | 0 | 0 | 0.00462155 | 0.00113616 | 1.43628 |
| fixed_sppf | likelihood | 0.075 | 5 | 0.2 | 0 | 0 | 0.00462155 | 0.00113616 | 1.43628 |
| adaptive_kl | residual | 0.075 | 5 | 0.2 | 0 | 0 | 0.00467866 | 0.00112231 | 1.27088 |
| adaptive_kl | likelihood | 0.075 | 5 | 0.2 | 0 | 0 | 0.00467866 | 0.00112231 | 1.27088 |
| full_pf | residual | 0.1 | 5 | 0.2 | 1 | 0 | 0.0114641 | 0.00459658 | 0.644296 |
| full_pf | likelihood | 0.1 | 5 | 0.2 | 1 | 0 | 0.0114641 | 0.00459658 | 0.644296 |
| fixed_sppf | residual | 0.1 | 5 | 0.6 | 4 | 0 | 0.00469194 | 0.00115117 | 1.47827 |
| fixed_sppf | likelihood | 0.1 | 5 | 0.6 | 4 | 0 | 0.00469194 | 0.00115117 | 1.47827 |
| adaptive_kl | residual | 0.1 | 5 | 0.6 | 4 | 0 | 0.00476978 | 0.00104335 | 1.34352 |
| adaptive_kl | likelihood | 0.1 | 5 | 0.6 | 4 | 0 | 0.00476978 | 0.00104335 | 1.34352 |
| full_pf | residual | 0.15 | 5 | 0.4 | 0 | 0 | 0.011594 | 0.00456311 | 0.678057 |
| full_pf | likelihood | 0.15 | 5 | 0.4 | 0 | 0 | 0.011594 | 0.00456311 | 0.678057 |
| fixed_sppf | residual | 0.15 | 5 | 1 | 0.4 | 0 | 0.00481543 | 0.00107816 | 1.45247 |
| fixed_sppf | likelihood | 0.15 | 5 | 1 | 0.4 | 0 | 0.00481543 | 0.00107816 | 1.45247 |
| adaptive_kl | residual | 0.15 | 5 | 1 | 0.4 | 0 | 0.00487939 | 0.00102434 | 1.39986 |
| adaptive_kl | likelihood | 0.15 | 5 | 1 | 0.4 | 0 | 0.00487939 | 0.00102434 | 1.39986 |
| full_pf | residual | 0.2 | 5 | 0.6 | 0 | 0 | 0.0117464 | 0.00461199 | 0.652068 |
| full_pf | likelihood | 0.2 | 5 | 0.6 | 0 | 0 | 0.0117464 | 0.00461199 | 0.652068 |
| fixed_sppf | residual | 0.2 | 5 | 1 | 0 | 0 | 0.00494587 | 0.00107974 | 1.42609 |
| fixed_sppf | likelihood | 0.2 | 5 | 1 | 0 | 0 | 0.00494587 | 0.00107974 | 1.42609 |
| adaptive_kl | residual | 0.2 | 5 | 1 | 0 | 0 | 0.00504809 | 0.00100511 | 1.36529 |
| adaptive_kl | likelihood | 0.2 | 5 | 1 | 0 | 0 | 0.00504809 | 0.00100511 | 1.36529 |
| full_pf | residual | 0.3 | 5 | 1 | 0 | 0 | 0.0118571 | 0.00464309 | 0.642969 |
| full_pf | likelihood | 0.3 | 5 | 1 | 0 | 0 | 0.0118571 | 0.00464309 | 0.642969 |
| fixed_sppf | residual | 0.3 | 5 | 1 | 0 | 0 | 0.00511269 | 0.00107482 | 1.38153 |
| fixed_sppf | likelihood | 0.3 | 5 | 1 | 0 | 0 | 0.00511269 | 0.00107482 | 1.38153 |
| adaptive_kl | residual | 0.3 | 5 | 1 | 0 | 0 | 0.00522148 | 0.00100443 | 1.34565 |
| adaptive_kl | likelihood | 0.3 | 5 | 1 | 0 | 0 | 0.00522148 | 0.00100443 | 1.34565 |

## Interpretation rule

These are prototype measurements. A mode is not described as superior unless detection, RMSE, runtime, and multi-seed variation support that claim. Adaptive-KL is an optional prototype and is not assumed to improve the fixed partition.
