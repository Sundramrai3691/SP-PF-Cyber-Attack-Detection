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
| full_pf | residual | 0 | 5 | 0 | N/A | 0 | 0.0112396 | 0.00465871 | 0.873805 |
| full_pf | likelihood | 0 | 5 | 0 | N/A | 0 | 0.0112396 | 0.00465871 | 0.873805 |
| fixed_sppf | residual | 0 | 5 | 0 | N/A | 0 | 0.00448536 | 0.00118107 | 1.92248 |
| fixed_sppf | likelihood | 0 | 5 | 0 | N/A | 0 | 0.00448536 | 0.00118107 | 1.92248 |
| adaptive_kl | residual | 0 | 5 | 0 | N/A | 0 | 0.00450443 | 0.00116457 | 1.73262 |
| adaptive_kl | likelihood | 0 | 5 | 0 | N/A | 0 | 0.00450443 | 0.00116457 | 1.73262 |
| full_pf | residual | 0.01 | 5 | 0 | N/A | 0 | 0.0112087 | 0.00467279 | 0.876915 |
| full_pf | likelihood | 0.01 | 5 | 0 | N/A | 0 | 0.0112087 | 0.00467279 | 0.876915 |
| fixed_sppf | residual | 0.01 | 5 | 0 | N/A | 0 | 0.00448768 | 0.00117591 | 2.11212 |
| fixed_sppf | likelihood | 0.01 | 5 | 0 | N/A | 0 | 0.00448768 | 0.00117591 | 2.11212 |
| adaptive_kl | residual | 0.01 | 5 | 0 | N/A | 0 | 0.00451774 | 0.00115951 | 1.89465 |
| adaptive_kl | likelihood | 0.01 | 5 | 0 | N/A | 0 | 0.00451774 | 0.00115951 | 1.89465 |
| full_pf | residual | 0.02 | 5 | 0 | N/A | 0 | 0.0111644 | 0.00465067 | 1.03933 |
| full_pf | likelihood | 0.02 | 5 | 0 | N/A | 0 | 0.0111644 | 0.00465067 | 1.03933 |
| fixed_sppf | residual | 0.02 | 5 | 0 | N/A | 0 | 0.00449875 | 0.00117554 | 2.63231 |
| fixed_sppf | likelihood | 0.02 | 5 | 0 | N/A | 0 | 0.00449875 | 0.00117554 | 2.63231 |
| adaptive_kl | residual | 0.02 | 5 | 0 | N/A | 0 | 0.00453362 | 0.00115971 | 2.2131 |
| adaptive_kl | likelihood | 0.02 | 5 | 0 | N/A | 0 | 0.00453362 | 0.00115971 | 2.2131 |
| full_pf | residual | 0.05 | 5 | 0 | N/A | 0 | 0.0111741 | 0.00464512 | 0.905371 |
| full_pf | likelihood | 0.05 | 5 | 0 | N/A | 0 | 0.0111741 | 0.00464512 | 0.905371 |
| fixed_sppf | residual | 0.05 | 5 | 0 | N/A | 0 | 0.0045521 | 0.00115112 | 2.18889 |
| fixed_sppf | likelihood | 0.05 | 5 | 0 | N/A | 0 | 0.0045521 | 0.00115112 | 2.18889 |
| adaptive_kl | residual | 0.05 | 5 | 0 | N/A | 0 | 0.00459264 | 0.00113761 | 2.17787 |
| adaptive_kl | likelihood | 0.05 | 5 | 0 | N/A | 0 | 0.00459264 | 0.00113761 | 2.17787 |
| full_pf | residual | 0.075 | 5 | 0 | N/A | 0 | 0.0113603 | 0.00463848 | 1.03886 |
| full_pf | likelihood | 0.075 | 5 | 0 | N/A | 0 | 0.0113603 | 0.00463848 | 1.03886 |
| fixed_sppf | residual | 0.075 | 5 | 0.2 | 0 | 0 | 0.00462155 | 0.00113616 | 2.39764 |
| fixed_sppf | likelihood | 0.075 | 5 | 0.2 | 0 | 0 | 0.00462155 | 0.00113616 | 2.39764 |
| adaptive_kl | residual | 0.075 | 5 | 0.2 | 0 | 0 | 0.00467866 | 0.00112231 | 2.11636 |
| adaptive_kl | likelihood | 0.075 | 5 | 0.2 | 0 | 0 | 0.00467866 | 0.00112231 | 2.11636 |
| full_pf | residual | 0.1 | 5 | 0.2 | 1 | 0 | 0.0114641 | 0.00459658 | 0.96594 |
| full_pf | likelihood | 0.1 | 5 | 0.2 | 1 | 0 | 0.0114641 | 0.00459658 | 0.96594 |
| fixed_sppf | residual | 0.1 | 5 | 0.6 | 4 | 0 | 0.00469194 | 0.00115117 | 2.19061 |
| fixed_sppf | likelihood | 0.1 | 5 | 0.6 | 4 | 0 | 0.00469194 | 0.00115117 | 2.19061 |
| adaptive_kl | residual | 0.1 | 5 | 0.6 | 4 | 0 | 0.00476978 | 0.00104335 | 1.89067 |
| adaptive_kl | likelihood | 0.1 | 5 | 0.6 | 4 | 0 | 0.00476978 | 0.00104335 | 1.89067 |
| full_pf | residual | 0.15 | 5 | 0.4 | 0 | 0 | 0.011594 | 0.00456311 | 0.863917 |
| full_pf | likelihood | 0.15 | 5 | 0.4 | 0 | 0 | 0.011594 | 0.00456311 | 0.863917 |
| fixed_sppf | residual | 0.15 | 5 | 1 | 0.4 | 0 | 0.00481543 | 0.00107816 | 1.86592 |
| fixed_sppf | likelihood | 0.15 | 5 | 1 | 0.4 | 0 | 0.00481543 | 0.00107816 | 1.86592 |
| adaptive_kl | residual | 0.15 | 5 | 1 | 0.4 | 0 | 0.00487939 | 0.00102434 | 1.71283 |
| adaptive_kl | likelihood | 0.15 | 5 | 1 | 0.4 | 0 | 0.00487939 | 0.00102434 | 1.71283 |
| full_pf | residual | 0.2 | 5 | 0.6 | 0 | 0 | 0.0117464 | 0.00461199 | 0.756205 |
| full_pf | likelihood | 0.2 | 5 | 0.6 | 0 | 0 | 0.0117464 | 0.00461199 | 0.756205 |
| fixed_sppf | residual | 0.2 | 5 | 1 | 0 | 0 | 0.00494587 | 0.00107974 | 1.97894 |
| fixed_sppf | likelihood | 0.2 | 5 | 1 | 0 | 0 | 0.00494587 | 0.00107974 | 1.97894 |
| adaptive_kl | residual | 0.2 | 5 | 1 | 0 | 0 | 0.00504809 | 0.00100511 | 2.13784 |
| adaptive_kl | likelihood | 0.2 | 5 | 1 | 0 | 0 | 0.00504809 | 0.00100511 | 2.13784 |
| full_pf | residual | 0.3 | 5 | 1 | 0 | 0 | 0.0118571 | 0.00464309 | 0.700024 |
| full_pf | likelihood | 0.3 | 5 | 1 | 0 | 0 | 0.0118571 | 0.00464309 | 0.700024 |
| fixed_sppf | residual | 0.3 | 5 | 1 | 0 | 0 | 0.00511269 | 0.00107482 | 1.59084 |
| fixed_sppf | likelihood | 0.3 | 5 | 1 | 0 | 0 | 0.00511269 | 0.00107482 | 1.59084 |
| adaptive_kl | residual | 0.3 | 5 | 1 | 0 | 0 | 0.00522148 | 0.00100443 | 1.55724 |
| adaptive_kl | likelihood | 0.3 | 5 | 1 | 0 | 0 | 0.00522148 | 0.00100443 | 1.55724 |

## Interpretation rule

These are prototype measurements. A mode is not described as superior unless detection, RMSE, runtime, and multi-seed variation support that claim. Adaptive-KL is an optional prototype and is not assumed to improve the fixed partition.
