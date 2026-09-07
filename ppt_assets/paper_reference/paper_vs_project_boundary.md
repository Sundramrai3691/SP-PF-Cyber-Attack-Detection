# Paper Reference Boundary

Use research-paper slides only as literature context. This project does **not** claim exact reproduction of the paper's generator parameters, network topology, H1 likelihood-ratio model, detector thresholds, adaptive partition equations and thresholds, or replay results.

All quantitative presentation claims must cite files in `project_results`, `figures`, or `tables` from this package.

## Boundary summary

| Aspect | Paper reference | This project |
| --- | --- | --- |
| Generator model | Paper's parameters | Fourth-order 3-generator synthetic prototype |
| Observation model | Paper's measurements | Approximate nonlinear measurement model |
| Likelihood ratio | Paper H1 formulation | Normal-reference degradation g_k = mu0 - log L_k |
| Partition adaptation | Paper equations | KL divergence + persistence heuristic |
| Thresholds | Paper values | 4 sigma0 from calibration samples only |
| Replay mechanism | Paper setup | 20-step delayed synthetic blend |
| Reproduction claim | Exact match expected | Honest prototype comparison only |
