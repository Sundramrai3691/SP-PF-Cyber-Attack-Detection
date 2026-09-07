# Likelihood Detector: Paper Context vs This Prototype

## Paper direction

The reference work motivates likelihood or likelihood-ratio behaviour from
particle-filter state estimation. Its equations, system parameters, and attack
model are not numerically reproduced here.

## Implemented prototype

For each SP-PF partition and sample, the detector evaluates the full Gaussian
measurement log-density for every predicted particle:

`log p(y|x_i) = -0.5[(y-h_i)^T R^-1(y-h_i) + log|R| + m log(2π)]`.

The partition predictive likelihood is
`log L_p = log Σ_i w_prior,i exp(log p(y_p|x_i))`, evaluated with log-sum-exp.
Because the current SP-PF update is conditionally partitioned, the prototype
global likelihood is `log L = Σ_p log L_p`.

Attack evidence is a **normal-reference likelihood degradation**,
`g_k = log(L_ref / L_k) = μ0 - log L_k`, where `μ0` is the mean global
log-likelihood over attack-free calibration samples. The threshold is
`4σ0`, where `σ0` is the calibration standard deviation. This is a calibrated
one-sided likelihood-ratio-style test under H0, not the paper's exact H1
likelihood-ratio equation because this prototype has no parameterised attack
measurement distribution.

## Important limitations

- Thresholds are calibrated from this prototype's attack-free data and are
  not copied from the paper.
- The model is a fourth-order, three-generator synthetic prototype with
  approximate parameters and a simplified observation model.
- Adaptive-KL is an optional partitioning mechanism; it is not claimed to
  improve either RMSE or detection.
- Results are experimental comparisons, not an exact reproduction of paper
  figures or numerical thresholds.
