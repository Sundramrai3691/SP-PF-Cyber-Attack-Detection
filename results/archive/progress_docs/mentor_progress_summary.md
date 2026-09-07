# SP-PF Cyber-Physical Attack Detection: Mentor Progress Summary

## PROJECT OBJECTIVE

Build and validate a working initial State Partition-Particle Filter (SP-PF) prototype for FDIA detection in a controlled nonlinear power-system simulation.

## CURRENT IMPLEMENTATION

- SP-PF means **State Partition-Particle Filter**: the six-dimensional state is split into three fixed two-state generator partitions, and each partition uses its own particle population before local estimates are recombined.
- Three-generator, second-order nonlinear swing-equation prototype.
- State: `[delta1, omega1, delta2, omega2, delta3, omega3]`.
- Fixed partitions: `[0,1]`, `[2,3]`, `[4,5]`; 600 particles per partition.
- FDIA injection and an innovation residual/J-statistic detector with a threshold calibrated from attack-free data only.
- Reproducible 21-scenario FDIA sweep and separate faculty-workbook inspection/adapter.

## CURRENT RESULTS

Representative FDIA: 10% on channels [6, 7].

- Detected: YES.
- Detection delay: 0 samples (0.000 s).
- Fixed threshold: 26.144; maximum score: 172.401.
- State RMSE: 0.002755.

Sweep detection rates:
- 0%: 0% detection across configured channels
- 5%: 50% detection across configured channels
- 10%: 75% detection across configured channels
- 15%: 100% detection across configured channels
- 20%: 100% detection across configured channels
- 30%: 100% detection across configured channels

## FACULTY DATASET FINDINGS

- Workbook: `data_statedata.xlsx`; sheets: Stable swing, Voltage stress, load encroachement.
- Sampling is uniform at 0.001 ms (1 microsecond).
- Numerical evidence supports sampled instantaneous three-phase voltage waveforms in labelled kV units, not PMU phasors and not direct generator states.
- Direct compatibility with the current estimator: **No**. A validated waveform-to-measurement/state observation model is required first.

## CURRENT LIMITATIONS

- Working prototype only: not an exact reproduction of the paper or a production detector.
- Second-order generator dynamics, fixed partitions, residual/J-statistic detector, synthetic nonlinear simulation, and FDIA only.
- No adaptive KL partitioning, replay/hybrid attacks, fourth-order machine model, or paper-exact likelihood-ratio detector.
- Faculty voltage waveforms are not currently mapped to the prototype state-space model.

## NEXT TECHNICAL OPTIONS

### Option A - Continue entirely in Python (default recommendation)

The current implementation already works, is reproducible with NumPy/SciPy, and processes Excel data directly. It avoids an environment switch while the algorithm and its validation mature.

### Option B - MATLAB/MATPOWER simulation with Python SP-PF

This can provide a more established power-system simulation source while retaining the working Python estimator. It requires a defined MATLAB/Python interface, conversion of measurements and states, plus setup and debugging of the cross-tool workflow.

### Option C - Move the entire implementation to MATLAB

This entails substantial reimplementation. It should be chosen only if MATLAB is specifically required by the project or mentor, not merely for presentation.

## QUESTIONS FOR MENTOR

1. Should we continue with the current Python-based implementation, or is MATLAB/MATPOWER expected for the final project?
2. Does the faculty dataset need to become the primary experimental dataset, or can it remain a separate validation/input dataset while the algorithm is developed on a controlled simulation?
3. Should we target the paper's full fourth-order generator model?
4. Should adaptive KL-based partitioning be the next implementation milestone?
5. Does Ma'am expect an exact reproduction of the paper's likelihood-ratio detection method, or is a validated prototype acceptable initially?
6. Are there additional system measurements/model parameters available for the faculty dataset so that a physically meaningful state/measurement model can be constructed?
