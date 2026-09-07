# State Partition-Particle Filter (SP-PF) FDIA Detection

This B.Tech Electrical Engineering prototype demonstrates a reproducible FDIA
detection path: `3-generator swing model -> noisy measurements -> FDIA ->
fixed SP-PF -> innovation residual/J statistic -> attack flag`.

It is inspired by SP-PF research and does **not** claim full paper
reproduction, an exact likelihood-ratio test, adaptive KL partitioning, or a
fourth-order generator model.

## Run

From the repository root, using Python 3.10+:

```powershell
python -m pip install -r requirements.txt
python run_demo.py
```

The deterministic default (seed `20260907`) saves these files in `results/`:

- `01_measurement_fdia.png` — true and attacked measurement.
- `02_state_estimation.png` — six true states and SP-PF estimates.
- `03_detection_score.png` — SP-PF and simple open-loop baseline scores.
- `04_attack_flags.png` — injected attack and detector flags.
- `fdia_sppf_summary.json` — numerical results.

Run the test suite with:

```powershell
python -m unittest discover -s tests -v
```

## SP-PF structure

State: `[delta1, omega1, delta2, omega2, delta3, omega3]`, using coupled
nonlinear swing equations. Measurements are three rotor angles, three speed
deviations, and three nonlinear electrical powers.

Fixed partitions in `sppf/partition.py` are:

```
generator_1: state [0, 1], measurements [0, 3, 6]
generator_2: state [2, 3], measurements [1, 4, 7]
generator_3: state [4, 5], measurements [2, 5, 8]
```

Every partition owns a separate NumPy particle population and weights. It uses
the preceding global estimate as context for coupled generators, estimates its
local state, then local posterior means are recombined. The generic PF has
log-likelihood normalization, effective sample size, systematic resampling,
covariance regularization, and finite-value guards.

The prototype detector uses the prior innovation residual
`J = r^T R^-1 r`; its threshold is normal-data mean plus four standard
deviations. It is explicitly not represented as the paper's exact LRT. A
separate open-loop nonlinear residual detector is retained as the baseline.

## Parameters and FDIA

`DemoConfig` in `experiments/run_fdia_sppf.py` centralizes seed, sample count,
timestep, particles, initial/process/measurement covariance, threshold
multiplier and `FDIAConfig`. The default injects a 30% relative FDIA into
electric-power channels 6 and 7 at indices 120–189. A small documented floor
makes a relative attack visible for near-zero measurements.

## Layout

- `power_system/network.py` — nonlinear three-generator simulation and model.
- `attacks/fdia.py` — validated time-bounded FDIA.
- `sppf/` — generic PF, fixed partition metadata, SP-PF coordinator.
- `detection/detector.py` — residual score, calibration, alarms.
- `experiments/run_fdia_sppf.py` — full scenario, results and plots.
- `experiments/inspect_faculty_dataset.py` — separate Excel inspection tool.
- `tests/test_sppf.py` — unit and end-to-end tests.

## Faculty Excel workbook

The supplied workbook is `data_statedata.xlsx`. Its sheets contain uniformly
sampled, signed three-phase voltage time series labelled in kV, not generator
states or PMU phasors. Inspect and regenerate the dataset report/plots with:

```powershell
python -m experiments.inspect_faculty_dataset
```

This writes `results/faculty_dataset_summary.json` and scenario plots under
`results/faculty_data/`. The adapter standardizes its output as time plus
`VA/VB/VC` measurements but performs no state inference, phasor conversion, or
resampling. The faculty dataset is retained as an external measurement dataset
and requires a validated measurement/state observation model before it can be
used directly by the current SP-PF state estimator.

## Imported DDET-MTD reference code

All original DDET-MTD files are retained. Its relevant reference pieces are
IEEE-14 AC measurement/state-estimation support (`utils/class_se.py`), FDIA
utilities (`utils/fdi_att.py`), and generation script (`gen_data/gen_data.py`).
Its LSTM/autoencoder, moving-target defence, CVXPY optimisation, notebooks and
large-data workflows are intentionally isolated from the new demo.

The original scripts currently cannot run in this checkout: generated
`gen_data/case14` inputs and PyPower are absent. The SP-PF demo needs neither
PyPower, PyTorch, CVXPY, MOSEK nor external data.

## Limitations / future work

- Second-order three-machine prototype, rather than fourth-order/IEEE transient
  simulation.
- Fixed partitions only; `partition.py` provides the adaptive extension point.
- FDIA only; replay and hybrid attacks are future work.
- Validate false-alarm and detection rates over many scenarios before making
  performance claims.
- Add a faculty-data adapter only after inspection confirms signal semantics.

## Multi-Scenario FDIA Evaluation

The default 30% FDIA demo is a smoke test, not a sensitivity study. Run the
deterministic single-seed sweep with:

```powershell
python -m experiments.run_fdia_sweep
```

It evaluates one no-attack control and 5%, 10%, 15%, 20%, and 30% FDIA cases
over four valid current-model channel configurations: `[6, 7]`, `[6]`, `[7]`,
and `[3, 4]`. Channels 3--5 are speed-deviation measurements; channels 6--8
are electrical-power measurements. The threshold is calibrated once from the
attack-free control only, so attacked samples never enter calibration.

Results are written to `results/fdia_sweep_results.csv`,
`results/fdia_sweep_results.json`, and `results/sweep/`. Per scenario the
report stores detection status/delay, threshold, maximum score, state RMSE,
and pre-attack false-alarm count/rate. These results evaluate this simplified
fixed-partition synthetic SP-PF prototype only; they do not establish the
paper's exact SP-PF method or superiority over another detector.
