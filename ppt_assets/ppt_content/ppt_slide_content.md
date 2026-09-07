# PPT-ready slide content

## Slide 1 - Title
**State Partition-Particle Filter Based Cyber-Physical Attack Detection in Power Systems**
Working research prototype: SP-PF estimation and FDIA/replay validation

**Speaker notes:**
- Frame this as a reproducible synthetic-model research prototype, not a paper reproduction.
- Audience should expect honest, conservative claims only.

---

## Slide 2 - Problem and objective
- Cyber-physical attacks can corrupt power-system measurements and degrade state estimation.
- Objective: estimate generator states with particle filtering and flag anomalous measurements.
- This project is a reproducible synthetic-model prototype, not an exact paper reproduction.

**Speaker notes:**
- State estimation feeds control-room decisions: bad state = bad control.
- Scope is detection of measurement attacks not grid stability only.
- Honesty about prototype level from the beginning prevents later pushback.

---

## Slide 3 - Implemented prototype
- Three-generator nonlinear simulation; second-order baseline and optional fourth-order model.
- Fixed generator-wise partitions; optional adaptive KL partitioning.
- FDIA, replay, and hybrid validation framework.
- Residual/J-statistic and particle-measurement likelihood detectors.

**Speaker notes:**
- Fourth-order = two mechanical states per generator plus two electrical.
- SP-PF splits the 12-dimensional state into three 4D partitions per machine.
- Likelihood detector = particle-observation model based, not residual based.

---

## Slide 4 - Evaluation pipeline
Use `diagrams/01_project_pipeline.png`.

**Speaker notes:**
- Walk left to right: model produces simulated truth; add Gaussian noise; inject attack type; estimate states by estimator choice; pass estimate to detector; output either residual or likelihood; collect metrics and PPT-ready evidence files.
- Important: everything downstream; detector and measurement attack is validation-only, detector decisions never seen detector is never retrains, attack- tuning attack-only.

---

## Slide 5 - Equal-particle-budget design
- Full PF: 1050 particles in one 12-state filter.
- Fixed SP-PF: 350 particles x 3 generator partitions = 1050 initial particles total.
- Adaptive-KL: begins at 1050; a merge reduces active count to 700 because each active filter retains 350 particles; merge count is ~0.8/run on average.
- Same fourth-order model, noise model noise, FDIA schedule, calibration indices, thresholds, five deterministic seeds [20260907, 20260908, 20260909, 20260910, 20260911].

**Speaker notes:**
- This is the fair-comparison baseline for all core metric comes from.
- Without the fairness slide is essential for defense: fair and credibility.
- Adaptive KL budget only approximately equal after merging reduces its own particles.
- Adaptive: before repartitioning, and slightly advantageous.

---

## Slide 6 - Detection sensitivity under fair initial budget
Use `figures/01_equal_budget_detection_rate.png`.
- Below 5% FDIA: all modes missed attacks in this experiment.
- 7.5% FDIA: Full PF 0%; Fixed SP-PF 20%; Adaptive-KL 20%.
- 10%: Full PF 0%; Fixed 60%; Adaptive-KL 60%.
- 15%: Full PF 40%; Fixed and adaptive SP-PF 100% scenario detection.
- 20%: Full PF 80%; Fixed and adaptive SP-PF 100%.
- 30% and above: all estimators 100%.

**Speaker notes:**
- Emphasize 15% FDIA crossover point.
- Fixed and adaptive SP-PF identical at every tested point point detection-wise.
- No claim SP-PF dramatically outperforms monolithic Full PF below 30%.

---

## Slide 7 - State-estimation accuracy
Use `figures/02_equal_budget_rmse.png` and `tables/01_equal_budget_key_metrics.png`.
- Control RMSE (no attack) five-seed mean):
  - Full PF: 0.00690
  - Fixed SP-PF: 0.00449
  - Adaptive-KL: 0.00450
- At 20% FDIA):
  - Full PF: RMSE 0.00731 / 80% detection
  - Fixed SP-PF: RMSE 0.00495 / 100% detection
  - Adaptive-KL: 0.00505 / 100% detection
- Fixed SP-PF has the lowest observed RMSE; Adaptive-KL did not improve it.

**Speaker notes:**
- Fixed SP-PF wins both detection and RMSE.
- Adaptive-KL very slightly trails fixed.
- No algorithm.
- Always cite five-seed means only.

---

## Slide 8 - Adaptive-KL behaviour
Use `tables/05_adaptive_kl_statistics.png` and `project_results/adaptive_kl_partition_evolution.png` if space allows.
- Mean partitions across samples: approximately 2.42.
- Mean merges per run: approximately 0.8.
- Splits: mostly zero across the matrix.
- Adaptive-KL did not show RMSE benefit.

**Speaker notes:**
- The adaptive mechanism is optional prototype only.
- Repartitioning often reduces compute (700 active particles after a merge = one generator merged) but not RMSE or better detection.
- Honest: not a claimed advance, just an optional exploration.

---

## Slide 9 - Detector comparison
Use `project_results/residual_vs_likelihood_scores.png` if available.
- Residual and particle-likelihood detectors made the identical FDIA detection decisions in the tested fourth-order equal-budget matrix (both residual and likelihood per row identical decisions).
- Likelihood uses weighted particle measurement likelihoods with stable log-sum-exp aggregation.
- This is paper-aligned evidence, not the paper's exact H1 likelihood-ratio formulation.
- See `paper_reference/likelihood_detector_formulation.md`.

**Speaker notes:**
- Two detectors produced identical FDIA decisions.
- The likelihood detector structure-direction is paper-reference paper paper, not the paper's exact H1 formulation.
- Do not claim the paper H1 likelihood ratio.

---

## Slide 10 - Replay validation (negative result)
Use `figures/04_replay_detection_rate.png`, `figures/05_replay_state_rmse.png` and `tables/04_replay_results_table.png`.
- Five seeds; 0-30% replay blend tested; 20-sample delayed synthetic history, dimensionless blend intensity.
- Residual detector: 0% detection across every blend.
- Likelihood detector: 0% detection across every blend.
- False alarms across all runs: 0%.
- RMSE stable and unchanged across blends.
- **Do not claim replay detection; the observation model/test is not sufficiently discriminative.

**Speaker notes:**
- This negative result; important negative result.
- The current replay detection; current detector thresholds are; current replay-discriminative measurement features.
- Next step must improve the measurement-assumption, not more tuning.
- Critical: claim zero. Do not over-state.

---

## Slide 11 - Faculty dataset
Use `project_results/faculty_voltage_stress_waveform.png`, `project_results/faculty_stable_swing_waveform.png`, `project_results/faculty_load_encroachement_waveform.png`, and `tables/07_faculty_dataset_summary.png`.
- Workbook `data_statedata.xlsx` sheets: Stable swing, Voltage stress, load encroachement.
- Uniform sampling: 0.001 ms (1 microsecond).
- Duration: 6-7 ms, 6001-7001 samples per sheet.
- Data behaves as sampled three-phase instantaneous voltage waveforms in labelled kV.
- NOT: NOT PMU phasor data, NOT direct generator state data.
- Physical observation model required before direct SP-PF integration.

**Speaker notes:**
- This is an external dataset. Sampling rate is 1 microsecond, 1 MHz.
- Dominant frequency ~50 kHz is high; phase RMS ~ 190-199 kV labelled unit.
- Do not say compatible; say will need a mapping from these voltage waveforms -> synthetic measurement model equivalent measurements.

---

## Slide 12 - Ablation context (optional, if time permits)
Use `project_results/ablation_detection_by_estimator.png` and `project_results/ablation_rmse_vs_magnitude.png`
- The ablation study used different particle budgets (Full PF = 350 total particles, not 1050).
- It is not the fair comparison, but confirms the SP-PF advantage is robust across setups.
- The fair comparison is always the 1050 equal-budget matrix.

**Speaker notes:**
- Ablation different unfair. Reference only when asked.
- The main result is the 1050 budget matrix.

---

## Slide 13 - Honest conclusions and next research questions
- Fixed SP-PF is the strongest observed estimator in this prototype under the tested setup.
- Adaptive-KL has not shown RMSE or detection improvement over fixed partitions.
- Likelihood detector did not improve replay detection (0% detection 0% FA).
- Faculty dataset is retained as external three-phase waveform set; observation model needed.
- **Next research decision: improve/validate the observation model and replay-discriminative measurement assumptions before claiming paper-level replay performance.**

**Speaker notes:**
- Emphasize honesty. Honesty. Honesty.
- Frame the next step as physical meaningful measurement-model improvement, not parameter tuning.
- End slide.
