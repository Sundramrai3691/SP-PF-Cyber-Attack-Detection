"""Build a presentation-only package from verified result artifacts.

This utility reads existing outputs only. It never runs, tunes, or changes an
algorithm or experiment. Run ``python -m experiments.build_ppt_assets``.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("ppt_assets")
FIGURES = ROOT / "figures"; TABLES = ROOT / "tables"; DIAGRAMS = ROOT / "diagrams"
PAPER = ROOT / "paper_reference"; PROJECT = ROOT / "project_results"
SUMMARIES = ROOT / "summaries"; CONTENT = ROOT / "ppt_content"


def _setup() -> None:
    for directory in (FIGURES, TABLES, DIAGRAMS, PAPER, PROJECT, SUMMARIES, CONTENT):
        directory.mkdir(parents=True, exist_ok=True)
    cache = ROOT / ".matplotlib"
    cache.mkdir(exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache.resolve())


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _save_table(frame: pd.DataFrame, title: str, filename: str, note: str = "",
                 figsize: tuple[int, int] | None = None) -> None:
    plt = _plt()
    if figsize is None:
        figsize = (14, 0.55 * (len(frame) + 3))
    fig, axis = plt.subplots(figsize=figsize)
    axis.axis("off")
    rendered = frame.copy()
    for column in rendered.columns:
        rendered[column] = rendered[column].map(
            lambda value: f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value)
        )
    table = axis.table(
        cellText=rendered.values, colLabels=rendered.columns, cellLoc="center", loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.65)
    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#17365D")
            cell.set_text_props(color="white", weight="bold")
        elif row % 2:
            cell.set_facecolor("#EAF2F8")
    fig.suptitle(title, fontsize=17, weight="bold", y=.98)
    if note:
        fig.text(.5, .02, note, ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .93))
    fig.savefig(TABLES / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_equal(summary: pd.DataFrame) -> None:
    plt = _plt()
    names = {"full_pf": "Full PF (1050)", "fixed_sppf": "Fixed SP-PF (3 x 350)", "adaptive_kl": "Adaptive-KL SP-PF"}
    colors = {"full_pf": "#7F7F7F", "fixed_sppf": "#0070C0", "adaptive_kl": "#ED7D31"}

    for metric, ylabel, filename, title, ylim in (
        ("detection_rate", "Detection rate", "01_equal_budget_detection_rate.png",
        "Equal initial particle budget: FDIA detection", (-.04, 1.05)),
        ("mean_state_rmse", "Mean state RMSE", "02_equal_budget_rmse.png",
        "Equal initial particle budget: state RMSE", (0, None)),
        ("mean_detection_delay", "Mean detection delay (samples)", "03_equal_budget_detection_delay.png",
        "Equal initial particle budget: detection delay (detected cases only", (0, None)),
    ):
        fig, axis = plt.subplots(figsize=(10, 6))
        data = summary[summary.detector_type == "likelihood"]
        for mode in ("full_pf", "fixed_sppf", "adaptive_kl"):
            row = data[data.estimator_mode == mode].sort_values("attack_magnitude")
            if metric == "mean_detection_delay":
                valid = row.dropna(subset=[metric])
            else:
                valid = row
            axis.plot(valid.attack_magnitude * 100, valid[metric],
                       marker="o", lw=2.5, ms=7, label=names[mode], color=colors[mode])
        axis.set(xlabel="FDIA magnitude (%)", ylabel=ylabel, title=title)
        axis.grid(alpha=.25)
        axis.legend()
        axis.set_ylim(bottom=ylim[0])
        if ylim[1] is not None:
            axis.set_ylim(top=ylim[1])
        fig.tight_layout()
        fig.savefig(FIGURES / filename, dpi=300)
        plt.close(fig)

    control = summary[(summary.detector_type == "likelihood") & (summary.attack_magnitude.isin([0.0, .20]))].copy()
    pivot = control.pivot(
        index="estimator_mode", columns="attack_magnitude", values="mean_runtime_seconds"
    ).reset_index()
    _save_table(
        pivot.rename(columns={
            "estimator_mode": "Estimator",
            "0.0": "Runtime control (s)",
            "0.2": "Runtime 20% FDIA (s)"
        }),
        "Runtime context: equal initial budget",
        "03_runtime_context.png",
        "Full PF: 1050 particles. Fixed SP-PF: 3 x 350 particles. Adaptive-KL may fall to 700 active particles after a merge."
    )


def _plot_replay(summary: pd.DataFrame) -> None:
    plt = _plt()
    fig, axis = plt.subplots(figsize=(10, 6))
    for detector, color in (("residual", "#4472C4"), ("likelihood", "#ED7D31")):
        row = summary[summary.detector_type == detector].sort_values("replay_intensity")
        axis.plot(row.replay_intensity * 100, row.detection_rate,
                   marker="o", lw=2.5, ms=7, label=detector.title(), color=color)
    axis.set(xlabel="Replay blend intensity (%)", ylabel="Detection rate",
             ylim=(-.04, 1.05),
             title="Replay validation: residual vs likelihood detector")
    axis.grid(alpha=.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "04_replay_detection_rate.png", dpi=300)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 6))
    for detector, color in (("residual", "#4472C4"), ("likelihood", "#ED7D31")):
        row = summary[summary.detector_type == detector].sort_values("replay_intensity")
        axis.plot(row.replay_intensity * 100, row.mean_state_rmse,
                   marker="s", lw=2.5, ms=7, label=f"{detector.title()} RMSE", color=color)
    axis.set(xlabel="Replay blend intensity (%)", ylabel="State RMSE",
             title="Replay validation: state RMSE by detector")
    axis.grid(alpha=.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "05_replay_state_rmse.png", dpi=300)
    plt.close(fig)

    table = summary[summary.replay_intensity.isin([0.0, .05, .10, .20, .30])][[
        "detector_type", "replay_intensity", "detection_rate",
        "mean_false_alarm_rate", "mean_state_rmse", "mean_threshold"
    ]]
    _save_table(
        table.rename(columns={
            "detector_type": "Detector",
            "replay_intensity": "Blend",
            "detection_rate": "Detection rate",
            "mean_false_alarm_rate": "False alarm rate",
            "mean_state_rmse": "State RMSE",
            "mean_threshold": "Mean threshold"
        }),
        "Replay validation: five deterministic seeds",
        "04_replay_results_table.png",
        "Replay blend = dimensionless interpolation: 0 = present measurement, 1 = full delayed replay. Not a physical power percentage."
    )


def _diagram() -> None:
    plt = _plt()
    from matplotlib.patches import FancyBboxPatch
    fig, axis = plt.subplots(figsize=(15, 5))
    axis.set_xlim(0, 15)
    axis.set_ylim(0, 5)
    axis.axis("off")
    labels = [
        ("Fourth-order\n3-generator model", .5, "#D9EAF7"),
        ("Measurements\n+ noise", 3.1, "#D9EAF7"),
        ("FDIA / replay\n(validation)", 5.7, "#FCE4D6"),
        ("Full PF or\nSP-PF", 8.3, "#E2F0D9"),
        ("Residual or\nlikelihood detector", 10.9, "#FFF2CC"),
        ("Alarm / metrics\n/ PPT evidence", 13.1, "#D9EAD3")
    ]
    for label, x, color in labels:
        axis.add_patch(FancyBboxPatch(
            (x, .95), 1.45, 2.5,
            boxstyle="round,pad=.08",
            facecolor=color, edgecolor="#17365D", lw=1.7
        ))
        axis.text(x + .725, 2.2, label, ha="center", va="center", fontsize=12, weight="bold")
    for x in [1.95, 4.55, 7.15, 9.75, 12.35]:
        axis.annotate("", xy=(x + .9, 2.2), xytext=(x, 2.2),
                     arrowprops=dict(arrowstyle="->", lw=2, color="#17365D"))
    axis.set_title("Project evaluation pipeline (implemented prototype)",
                   fontsize=18, weight="bold")
    fig.tight_layout()
    fig.savefig(DIAGRAMS / "01_project_pipeline.png", dpi=300)
    plt.close(fig)


def _equal_budget_tables(eq: pd.DataFrame) -> None:
    data = eq[eq.detector_type == "likelihood"].copy()
    data["FDIA (%)"] = (data["attack_magnitude"] * 100).astype(float)
    data.loc[data["attack_magnitude"] == 0.075, "FDIA (%)"] = 7.5

    det = data[["estimator_mode", "FDIA (%)", "detection_rate",
               "mean_detection_delay", "mean_false_alarm_rate"]].copy()
    det = det.rename(columns={
        "estimator_mode": "Estimator",
        "detection_rate": "Detection rate",
        "mean_detection_delay": "Delay (samples)",
        "mean_false_alarm_rate": "False alarm rate",
    })
    det["Estimator"] = det["Estimator"].map({
        "full_pf": "Full PF", "fixed_sppf": "Fixed SP-PF", "adaptive_kl": "Adaptive-KL"
    })
    _save_table(
        det,
        "Equal-budget FDIA detection matrix (five seeds, likelihood detector)",
        "02_equal_budget_full_detection_matrix.png",
        "Identical decisions were produced by the residual detector on this matrix. Detector thresholds are calibrated from attack-free calibration samples only."
    )

    rmse = data[["estimator_mode", "FDIA (%)", "mean_state_rmse",
                  "mean_runtime_seconds", "mean_num_partitions"]].copy()
    rmse = rmse.rename(columns={
        "estimator_mode": "Estimator",
        "mean_state_rmse": "State RMSE",
        "mean_runtime_seconds": "Runtime (s)",
        "mean_num_partitions": "Mean partitions"
    })
    rmse["Estimator"] = rmse["Estimator"].map({
        "full_pf": "Full PF", "fixed_sppf": "Fixed SP-PF", "adaptive_kl": "Adaptive-KL"
    })
    _save_table(
        rmse,
        "Equal-budget accuracy and runtime matrix (five seeds)",
        "06_equal_budget_accuracy_matrix.png",
        "Adaptive-KL mean partitions ~2.42; mean merge count ~0.8 per run; splits mostly zero."
    )


def _adaptive_kl_table(equal_df: pd.DataFrame) -> None:
    data = equal_df[
        (equal_df.detector_type == "likelihood") &
        (equal_df.estimator_mode == "adaptive_kl")
    ].copy()
    data["FDIA (%)"] = (data["attack_magnitude"] * 100).astype(float)
    data.loc[data["attack_magnitude"] == 0.075, "FDIA (%)"] = 7.5
    stats = data[["FDIA (%)", "mean_num_partitions", "mean_runtime_seconds",
                 "mean_state_rmse", "detection_rate"]].copy()
    stats = stats.rename(columns={
        "mean_num_partitions": "Mean partitions",
        "mean_runtime_seconds": "Runtime (s)",
        "mean_state_rmse": "State RMSE",
        "detection_rate": "Detection rate"
    })
    _save_table(
        stats,
        "Adaptive-KL SP-PF per-magnitude statistics",
        "05_adaptive_kl_statistics.png",
        "Starts at 3 partitions x 350 particles. After a merge the active count is 2 x 350 = 700 active particles."
    )


def _faculty_table() -> None:
    faculty = json.loads(Path("results/faculty_dataset_summary.json").read_text(encoding="utf-8"))
    rows = []
    for sheet in faculty["workbooks"][0]["sheets"]:
        phase_rms = []
        for key, val in sheet["signal_ranges"].items():
            phase_rms.append(val["rms"])
        first_sig = list(sheet["signal_ranges"].values())[0]
        rows.append({
            "Sheet": sheet["sheet"],
            "Samples": sheet["rows"],
            "Duration (ms)": sheet["time"]["time_end"],
            "dt (us)": round(sheet["time"]["mean_dt"] * 1000, 3),
            "Dominant freq (Hz)": int(round(first_sig["dominant_frequency_hz"], -2)),
            "Mean phase RMS (kV)": round(sum(phase_rms) / len(phase_rms), 2),
            "Max |V| (kV)": round(max(v["max"] for v in sheet["signal_ranges"].values()), 2)
        })
    frame = pd.DataFrame(rows)
    _save_table(
        frame,
        "Faculty dataset: sheet-level summary",
        "07_faculty_dataset_summary.png",
        "Uniform 1 microsecond sampling. Three-phase instantaneous voltage waveforms in labelled kV. Not PMU phasors or generator states.",
        figsize=(16, 3.5)
    )


def _copy_project_figures() -> None:
    mapping = {
        "results/adaptive_kl/partitions_vs_time.png":
            "adaptive_kl_partition_evolution.png",
        "results/adaptive_kl/fourth_order_state_estimation.png":
            "fourth_order_state_estimation.png",
        "results/adaptive_kl/fixed_vs_adaptive_state_rmse.png":
            "fixed_vs_adaptive_rmse_comparison.png",
        "results/equal_budget/residual_vs_likelihood.png":
            "residual_vs_likelihood_scores.png",
        "results/03_detection_score.png":
            "baseline_detection_score.png",
        "results/04_attack_flags.png":
            "baseline_attack_flags.png",
        "results/replay_comparison/representative_replay_measurement.png":
            "replay_representative_measurement.png",
        "results/faculty_data/stable_swing_voltage.png":
            "faculty_stable_swing_waveform.png",
        "results/faculty_data/load_encroachement_voltage.png":
            "faculty_load_encroachement_waveform.png",
        "results/ablation/detection_rate_by_estimator.png":
            "ablation_detection_by_estimator.png",
        "results/ablation/state_rmse_vs_magnitude.png":
            "ablation_rmse_vs_magnitude.png",
        "results/equal_budget/detection_delay_vs_magnitude.png":
            "equal_budget_detection_delay_context.png",
        "results/paper_test_cases/rotor_angle_estimation.png":
            "paper_context_rotor_angle.png",
        "results/paper_test_cases/rotor_speed_estimation.png":
            "paper_context_rotor_speed.png",
    }
    for src, dst in mapping.items():
        sp = Path(src)
        if sp.exists():
            shutil.copy2(sp, PROJECT / dst)


def _copy_paper_context() -> None:
    (PAPER / "likelihood_detector_formulation.md").write_text("""# Likelihood Detector: Paper Context vs This Prototype

## Paper direction

The reference work motivates likelihood or likelihood-ratio behaviour from
particle-filter state estimation. Its equations, system parameters, and attack
model are not numerically reproduced here.

## Implemented prototype

For each SP-PF partition and sample, the detector evaluates the full Gaussian
measurement log-density for every predicted particle:

`log p(y|x_i) = -0.5[(y-h_i)^T R^-1(y-h_i) + log|R| + m log(2pi)]`.

The partition predictive likelihood is
`log L_p = log sum_i w_prior,i exp(log p(y_p|x_i))`, evaluated with log-sum-exp.
Because the current SP-PF update is conditionally partitioned, the prototype
global likelihood is `log L = sum_p log L_p`.

Attack evidence is a **normal-reference likelihood degradation**,
`g_k = log(L_ref / L_k) = mu0 - log L_k`, where `mu0` is the mean global
log-likelihood over attack-free calibration samples. The threshold is
`4 sigma0`, where `sigma0` is the calibration standard deviation. This is a calibrated
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
""", encoding="utf-8")

    (PAPER / "replay_detection_context.md").write_text("""# Replay Detection: Paper Context vs This Prototype

The paper motivates replay detection through likelihood-ratio behaviour under its own system and observation model. This prototype uses delayed synthetic noisy measurements (20-sample window) and a normal-reference particle likelihood degradation with thresholds originally calibrated for FDIA only.

- No threshold was retuned for replay.
- Intensity is a dimensionless blend in [0,1] between the current measurement (0) and delayed history (1).
- Under the tested matrix both detectors produced 0% detection at every tested blend across five seeds. False-alarm rate was also 0%.
- Therefore this prototype does not claim replay detection.
""", encoding="utf-8")

    (PAPER / "adaptive_partitioning_context.md").write_text("""# Adaptive Partitioning: Paper Context vs This Prototype

The reference paper derives adaptive state-partition equations from information-theoretic and physical considerations.

## Prototype implementation

- Partition starts with three generator-wise fixed partitions of the 12-state fourth-order model (each partition covers one machine).
- KL-divergence monitors partition similarity; a merge combines two partitions when their KL divergence stays below threshold over a persistence window.
- Splits are reserved for high-divergence cases but were rarely triggered in the tested matrix.
- Each active partition retains 350 particles. Merging two partitions does not increase particles; it simply reduces the total active count.
- Observed behaviour across five-seed equal-budget run:
  - Mean partitions across samples: approximately 2.42
  - Mean merges per run: ~0.8
  - Splits: mostly zero
- No RMSE or detection-rate advantage was observed over the fixed 3-partition baseline in this matrix.
""", encoding="utf-8")

    (PAPER / "paper_vs_project_boundary.md").write_text("""# Paper Reference Boundary

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
""", encoding="utf-8")


def _slide_content_with_notes() -> None:
    (CONTENT / "ppt_slide_content.md").write_text("""# PPT-ready slide content

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
""", encoding="utf-8")

    (CONTENT / "slide_cheat_sheet.csv").write_text("""slide_number,asset_type,asset_path,say
1,title,,State Partition-Particle Filter Based Cyber-Physical Attack Detection in Power Systems
2,bullets,,Problem / objective / prototype-level honesty
3,bullets,,Implemented prototype components list
4,image,diagrams/01_project_pipeline.png,Pipeline walk-through
5,bullets,,Equal-budget design fairness explanation
6,image,figures/01_equal_budget_detection_rate.png,Detection sensitivity plot; 15% crossover
7,image+table,figures/02_equal_budget_rmse.png tables/01_equal_budget_key_metrics.png,RMSE numbers + key metrics
8,table,tables/05_adaptive_kl_statistics.png,Adaptive-KL partition stats
9,bullets+image,project_results/residual_vs_likelihood_scores.png,Detector comparison + residual likelihood parity
10,image+table,figures/04_replay_detection_rate.png tables/04_replay_results_table.png,Replay negative result (0%)
11,image+table,project_results/faculty_voltage_stress_waveform.png tables/07_faculty_dataset_summary.png,Faculty dataset waveform + NOT PMU caveat
12,optional,project_results/ablation_detection_by_estimator.png,Ablation context note: Full PF = 350, not equal budget
13,bullets,,Honest conclusions + next-step observation-model research decision
""", encoding="utf-8")


def _numerical_summaries(eq_df: pd.DataFrame, rp_df: pd.DataFrame, equal_raw: dict, replay_raw: dict) -> None:
    eq_all = eq_df.copy()
    eq_all["attack_magnitude_pct"] = eq_all["attack_magnitude"] * 100
    eq_all.to_csv(SUMMARIES / "equal_budget_aggregate.csv", index=False, float_format="%.6f")

    rp_all = rp_df.copy()
    rp_all["replay_intensity_pct"] = rp_all["replay_intensity"] * 100
    rp_all.to_csv(SUMMARIES / "replay_comparison_aggregate.csv", index=False, float_format="%.6f")

    # Exact numerical fingerprint for presentation claims
    fingerprint = {}
    likelihood = eq_df[eq_df.detector_type == "likelihood"]
    for mode in ["full_pf", "fixed_sppf", "adaptive_kl"]:
        sub = likelihood[likelihood.estimator_mode == mode]
        ctrl = sub[sub.attack_magnitude == 0.0].iloc[0]
        at20 = sub[sub.attack_magnitude == 0.20].iloc[0]
        at15 = sub[sub.attack_magnitude == 0.15].iloc[0]
        fingerprint[mode] = {
            "control_rmse": float(ctrl["mean_state_rmse"]),
            "at_15pct_detection": float(at15["detection_rate"]),
            "at_20pct_detection": float(at20["detection_rate"]),
            "at_20pct_rmse": float(at20["mean_state_rmse"]),
            "mean_runtime_control": float(ctrl["mean_runtime_seconds"]),
            "mean_partitions": float(ctrl["mean_num_partitions"]),
        }
    replay_fingerprint = {}
    for det in ["residual", "likelihood"]:
        sub = rp_df[rp_df.detector_type == det]
        replay_fingerprint[det] = {
            "max_detection_rate_any_blend": float(sub["detection_rate"].max()),
            "mean_false_alarm_rate_all": float(sub["mean_false_alarm_rate"].mean()),
            "at_30pct_blend_detection": float(sub[sub.replay_intensity == 0.30].iloc[0]["detection_rate"]),
        }
    fingerprint["replay"] = replay_fingerprint
    fingerprint["config"] = {
        "equal_budget_seeds": equal_raw["configuration"]["seeds"],
        "equal_budget_magnitudes": equal_raw["configuration"]["attack_magnitudes"],
        "total_particles_equal": equal_raw["configuration"]["total_particles"],
        "sppf_per_partition": equal_raw["configuration"]["sppf_particles_per_partition"],
        "replay_seeds": replay_raw["configuration"]["seeds"],
        "replay_delay_steps": replay_raw["configuration"]["replay_delay_steps"],
    }
    (SUMMARIES / "presentation_numerical_fingerprint.json").write_text(
        json.dumps(fingerprint, indent=2), encoding="utf-8"
    )

    (SUMMARIES / "exact_claim_values.md").write_text(f"""# Exact numerical values for presentation claims

All values below are five-seed means aggregated from `equal_budget_results.json` and `replay_results.json`.

## Equal-budget FDIA (likelihood detector)

| Estimator | Control RMSE | 15% FDIA det. | 20% FDIA det. | 20% FDIA RMSE | Runtime ctrl (s) | Mean partitions
| --- | --- | --- | --- | --- | --- | ---
| Full PF | {fingerprint['full_pf']['control_rmse']:.5f} | {fingerprint['full_pf']['at_15pct_detection']:.0%} | {fingerprint['full_pf']['at_20pct_detection']:.0%} | {fingerprint['full_pf']['at_20pct_rmse']:.5f} | {fingerprint['full_pf']['mean_runtime_control']:.3f} | {fingerprint['full_pf']['mean_partitions']:.2f}
| Fixed SP-PF | {fingerprint['fixed_sppf']['control_rmse']:.5f} | {fingerprint['fixed_sppf']['at_15pct_detection']:.0%} | {fingerprint['fixed_sppf']['at_20pct_detection']:.0%} | {fingerprint['fixed_sppf']['at_20pct_rmse']:.5f} | {fingerprint['fixed_sppf']['mean_runtime_control']:.3f} | {fingerprint['fixed_sppf']['mean_partitions']:.2f}
| Adaptive-KL | {fingerprint['adaptive_kl']['control_rmse']:.5f} | {fingerprint['adaptive_kl']['at_15pct_detection']:.0%} | {fingerprint['adaptive_kl']['at_20pct_detection']:.0%} | {fingerprint['adaptive_kl']['at_20pct_rmse']:.5f} | {fingerprint['adaptive_kl']['mean_runtime_control']:.3f} | {fingerprint['adaptive_kl']['mean_partitions']:.2f}

## Replay validation (negative result)

| Detector | Max detection rate any blend | 30% blend detection | Mean false-alarm rate
| --- | --- | --- | ---
| Residual | {replay_fingerprint['residual']['max_detection_rate_any_blend']:.0%} | {replay_fingerprint['residual']['at_30pct_blend_detection']:.0%} | {replay_fingerprint['residual']['mean_false_alarm_rate_all']:.4f}
| Likelihood | {replay_fingerprint['likelihood']['max_detection_rate_any_blend']:.0%} | {replay_fingerprint['likelihood']['at_30pct_blend_detection']:.0%} | {replay_fingerprint['likelihood']['mean_false_alarm_rate_all']:.4f}

## Configuration

- Equal-budget seeds: {', '.join(str(s) for s in fingerprint['config']['equal_budget_seeds'])}
- Equal-budget attack magnitudes: {fingerprint['config']['equal_budget_magnitudes']}
- Total initial particles Full PF: {fingerprint['config']['total_particles_equal']}
- SP-PF particles per partition: {fingerprint['config']['sppf_per_partition']}
- Replay delay steps: {fingerprint['config']['replay_delay_steps']} ( = 0.4 seconds at 0.02 s per sample)
""", encoding="utf-8")


def _update_archive_readme() -> None:
    Path("results/archive/README.md").write_text("""# Archived non-final / intermediate artifacts

## Archived contents

### early_sweep/
- sweep/: single-seed initial FDIA sweep before standardised multi-seed equal-budget experiment superseded by results/equal_budget/. This was an earlier configuration (600 particles, one seed only).
- fdia_sweep_results.json: JSON source data for early sweep.
- fdia_sppf_summary.json: single-seed single-run summary.
- mentor_demo_summary.png: early demo figure.

### progress_docs/
- mentor_progress_summary.md: earlier mentor check-in document with earlier project-level options and questions, not final presentation evidence.

### matplotlib_caches/
- Runtime fontlist-v3.11.0.json files copied out of result-subfolders .matplotlib folders (reproduced runtime only).

All experimental source result data (JSON in equal_budget, adaptive_kl, replay_comparison, ablation, paper_test_cases, likelihood_detector, faculty_data and results root JSON) remain in their original locations.

*No experimental evidence was deleted or altered.* The PPT-ready package is ppt_assets/.

## Archive manifest (moves performed 2025-03-18)

| Source | Destination | Reason |
| --- | --- | --- |
| results/sweep/ (directory) | results/archive/early_sweep/sweep/ | Pre-equal-budget single-seed sweep; superseded |
| results/fdia_sweep_results.json | results/archive/early_sweep/fdia_sweep_results.json | Pre-equal-budget sweep data |
| results/fdia_sppf_summary.json | results/archive/early_sweep/fdia_sppf_summary.json | Pre-equal-budget single-run summary |
| results/mentor_demo_summary.png | results/archive/early_sweep/mentor_demo_summary.png | Early demo summary figure |
| results/mentor_progress_summary.md | results/archive/progress_docs/mentor_progress_summary.md | Earlier mentor check-in / superseded |
| results/*/.matplotlib/fontlist-v3.11.0.json | results/archive/matplotlib_caches/ | Runtime-only font cache artifacts |
""", encoding="utf-8")

    (SUMMARIES / "archive_manifest.md").write_text("""# PPT Package Archive Manifest

This manifest records which earlier outputs were relocated to `results/archive/`. No file was permanently deleted.

## Relocation performed 2025-03-18

| Source path | Archive path | Rationale |
| --- | --- | --- |
| results/sweep/ (directory tree) | results/archive/early_sweep/sweep/ | 21-scenario single-seed pre-equal-budget FDIA sweep (600 particles, seed 20260907 only). Superseded by results/equal_budget/ five-seed equal-budget matrix. |
| results/fdia_sweep_results.json | results/archive/early_sweep/fdia_sweep_results.json | Aggregate data for the early sweep. Superseded. |
| results/fdia_sppf_summary.json | results/archive/early_sweep/fdia_sppf_summary.json | Single-run single-seed summary. Superseded. |
| results/mentor_demo_summary.png | results/archive/early_sweep/mentor_demo_summary.png | Composite early-demo figure. Superseded by figures in ppt_assets/. |
| results/mentor_progress_summary.md | results/archive/progress_docs/mentor_progress_summary.md | Earlier mentor check-in document. Superseded by final presentation materials. |

## Runtime-only artifacts not part of scientific evidence

Matplotlib fontlist caches generated at render time are stored under:
`results/archive/matplotlib_caches/`. They can be regenerated on any machine and contain no results.

## Retained in place

All result folders with aggregate JSON used by this presentation package remain at their original locations:
- results/equal_budget/ (fair comparison core dataset)
- results/adaptive_kl/
- results/replay_comparison/ (replay negative result)
- results/ablation/ (different particle-budget regime; linked from paper_reference as not-fair)
- results/paper_test_cases/ (rotor angle/speed reference waveforms)
- results/likelihood_detector/ (detector formulation notes)
- results/faculty_data/ (faculty waveform plots)
- Root-level results JSONs and PNGs (01_measurement_fdia, 02_state_estimation, 03_detection_score, 04_attack_flags, fdia_grid_simulation, etc.)
""", encoding="utf-8")

    try:
        ablation = pd.read_csv(Path("results/ablation/summary.md"), sep="|", skiprows=1, header=0, skipinitialspace=True, on_bad_lines="skip")
        # summary.md is not actually CSV; just copy a lightweight JSON if present
        ablation_json = Path("results/ablation/ablation_results.json")
        if ablation_json.exists():
            shutil.copy2(ablation_json, SUMMARIES / "ablation_results_reference.json")
    except Exception:
        pass


def build() -> None:
    _setup()
    equal = json.loads(Path("results/equal_budget/equal_budget_results.json").read_text(encoding="utf-8"))
    eq = pd.DataFrame(equal["aggregate_results"])
    replay = json.loads(Path("results/replay_comparison/replay_results.json").read_text(encoding="utf-8"))
    rp = pd.DataFrame(replay["aggregate_results"])

    metrics = eq[(eq.detector_type == "likelihood") & eq.attack_magnitude.isin([0.0, .20])][[
        "estimator_mode", "attack_magnitude", "detection_rate",
        "mean_state_rmse", "mean_runtime_seconds", "mean_num_partitions"
    ]]
    _save_table(
        metrics.rename(columns={
            "estimator_mode": "Estimator",
            "attack_magnitude": "FDIA magnitude",
            "detection_rate": "Detection rate",
            "mean_state_rmse": "State RMSE",
            "mean_runtime_seconds": "Runtime (s)",
            "mean_num_partitions": "Mean partitions"
        }),
        "Key equal-budget results (five seeds, likelihood detector)",
        "01_equal_budget_key_metrics.png"
    )

    _plot_equal(eq)
    _plot_replay(rp)
    _diagram()
    _equal_budget_tables(eq)
    _adaptive_kl_table(eq)
    _faculty_table()
    _copy_project_figures()
    _copy_paper_context()
    _slide_content_with_notes()
    _numerical_summaries(eq, rp, equal, replay)
    _update_archive_readme()

    # Standard baseline visuals to project_results
    shutil.copy2("results/faculty_data/voltage_stress_voltage.png",
                PROJECT / "faculty_voltage_stress_waveform.png")
    shutil.copy2("results/01_measurement_fdia.png",
                PROJECT / "baseline_fdia_measurement.png")
    shutil.copy2("results/02_state_estimation.png",
                PROJECT / "baseline_state_estimation.png")

    (SUMMARIES / "results_at_a_glance.md").write_text("""# Results at a glance

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
""", encoding="utf-8")


if __name__ == "__main__":
    build()
