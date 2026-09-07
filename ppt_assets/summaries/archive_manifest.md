# PPT Package Archive Manifest

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
