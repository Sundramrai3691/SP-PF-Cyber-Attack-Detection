# Archived non-final / intermediate artifacts

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
