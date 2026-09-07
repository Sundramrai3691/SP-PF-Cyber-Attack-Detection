# Adaptive Partitioning: Paper Context vs This Prototype

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
