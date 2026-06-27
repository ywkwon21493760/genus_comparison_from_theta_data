# Baseline result files

This directory contains the official baseline result files used for the final
manuscript version.

Files:
- `results_baselines_per_seed.csv`: seed-wise baseline results over all reported
  truncation lengths, preprocessing schemes, and baseline models.
- `results_baselines_summary_mean_std.csv`: aggregated mean and standard
  deviation values computed from the seed-wise baseline results.
- `run_config_baseline.json`: configuration used to generate these baseline
  results with the released script.

The baseline tables and figures in the manuscript are regenerated from
`results_baselines_summary_mean_std.csv`.
