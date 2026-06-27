# Nonlinear/neural result files

This directory contains the official nonlinear/neural result files used for the
final manuscript version.

Files:
- `results_neural_per_seed.csv`: seed-wise nonlinear/neural results over all
  reported truncation lengths, preprocessing schemes, and model classes.
- `results_neural_summary_mean_std.csv`: aggregated mean and standard deviation
  values computed from the seed-wise nonlinear/neural results.
- `run_config_neural.json`: configuration used to generate these nonlinear/neural
  results with the released script.

The nonlinear/neural tables and figures in the manuscript are regenerated from
`results_neural_summary_mean_std.csv`.
