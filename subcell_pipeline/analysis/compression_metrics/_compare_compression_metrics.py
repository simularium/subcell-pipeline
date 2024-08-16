# %% [markdown]
# # Compare compression metrics between simulators

# %% [markdown]
"""
Notebook contains steps to compare metrics of fiber compression across different
simulators. Currently supports comparison of Cytosim and ReaDDy simulations.

- [Define simulation conditions](#define-simulation-conditions)
- [Select metrics to analyze](#select-metrics-to-analyze)
- [Calculate metrics for Cytosim data](#calculate-metrics-for-cytosim-data)
- [Calculate metrics for ReaDDy data](#calculate-metrics-for-readdy-data)
- [Combine metrics from both simulators](#combine-metrics-from-both-simulators)
- [Save combined compression metrics](#save-combined-compression-metrics)
- [Plot metrics vs time](#plot-metrics-vs-time)
- [Plot metrics histograms](#plot-metrics-histograms)
"""

# %%
if __name__ != "__main__":
    raise ImportError("This module is a notebook and is not meant to be imported")

# %%
import pandas as pd

from subcell_pipeline.analysis.compression_metrics.compression_analysis import (
    get_compression_metric_data,
    plot_metric_distribution,
    plot_metrics_vs_time,
    save_compression_metrics,
)
from subcell_pipeline.analysis.compression_metrics.compression_metric import (
    CompressionMetric,
)

# %% [markdown]
"""
## Define simulation conditions

Defines the `COMPRESSION_VELOCITY` simulation series, which compresses a single
500 nm actin fiber at four different velocities (4.7, 15, 47, and 150 μm/s) with
five replicates each and the baseline `NO_COMPRESSION` simulation series, which
simulates a single actin fiber with a free barbed end across five replicates.
"""

# %%
# Name of the simulation series
compression_series_name: str = "COMPRESSION_VELOCITY"
no_compression_series_name: str = "NO_COMPRESSION"

# S3 bucket Cytosim for input and output files
cytosim_bucket: str = "s3://cytosim-working-bucket"

# S3 bucket ReaDDy for input and output files
readdy_bucket: str = "s3://readdy-working-bucket"

# Random seeds for simulations
random_seeds: list[int] = [1, 2, 3, 4, 5]

# List of condition file keys for each velocity
condition_keys: list[str] = ["0047", "0150", "0470", "1500"]

# Location to save analysis results (S3 bucket or local path)
save_location: str = "s3://subcell-working-bucket"

# Specify whether the metrics should be recalculated. Set this to true if you
# make changes to any metric calculation functions.
recalculate: bool = True

# %% [markdown]
"""
## Select metrics to analyze

Available metrics are defined in the `CompressionMetric` enum.
"""

# %%
metrics = [
    CompressionMetric.NON_COPLANARITY,
    CompressionMetric.PEAK_ASYMMETRY,
    CompressionMetric.AVERAGE_PERP_DISTANCE,
    CompressionMetric.CALC_BENDING_ENERGY,
    CompressionMetric.CONTOUR_LENGTH,
    CompressionMetric.COMPRESSION_RATIO,
    CompressionMetric.TOTAL_FIBER_TWIST,
    CompressionMetric.TWIST_ANGLE,
]

# %% [markdown]
"""
## Calculate metrics for Cytosim data
"""

# %%
cytosim_metrics_compression = get_compression_metric_data(
    bucket=cytosim_bucket,
    series_name=compression_series_name,
    condition_keys=condition_keys,
    random_seeds=random_seeds,
    metrics=metrics,
    recalculate=recalculate,
)
cytosim_metrics_compression["simulator"] = "cytosim"

# %%
cytosim_metrics_no_compression = get_compression_metric_data(
    bucket=cytosim_bucket,
    series_name=no_compression_series_name,
    condition_keys=[""],
    random_seeds=random_seeds,
    metrics=metrics,
    recalculate=recalculate,
)
cytosim_metrics_no_compression["simulator"] = "cytosim"

# %% [markdown]
"""
## Calculate metrics for ReaDDy data
"""

# %%
readdy_metrics_compression = get_compression_metric_data(
    bucket=readdy_bucket,
    series_name=f"ACTIN_{compression_series_name}",
    condition_keys=condition_keys,
    random_seeds=random_seeds,
    metrics=metrics,
    recalculate=recalculate,
)
readdy_metrics_compression["simulator"] = "readdy"

# %%
readdy_metrics_no_compression = get_compression_metric_data(
    bucket=readdy_bucket,
    series_name=f"ACTIN_{no_compression_series_name}",
    condition_keys=[""],
    random_seeds=random_seeds,
    metrics=metrics,
    recalculate=recalculate,
)
readdy_metrics_no_compression["simulator"] = "readdy"

# %% [markdown]
"""
## Combine metrics from both simulators
"""

# %%
combined_metrics = pd.concat([cytosim_metrics_compression, readdy_metrics_compression])
combined_metrics["repeat"] = combined_metrics["seed"] - 1
combined_metrics["velocity"] = combined_metrics["key"].astype("int") / 10

# %% [markdown]
"""
## Save combined compression metrics
"""

# %%
save_compression_metrics(
    combined_metrics,
    save_location,
    "compression_metrics/actin_compression_combined_metrics.csv",
)

# %% [markdown]
"""
## Plot metrics vs time
"""

# %%
plot_metrics_vs_time(
    df=combined_metrics,
    metrics=metrics,
    save_location=save_location,
    save_key_template="compression_metrics/actin_compression_metrics_over_time_subsampled_%s.png",
)

# %% [markdown]
"""
## Plot metrics histograms
"""

# %%
plot_metric_distribution(
    df=combined_metrics,
    metrics=metrics,
    save_location=save_location,
    save_key_template="compression_metrics/actin_compression_metrics_histograms_subsampled_%s.png",
)
