# %% [markdown]
# # Compare metrics across simulators

# %% [markdown]
"""
Notebook contains steps to compare metrics of fiber compression across different
simulators. Currently supports comparison of Cytosim and ReaDDy simulations.

- [Load dataframes](#load-dataframes)
- [Calculate metrics and add to dataframe](#calculate-metrics)
- [Plot metrics vs time](#plot-metrics-vs-time)
"""

# %%
if __name__ != "__main__":
    raise ImportError("This module is a notebook and is not meant to be imported")

# %%
from pathlib import Path

import pandas as pd

from subcell_pipeline.analysis.compression_metrics.compression_analysis import (
    COMPRESSIONMETRIC,
    get_compression_metric_data,
)
from subcell_pipeline.analysis.compression_metrics.plot_compression_metrics import (
    plot_metrics_vs_time,
)

# %% [markdown]
"""
## Define simulation conditions

Defines the `COMPRESSION_VELOCITY` simulation series, which compresses a single
500 nm actin fiber at four different velocities (4.7, 15, 47, and 150 μm/s) with
five replicates each (random seeds 1, 2, 3, 4, and 5).
"""

# %%
# Name of the simulation series
series_name: str = "COMPRESSION_VELOCITY"

# S3 bucket Cytosim for input and output files
cytosim_bucket: str = "s3://cytosim-working-bucket"

# S3 bucket ReaDDy for input and output files
readdy_bucket: str = "s3://readdy-working-bucket"

# Random seeds for simulations
random_seeds: list[int] = [1, 2, 3, 4, 5]

# List of condition file keys for each velocity
condition_keys: list[str] = ["0047", "0150", "0470", "1500"]

# Location to save analysis results (S3 bucket or local path)
save_location: str = str(Path(__file__).parents[3] / "analysis_outputs")

# Specify whether the metrics should be recalculated. Set this to true if you
# make changes to any metric calculation functions.
recalculate: bool = False

# %% [markdown]
"""
## Select metrics to analyze

Available metrics are defined in the `COMPRESSIONMETRIC` enum.
"""

# %%
metrics = [
    COMPRESSIONMETRIC.NON_COPLANARITY,
    COMPRESSIONMETRIC.PEAK_ASYMMETRY,
    COMPRESSIONMETRIC.AVERAGE_PERP_DISTANCE,
    COMPRESSIONMETRIC.CALC_BENDING_ENERGY,
    COMPRESSIONMETRIC.CONTOUR_LENGTH,
    COMPRESSIONMETRIC.COMPRESSION_RATIO,
]

# %% [markdown]
"""
## Calculate metrics for cytosim data

"""

# %%
cytosim_metrics = get_compression_metric_data(
    bucket=cytosim_bucket,
    series_name=series_name,
    condition_keys=condition_keys,
    random_seeds=random_seeds,
    metrics=metrics,
    recalculate=recalculate,
)
cytosim_metrics["simulator"] = "cytosim"

# %% [markdown]
"""
## Calculate metrics for readdy data

"""

# %%
readdy_metrics = get_compression_metric_data(
    bucket=readdy_bucket,
    series_name=series_name,
    condition_keys=condition_keys,
    random_seeds=random_seeds,
    metrics=metrics,
    recalculate=recalculate,
)
readdy_metrics["simulator"] = "readdy"

# %% [markdown]
"""
## Combine metrics from both simulators

"""

# %%
combined_metrics = pd.concat([cytosim_metrics, readdy_metrics])
combined_metrics["repeat"] = combined_metrics["seed"] - 1
combined_metrics["velocity"] = combined_metrics["key"].astype("int") / 10

# %% [markdown]
"""
## Plot metrics vs time

"""

# %%
plot_metrics_vs_time(df=combined_metrics, metrics=metrics, suffix="_subsampled")
