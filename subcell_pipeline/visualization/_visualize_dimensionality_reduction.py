# %% [markdown]
# # Visualize dimensionality reduction analysis of actin filaments

# %% [markdown]
"""

Notebook contains steps for visualizing PCA space 
for actin fibers.

- [Pre-process Inputs](#pre-process-inputs)
- [Visualize Inverse PCA](#visualize-inverse-pca)
"""

# %%
if __name__ != "__main__":
    raise ImportError("This module is a notebook and is not meant to be imported")


# %% [markdown]
"""
## Pre-process Inputs

If more analysis outputs for PCA are saved in S3, this will no longer be necessary.

- Input: `(series_name)/analysis/(series_name)_(align_key).csv` (for Cytosim and ReaDDy)
- Output: `actin_compression_pca_results.csv` and `actin_compression_pca.pkl`
"""

# %%
import pandas as pd
from io_collection.save.save_dataframe import save_dataframe
from io_collection.save.save_pickle import save_pickle
from subcell_pipeline.analysis.dimensionality_reduction.fiber_data import get_merged_data
from subcell_pipeline.analysis.dimensionality_reduction.pca_dim_reduction import run_pca

# Name of the simulation series
series_name: str = "COMPRESSION_VELOCITY"

# S3 bucket for input and output files
bucket = "s3://subcell-working-bucket"

# S3 bucket Cytosim for input and output files
cytosim_bucket: str = "s3://cytosim-working-bucket"

# S3 bucket ReaDDy for input and output files
readdy_bucket: str = "s3://readdy-working-bucket"

# Random seeds for simulations
random_seeds: list[int] = [1, 2, 3, 4, 5]

# List of condition file keys for each velocity
condition_keys: list[str] = ["0047", "0150", "0470", "1500"]

readdy_data = get_merged_data(readdy_bucket, f"ACTIN_{series_name}", condition_keys, random_seeds)
readdy_data["simulator"] = "readdy"

cytosim_data = get_merged_data(
    cytosim_bucket, series_name, condition_keys, random_seeds
)
cytosim_data["simulator"] = "cytosim"

data = pd.concat([cytosim_data, readdy_data])
data["repeat"] = data["seed"] - 1
data["velocity"] = data["key"].astype("int") / 10

time_map = {
    ("cytosim", "0047"): 0.031685,
    ("cytosim", "0150"): 0.01,
    ("cytosim", "0470"): 0.00316,
    ("cytosim", "1500"): 0.001,
    ("readdy", "0047"): 1000,
    ("readdy", "0150"): 1000,
    ("readdy", "0470"): 1000,
    ("readdy", "1500"): 1000,
}

pca_results, pca = run_pca(data)

save_dataframe(bucket, "actin_compression_pca_results.csv", pca_results)
save_pickle(bucket, "actin_compression_pca.pkl", pca)

# %% [markdown]
"""
## Visualize Inverse PCA

Visualize PCA space for actin fibers.

- Input: `actin_compression_pca_results.csv` and `actin_compression_pca.pkl`
- Output: `(name)/(name).simularium`
"""

# %% 
from subcell_pipeline.visualization.visualizer import (
    visualize_dimensionality_reduction,
)

# %%
# S3 bucket for input and output files
bucket = "s3://subcell-working-bucket"

# File key for PCA results dataframe
pca_results_key = "actin_compression_pca_results.csv"

# File key for PCA object pickle
pca_pickle_key = "actin_compression_pca.pkl"

# Scroll through the PC distributions over time if True, otherwise show all together in one timestep
distribution_over_time = False

# Also show distributions for ReaDDy and Cytosim if True, otherwise just all together
simulator_detail = False

visualize_dimensionality_reduction(
    bucket, pca_results_key, pca_pickle_key, distribution_over_time, simulator_detail
)
