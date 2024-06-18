# %% [markdown]
# # Run PCA on compression simulations

# %% [markdown]
"""
Notebooks contains steps for plotting combined Cytosim and Readdy compression
velocity simulations and applying Principal Component Analysis (PCA) on
individual fibers. By default, fibers coordinates are aligned in the yz-plane to
the positive y axis, keeping x axis coordinates unchanged, before running PCA.

- [Define simulation conditions](#define-simulation-conditions)
- [Load merged data](#load-merged-data)
- [Save aligned fibers](#save-aligned-fibers)
- [Plot aligned fibers](#plot-aligned-fibers)
- [Run PCA](#run-pca)
- [Save PCA trajectories](#save-pca-trajectories)
- [Save PCA transforms](#save-pca-transforms)
- [Plot PCA feature scatter](#plot-pca-feature-scatter)
- [Plot PCA inverse transform](#plot-pca-inverse-transform)
"""

# %%
if __name__ != "__main__":
    raise ImportError("This module is a notebook and is not meant to be imported")

# %%
from pathlib import Path

import pandas as pd

from subcell_pipeline.analysis.dimensionality_reduction.fiber_data import (
    get_merged_data,
    plot_fibers_by_key_and_seed,
    save_aligned_fibers,
)
from subcell_pipeline.analysis.dimensionality_reduction.pca_dim_reduction import (
    plot_pca_feature_scatter,
    plot_pca_inverse_transform,
    run_pca,
    save_pca_trajectories,
    save_pca_transforms,
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

# %% [markdown]
"""
## Load merged data

Load merged simulation data from Cytosim and ReaDDy. Data is aligned in the
yz-plane to the positive y axis, keeping x axis coordinates unchanged. Set
`align=False` to load un-aligned data instead.
"""

# %%
readdy_data = get_merged_data(readdy_bucket, series_name, condition_keys, random_seeds)
readdy_data["simulator"] = "readdy"

# %%
cytosim_data = get_merged_data(
    cytosim_bucket, series_name, condition_keys, random_seeds
)
cytosim_data["simulator"] = "cytosim"

# %%
data = pd.concat([cytosim_data, readdy_data])
data["repeat"] = data["seed"] - 1
data["velocity"] = data["key"].astype("int") / 10

# %% [markdown]
"""
## Save aligned fibers
"""

# %%
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

save_aligned_fibers(
    data, time_map, save_location, "actin_compression_aligned_fibers.json"
)

# %% [markdown]
"""
## Plot aligned fibers
"""

# %%
plot_fibers_by_key_and_seed(data)

# %% [markdown]
"""
## Run PCA
"""

# %%
pca_results, pca = run_pca(data)

# %% [markdown]
"""
## Save PCA trajectories
"""
# %%
save_pca_trajectories(
    pca_results, save_location, "actin_compression_pca_trajectories.json"
)

# %% [markdown]
"""
## Save PCA transforms
"""
# %%
points = [
    [-600, -300, 0, 300, 600, 900],
    [-200, 0, 200, 400],
]

save_pca_transforms(pca, points, save_location, "actin_compression_pca_transforms.json")

# %% [markdown]
"""
## Plot PCA feature scatter
"""

# %%
features = {
    "SIMULATOR": {"READDY": "red", "CYTOSIM": "blue"},
    "TIME": "magma_r",
    "VELOCITY": (
        {
            4.7: 0,
            15: 1,
            47: 2,
            150: 3,
        },
        "magma_r",
    ),
    "REPEAT": "viridis",
}

plot_pca_feature_scatter(pca_results, features, pca)

# %% [markdown]
"""
## Plot PCA inverse transform
"""

# %%
plot_pca_inverse_transform(pca, pca_results)
