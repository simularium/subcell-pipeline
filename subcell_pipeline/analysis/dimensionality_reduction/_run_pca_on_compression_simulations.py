# %% [markdown]
# # Run PCA on compression simulations

# %% [markdown]
"""
Notebooks contains steps for plotting combined Cytosim and Readdy compression
velocity simulations and applying Principal Component Analysis (PCA) on
individual fibers. By default, fibers are aligned to the positive x axis before
running PCA.

- [Define simulation conditions](#define-simulation-conditions)
- [Load merged data](#load-merged-data)
- [Plot aligned fibers](#plot-aligned-fibers)
- [Run PCA](#run-pca)
- [Plot PCA feature scatter](#plot-pca-feature-scatter)
- [Plot PCA inverse transform](#plot-pca-inverse-transform)
"""

# %%
if __name__ != "__main__":
    raise ImportError("This module is a notebook and is not meant to be imported")

# %%
import pandas as pd

from subcell_pipeline.analysis.dimensionality_reduction import (
    get_merged_data,
    plot_fibers_by_key_and_seed,
    plot_pca_feature_scatter,
    plot_pca_inverse_transform,
    run_pca,
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

# %% [markdown]
"""
## Load merged data

Load merged simulation data from Cytosim and ReaDDy. Data is aligned to the
positive x axis by default. Set `align=False` to load un-aligned data instead.
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

plot_pca_feature_scatter(pca_results, features)

# %% [markdown]
"""
## Plot PCA inverse transform
"""

# %%
plot_pca_inverse_transform(pca, pca_results)
