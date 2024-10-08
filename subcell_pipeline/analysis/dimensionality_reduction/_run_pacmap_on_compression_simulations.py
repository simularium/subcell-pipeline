# %% [markdown]
# # Run PaCMAP on compression simulations

# %% [markdown]
"""
Notebooks contains steps for plotting combined Cytosim and Readdy compression
velocity simulations and applying Pairwise Controlled Manifold Approximation
(PaCMAP) on individual fibers. By default, fibers coordinates are aligned in the
yz-plane to the positive y axis, keeping x axis coordinates unchanged, before
running PaCMAP.

- [Define simulation conditions](#define-simulation-conditions)
- [Load merged data](#load-merged-data)
- [Plot aligned fibers](#plot-aligned-fibers)
- [Run PaCMAP](#run-pacmap)
- [Plot PaCMAP feature scatter](#plot-pacmap-feature-scatter)
"""

# %%
if __name__ != "__main__":
    raise ImportError("This module is a notebook and is not meant to be imported")

# %%
import pandas as pd

from subcell_pipeline.analysis.dimensionality_reduction.fiber_data import (
    get_merged_data,
    plot_fibers_by_key_and_seed,
)
from subcell_pipeline.analysis.dimensionality_reduction.pacmap_dim_reduction import (
    plot_pacmap_feature_scatter,
    run_pacmap,
)

# %% [markdown]
"""
## Define simulation conditions

Defines the `ACTIN_COMPRESSION_VELOCITY` simulation series, which compresses a
single 500 nm actin fiber at four different velocities (4.7, 15, 47, and 150
μm/s) with five replicates each (random seeds 1, 2, 3, 4, and 5).
"""

# %%
# Name of the simulation series
series_name: str = "ACTIN_COMPRESSION_VELOCITY"

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
## Plot aligned fibers
"""

# %%
plot_fibers_by_key_and_seed(
    data, save_location, "dimensionality_reduction/actin_compression_aligned_fibers.png"
)

# %% [markdown]
"""
## Run PaCMAP
"""

# %%
pacmap_results, pacmap = run_pacmap(data)

# %% [markdown]
"""
## Plot PaCMAP feature scatter
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

plot_pacmap_feature_scatter(
    pacmap_results,
    features,
    save_location,
    "dimensionality_reduction/actin_compression_pacmap_feature_scatter.png",
)
