# %% [markdown]
# # Process ReaDDy simulations

# %% [markdown]
"""

Notebook contains steps for visualizing ReaDDy and Cytosim 
simulations of a single actin fiber.

- [Visualize ReaDDy](#visualize-readdy)
- [Visualize Cytosim](#visualize-cytosim)
"""

# %%
if __name__ != "__main__":
    raise ImportError("This module is a notebook and is not meant to be imported")

# %% [markdown]
"""
## Visualize ReaDDy

Iterate through all condition keys and replicates to load simulation 
output files and visualize them. If the visualization file for a given
condition key and replicate already exists and overwrite_existing is False, 
parsing is skipped.

- Input: `(series_name)/outputs/(series_name)_(condition_key)_(index+1).h5`
- Output: `(series_name)/viz/(series_name)_(condition_key)_(index+1).simularium`
"""

# %% 
from subcell_pipeline.visualization.visualizer import (
    visualize_individual_readdy_trajectories,
)
# %%
# Name of the simulation series
series_name: str = "ACTIN_COMPRESSION_VELOCITY"

# S3 bucket for input and output files
bucket: str = "s3://readdy-working-bucket"

# Number of simulation replicates
n_replicates: int = 5

# List of condition file keys for each velocity
condition_keys: list[str] = ["0047", "0150", "0470", "1500"]

# Number of timepoints
n_timepoints = 200

# Number of monomer points per fiber
n_monomer_points = 200

visualize_individual_readdy_trajectories(
    bucket,
    "ACTIN_NO_COMPRESSION",
    [""],
    n_replicates,
    n_timepoints,
    n_monomer_points,
    overwrite_existing=True,
)

visualize_individual_readdy_trajectories(
    bucket,
    "ACTIN_COMPRESSION_VELOCITY",
    condition_keys,
    n_replicates,
    n_timepoints,
    n_monomer_points,
    overwrite_existing=True,
)

# %% [markdown]
"""
## Visualize Cytosim

Iterate through all condition keys and random seeds to load simulation output
dataframes and visualize them. If the visualization file for a given
condition key and random seed already exists, parsing is skipped.

- Input: `(series_name)/samples/(series_name)_(condition_key)_(seed)/`
- Output: `(series_name)/viz/(series_name)_(condition_key)_(seed).simularium`
"""

# %% 
from subcell_pipeline.visualization.visualizer import (
    visualize_individual_cytosim_trajectories,
)
# %%
# S3 bucket for input and output files
bucket: str = "s3://cytosim-working-bucket"

# Random seeds for simulations
random_seeds: list[int] = [1, 2, 3, 4, 5]

# List of condition file keys for each velocity
condition_keys: list[str] = ["0047", "0150", "0470", "1500"]

# Number of timepoints
n_timepoints = 200

visualize_individual_cytosim_trajectories(
    bucket,
    "NO_COMPRESSION",
    [""],
    random_seeds,
    n_timepoints,
    overwrite_existing=True,
)

visualize_individual_cytosim_trajectories(
    bucket,
    "COMPRESSION_VELOCITY",
    condition_keys,
    random_seeds,
    n_timepoints,
    overwrite_existing=True,
)
