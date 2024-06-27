# %% [markdown]
# # Process ReaDDy simulations

# %% [markdown]
"""

Notebook contains steps for visualizing ReaDDy and Cytosim 
simulations of a single actin fiber.

- [Visualize Combined](#visualize-combined)
"""

# %%
if __name__ != "__main__":
    raise ImportError("This module is a notebook and is not meant to be imported")

# %% [markdown]
"""
## Visualize Combined

Visualize all simulations with compression from ReaDDy and Cytosim together in Simularium.

- Input: `(readdy_series_name)/data/(readdy_series_name)_(condition_key)_(index+1).csv` 
    and `(cytosim_series_name)/samples/(cytosim_series_name)_(condition_key)_(seed).csv`
- Output: `actin_compression_cytosim_readdy.simularium`
"""

# %% 
from subcell_pipeline.visualization.visualizer import (
    visualize_all_compressed_trajectories_together,
)
# %%
# S3 bucket for combined input and output files
subcell_bucket: str = "s3://subcell-working-bucket"

# S3 bucket for ReaDDy input and output files
readdy_bucket: str = "s3://readdy-working-bucket"

# Name of the ReaDDy simulation series
readdy_series_name: str = "ACTIN_COMPRESSION_VELOCITY"

# S3 bucket for input and output files
cytosim_bucket: str = "s3://cytosim-working-bucket"

# Name of the simulation series
cytosim_series_name: str = "COMPRESSION_VELOCITY"

# List of condition file keys for each velocity
condition_keys: list[str] = ["0047", "0150", "0470", "1500"]

# Number of simulation replicates
n_replicates: int = 5

# Number of timepoints
n_timepoints = 200

visualize_all_compressed_trajectories_together(
    subcell_bucket,
    readdy_bucket,
    readdy_series_name,
    cytosim_bucket,
    cytosim_series_name,
    condition_keys,
    n_replicates,
    n_timepoints,
)
