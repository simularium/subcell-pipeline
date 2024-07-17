# %% [markdown]
# # Visualize tangent angles for fibers

# %% [markdown]
"""
Notebook contains steps to visualize the twisting angles for fibers.

- [Define visualization settings](#define-visualization-settings)
- [Visualize combined trajectories](#visualize-combined-trajectories)
"""

# %%
if __name__ != "__main__":
    raise ImportError("This module is a notebook and is not meant to be imported")

# %%
from pathlib import Path

import pandas as pd

from subcell_pipeline.analysis.dimensionality_reduction.fiber_data import (
    get_merged_data,
)
from subcell_pipeline.visualization.fiber_angles import visualize_tangent_angles

# %% [markdown]
"""
## Define visualization settings
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

# Location to save plot of metrics vs time (local path)
save_location: Path = Path(__file__).parents[3] / "analysis_outputs"
save_location.mkdir(parents=True, exist_ok=True)

# Specify whether the metrics should be recalculated. Set this to true if you
# make changes to any metric calculation functions.
recalculate: bool = True

# %%
readdy_data = get_merged_data(
    readdy_bucket, f"ACTIN_{series_name}", condition_keys, random_seeds
)
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
## Visualize tangent angles
"""
visualize_tangent_angles(data)

# %%
