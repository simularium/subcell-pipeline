# %% [markdown]
# # Visualize Cytosim simulation trajectories

# %% [markdown]
"""
Notebook contains steps for visualizing Cytosim simulations of a single actin
fiber using [Simularium](https://simularium.allencell.org/).

- [Define visualization settings](#define-visualization-settings)
- [Visualize compression simulations](#visualize-compression-simulations)
- [Visualize no compression simulations](#visualize-no-compression-simulations)
"""

# %%
if __name__ != "__main__":
    raise ImportError("This module is a notebook and is not meant to be imported")

# %%
from pathlib import Path

from subcell_pipeline.analysis.compression_metrics.compression_metric import (
    CompressionMetric,
)
from subcell_pipeline.visualization.individual_trajectory import (
    visualize_individual_cytosim_trajectories,
)

# %% [markdown]
"""
## Define visualization settings

Define simulation and visualization settings that are shared between different
simulation series.
"""

# %%
# S3 bucket for input and output files
bucket: str = "s3://cytosim-working-bucket"

# Number of timepoints
n_timepoints = 200

# Specify whether the visualization should be recalculated. Set this to true if
# you make changes to any visualization functions.
recalculate: bool = True

# Random seeds for simulations
random_seeds: list[int] = [1, 2, 3, 4, 5]

# Temporary path to save visualization files
temp_path: Path = Path(__file__).parents[2] / "viz_outputs"
temp_path.mkdir(parents=True, exist_ok=True)

# List of compression metrics to include
metrics = [
    CompressionMetric.NON_COPLANARITY,
    CompressionMetric.PEAK_ASYMMETRY,
    CompressionMetric.AVERAGE_PERP_DISTANCE,
    CompressionMetric.CONTOUR_LENGTH,
    CompressionMetric.COMPRESSION_RATIO,
]

# %% [markdown]
"""
## Visualize compression simulations

The `ACTIN_COMPRESSION_VELOCITY` simulation series compresses a single 500 nm
actin fiber at four different velocities (4.7, 15, 47, and 150 μm/s) with five
replicates each.

Iterate through all condition keys and replicates to load simulation output
files and visualize them. If the visualization file for a given condition key
and replicate already exists and recalculate is False, visualization is skipped.

- Input: `(series_name)/outputs/(series_name)_(condition_key)_(index)/`
- Output: `(series_name)/viz/(series_name)_(condition_key)_(seed).simularium`
"""

# %%
# Name of the simulation series
compression_series_name: str = "ACTIN_COMPRESSION_VELOCITY"

# List of condition file keys for each velocity
compression_condition_keys: list[str] = ["0047", "0150", "0470", "1500"]

# %%
visualize_individual_cytosim_trajectories(
    bucket,
    compression_series_name,
    compression_condition_keys,
    random_seeds,
    n_timepoints,
    str(temp_path),
    metrics=metrics,
    recalculate=recalculate,
)

# %% [markdown]
"""
## Visualize no compression simulations

The `ACTIN_NO_COMPRESSION` simulation series simulates a single actin fiber with
a free barbed end across five replicates.

Iterate through all replicates to load simulation output files and visualize
them. If the visualization file for a given replicate already exists and
recalculate is False, visualization is skipped.

- Input: `(series_name)/outputs/(series_name)_(index)/`
- Output: `(series_name)/viz/(series_name)_(seed).simularium`
"""

# %%
# Name of the simulation series
no_compression_series_name: str = "ACTIN_NO_COMPRESSION"

# %%
visualize_individual_cytosim_trajectories(
    bucket,
    no_compression_series_name,
    [""],
    random_seeds,
    n_timepoints,
    str(temp_path),
    metrics=metrics,
    recalculate=recalculate,
)
