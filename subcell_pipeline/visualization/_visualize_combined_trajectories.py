# %% [markdown]
# # Visualize combined ReaDDy and Cytosim simulation trajectories

# %% [markdown]
"""
Notebook contains steps for visualizing ReaDDy and Cytosim simulations of a
single actin fiber using [Simularium](https://simularium.allencell.org/).

- [Define visualization settings](#define-visualization-settings)
- [Visualize combined trajectories](#visualize-combined-trajectories)
"""

# %%
if __name__ != "__main__":
    raise ImportError("This module is a notebook and is not meant to be imported")

# %%
from pathlib import Path

from subcell_pipeline.analysis.compression_metrics.compression_metric import (
    CompressionMetric,
)
from subcell_pipeline.visualization.combined_trajectory import (
    visualize_combined_trajectories,
)

# %% [markdown]
"""
## Define visualization settings
"""

# %%
# S3 buckets for simulation and visualization input and output files
buckets: dict[str, str] = {
    "combined": "s3://subcell-working-bucket",
    "readdy": "s3://readdy-working-bucket",
    "cytosim": "s3://cytosim-working-bucket",
}

# Names of the simulation series for each simulator
series_names: dict[str, str] = {
    "readdy": "ACTIN_COMPRESSION_VELOCITY",
    "cytosim": "COMPRESSION_VELOCITY",
}

# List of condition file keys for each velocity
condition_keys: list[str] = ["0047", "0150", "0470", "1500"]

# Replicate ids for simulations
replicates: list[int] = [1, 2, 3, 4, 5]

# Number of timepoints
n_timepoints = 201

# List of simulators and colors
simulator_colors = {
    "cytosim": "#1cbfa4",
    "readdy": "#ffae52",
}

# Temporary path to save visualization files
temp_path: Path = Path(__file__).parents[2] / "viz_outputs"
temp_path.mkdir(parents=True, exist_ok=True)

# List of compression metrics to include
metrics = [
    CompressionMetric.NON_COPLANARITY,
    CompressionMetric.PEAK_ASYMMETRY,
    CompressionMetric.AVERAGE_PERP_DISTANCE,
    CompressionMetric.CALC_BENDING_ENERGY,
    CompressionMetric.CONTOUR_LENGTH,
    CompressionMetric.COMPRESSION_RATIO,
]


# %% [markdown]
"""
## Visualize combined trajectories

Visualize all compression simulations from ReaDDy and Cytosim together in
Simularium.

- Input: `(series_name)/samples/(series_name)_(condition_key)_(replicate).csv`
- Output: `actin_compression_cytosim_readdy.simularium`
"""

# %%
visualize_combined_trajectories(
    buckets,
    series_names,
    condition_keys,
    replicates,
    n_timepoints,
    simulator_colors,
    str(temp_path),
    metrics=metrics,
)
