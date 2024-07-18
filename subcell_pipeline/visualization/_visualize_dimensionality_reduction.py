# %% [markdown]
# # Visualize dimensionality reduction analysis of actin filaments

# %% [markdown]
"""
Notebook contains steps for visualizing dimensionality reduction using PCA for
actin fibers.

- [Define visualization settings](#define-visualization-settings)
- [Visualize inverse PCA](#visualize-inverse-pca)
"""

# %%
if __name__ != "__main__":
    raise ImportError("This module is a notebook and is not meant to be imported")

# %%
from pathlib import Path

from subcell_pipeline.visualization.dimensionality_reduction import (
    visualize_dimensionality_reduction,
)

# %% [markdown]
"""
## Define visualization settings

Define simulation and visualization settings that are shared between different
simulation series.
"""

# %%
# S3 bucket for input and output files
bucket = "s3://subcell-working-bucket"

# File key for PCA results dataframe
pca_results_key = "actin_compression_pca_results.csv"

# File key for PCA object pickle
pca_pickle_key = "actin_compression_pca.pkl"

# Temporary path to save visualization files
temp_path: Path = Path(__file__).parents[2] / "viz_outputs"
temp_path.mkdir(parents=True, exist_ok=True)

# Select how PC distributions are shown
# - True to scroll through the PC distributions over time
# - False to show all together in one timestep
distribution_over_time = True

# Select if simulator distributions are shown
# - True to show ReaDDy and Cytosim separately
# - False to show all together
simulator_detail = False

# Ranges to sample for each PC
range_pc1 = [-1200, 900]
range_pc2 = [-550, 250]

# Select how PCs are saved
# - True to save each PC in a separate file
# - False to save all together
separate_pcs = True

# Number of samples for each PC distribution
sample_resolution = 200

# %% [markdown]
"""
## Visualize inverse PCA

Visualize PCA space for actin fibers.

- Input: `actin_compression_pca_results.csv` and `actin_compression_pca.pkl`
- Output: `(name)/(name).simularium`
"""

# %%
visualize_dimensionality_reduction(
    bucket,
    pca_results_key,
    pca_pickle_key,
    distribution_over_time,
    simulator_detail,
    range_pc1,
    range_pc2,
    separate_pcs,
    sample_resolution,
    str(temp_path),
)
