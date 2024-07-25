# %% [markdown]
# # Visualize actin CME tomography data

# %% [markdown]
"""
Notebook contains steps for visualizing segmented tomography data for actin
fibers using [Simularium](https://simularium.allencell.org/).

- [Define visualization settings](#define-visualization-settings)
- [Visualize tomography data](#visualize-tomography-data)
"""

# %%
if __name__ != "__main__":
    raise ImportError("This module is a notebook and is not meant to be imported")

# %%
from pathlib import Path

from subcell_pipeline.analysis.compression_metrics.compression_metric import (
    CompressionMetric,
)
from subcell_pipeline.visualization.tomography import visualize_tomography

# %% [markdown]
"""
## Define visualization settings

Define simulation and visualization settings that are shared between different
simulation series.
"""

# %%
# Tomography dataset name
name = "actin_cme_tomography"

# S3 bucket for input and output files
bucket = "s3://subcell-working-bucket"

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
## Visualize tomography data

- Input: `(name)/(name)_coordinates_sampled.csv`
- Output: `(name)/(name).simularium`
"""

# %%
visualize_tomography(bucket, name, str(temp_path), metrics)
