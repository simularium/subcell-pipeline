# %% [markdown]
# # Visualize actin CME tomography data

# %% [markdown]
"""

Notebook contains steps for visualizing segmented tomography data
for actin fibers.

- [Visualize Tomography](#visualize-tomography)
"""

# %%
if __name__ != "__main__":
    raise ImportError("This module is a notebook and is not meant to be imported")

# %% [markdown]
"""
## Visualize Tomography

Visualize segmented tomography data for actin fibers.

- Input: `(name)/(name)_coordinates_sampled.csv`
- Output: `(name)/(name).simularium`
"""

# %% 
from subcell_pipeline.visualization.visualizer import (
    visualize_tomography,
)

# %%
# Dataset name
name = "actin_cme_tomography"

# S3 bucket for input and output files
bucket = "s3://subcell-working-bucket"

visualize_tomography(bucket, name)
