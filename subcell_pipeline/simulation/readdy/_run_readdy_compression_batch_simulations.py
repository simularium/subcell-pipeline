# %% [markdown]
# # Run ReaDDy compression simulations

# %% [markdown]
"""
Notebook contains steps for running ReaDDy simulations in which a single actin
fiber is compressed at different compression velocities.

Simulations use the ReaDDy actin model defined
[here](https://github.com/simularium/readdy-models/tree/main/examples/actin).
Instructions for running this model on AWS Batch are provided
[here](https://github.com/simularium/readdy-models/blob/main/examples/README.md).

After simulations are complete, use this notebook to copy output files into the
file structure used by this pipeline.

- [Define simulation conditions](#define-simulation-conditions)
- [Copy simulation outputs](#copy-simulation-outputs)
"""

# %%
if __name__ != "__main__":
    raise ImportError("This module is a notebook and is not meant to be imported")

# %%
from subcell_pipeline.simulation.batch_simulations import copy_simulation_outputs

# %% [markdown]
"""
## Define simulation conditions

Defines the `ACTIN_COMPRESSION_VELOCITY` simulation series, which compresses a
single 500 nm actin fiber at four different velocities (4.7, 15, 47, and 150
μm/s) with five replicates each.
"""

# %%
# Name of the simulation series
series_name: str = "ACTIN_COMPRESSION_VELOCITY"

# Template for simulation output files
source_template: str = "outputs/actin_compression_velocity=%s_%d.h5"

# S3 bucket for input and output files
bucket: str = "s3://readdy-working-bucket"

# Number of simulation replicates
n_replicates: int = 5

# File keys for each velocity
velocity_keys: dict[str, str] = {
    "4.7": "0047",
    "15": "0150",
    "47": "0470",
    "150": "1500",
}

# %% [markdown]
"""
## Copy simulation outputs
"""

# %%
copy_simulation_outputs(
    bucket, series_name, source_template, n_replicates, velocity_keys
)
