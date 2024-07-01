# %% [markdown]
# # Run ReaDDy no compression simulations

# %% [markdown]
"""
Notebook contains steps for running ReaDDy simulations for a baseline single
actin fiber with no compression.

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

Defines the `ACTIN_NO_COMPRESSION` simulation series, which simulates a single
actin fiber with a free barbed end across five replicates.
"""

# %%
# Name of the simulation series
series_name: str = "ACTIN_COMPRESSION_VELOCITY"

# Template for simulation output files
source_template: str = "outputs/actin_compression_baseline_%d.h5"

# S3 bucket for input and output files
bucket: str = "s3://readdy-working-bucket"

# Number of simulation replicates
n_replicates: int = 5

# %% [markdown]
"""
## Copy simulation outputs
"""

# %%
copy_simulation_outputs(bucket, series_name, source_template, n_replicates)
