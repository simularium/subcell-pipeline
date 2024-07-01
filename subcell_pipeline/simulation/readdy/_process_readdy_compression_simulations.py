# %% [markdown]
# # Process ReaDDy simulations

# %% [markdown]
"""
Notebook contains steps for post processing of ReaDDy simulations in which a
single actin fiber is compressed at different compression velocities.

This notebook provides an example of processing a simulation series in which
there are multiple conditions, each of which were run with multiple replicates.
For an example of processing a simulation series with a single condition with
multiple replicates, see `process_readdy_no_compression_simulations.py`.

- [Define simulation conditions](#define-simulation-conditions)
- [Parse simulation data](#parse-simulation-data)
"""

# %%
if __name__ != "__main__":
    raise ImportError("This module is a notebook and is not meant to be imported")

# %%
from subcell_pipeline.simulation.readdy.parser import parse_readdy_simulation_data

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

# %% [markdown]
"""
## Parse simulation data

Iterate through all condition keys and replicates to load simulation output
files and parse them into a tidy data format. If the parsed file for a given
condition key and replicate already exists, parsing is skipped.

- Input: `(series_name)/outputs/(series_name)_(condition_key)_(index + 1).h5`
- Output: `(series_name)/data/(series_name)_(condition_key)_(index + 1).csv` and
  `(series_name)/data/(series_name)_(condition_key)_(index + 1).pkl`
"""

# %%
parse_readdy_simulation_data(
    bucket, series_name, condition_keys, n_replicates, n_timepoints, n_monomer_points
)
