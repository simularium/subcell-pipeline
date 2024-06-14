# %% [markdown]
# # Process Cytosim no compression simulations

# %% [markdown]
"""
Notebook contains steps for post processing of Cytosim simulations for a
baseline single actin fiber with no compression.

This notebook provides an example of processing a simulation series for a single
condition with multiple replicates. For an example of processing a simulation
series with multiple conditions, each of which have multiple replicates, see
`process_cytosim_compression_simulations.py`.

- [Define simulation conditions](#define-simulation-conditions)
- [Parse simulation data](#parse-simulation-data)
- [Define sampling settings](#define-sampling-settings)
- [Sample simulation data](#sample-simulation-data)
"""

# %%
from subcell_pipeline.simulation.cytosim.post_processing import (
    parse_cytosim_simulation_data,
)
from subcell_pipeline.simulation.post_processing import sample_simulation_data

# %% [markdown]
"""
## Define simulation conditions

Defines the `NO_COMPRESSION` simulation series, which simulates a single actin
fiber with a free barbed end across five replicates (random seeds 1, 2, 3, 4,
and 5).
"""

# %%
# Name of the simulation series
series_name: str = "NO_COMPRESSION"

# S3 bucket for input and output files
bucket: str = "s3://cytosim-working-bucket"

# Random seeds for simulations
random_seeds: list[int] = [1, 2, 3, 4, 5]

# %% [markdown]
"""
## Parse simulation data

Iterate through all condition keys and random seeds to load simulation output
files and parse them into a tidy data format. If the parsed file for a given
condition key and random seed already exists, parsing is skipped.

- Input: `(name)/outputs/(name)_(condition_key)_(index)/`
- Output: `(name)/data/(name)_(condition_key)_(seed).csv`
"""

# %%
parse_cytosim_simulation_data(bucket, series_name, [""], random_seeds)

# %% [markdown]
"""
## Define sampling settings

Defines the settings used for sub sampling data timepoints and monomer points.
"""

# %%
# Number of timepoints
n_timepoints = 200

# Number of monomer points per fiber
n_monomer_points = 200

# %% [markdown]
"""
## Sample simulation data

Iterate through all condition keys and random seeds to load the parsed data and
sample the timepoints and monomer points. If the sampled file for a given
condition key and random seed already exists, sampling is skipped.

- Input: `(name)/data/(name)_(condition_key)_(seed).csv`
- Output: `(name)/samples/(name)_(condition_key)_(seed).csv`
"""

# %%
sample_simulation_data(
    bucket, series_name, [""], random_seeds, n_timepoints, n_monomer_points
)

# %%