# %% [markdown]
# # Process ReaDDy no compression simulations

# %% [markdown]
"""
Notebook contains steps for post processing of ReaDDy simulations for a baseline
single actin fiber with no compression.

This notebook provides an example of processing a simulation series for a single
condition with multiple replicates. For an example of processing a simulation
series with multiple conditions, each of which have multiple replicates, see
`process_readdy_compression_simulations.py`.

- [Define simulation conditions](#define-simulation-conditions)
- [Parse simulation data](#parse-simulation-data)
"""

# %%
if __name__ != "__main__":
    raise ImportError("This module is a notebook and is not meant to be imported")

# %%
from pathlib import Path

from subcell_pipeline.simulation.readdy.parser import parse_readdy_simulation_data

# %% [markdown]
"""
## Define simulation conditions

Defines the `ACTIN_NO_COMPRESSION` simulation series, which simulates a single
actin fiber with a free barbed end across five replicates.
"""

# %%
# Name of the simulation series
series_name: str = "ACTIN_NO_COMPRESSION"

# S3 bucket for input and output files
bucket: str = "s3://readdy-working-bucket"

# Number of simulation replicates
n_replicates: int = 5

# Number of timepoints
n_timepoints = 200

# Number of monomer points per fiber
n_monomer_points = 200

# Total number of steps for each condition
total_steps: dict[str, int] = {"": int(1e7)}

# Temporary path to save downloaded trajectories
temp_path: Path = Path(__file__).parents[3] / "aws_downloads"
temp_path.mkdir(parents=True, exist_ok=True)

# %% [markdown]
"""
## Parse simulation data

Iterate through all replicates to load simulation output files and parse them
into a tidy data format. If the parsed file for a given replicate already
exists, parsing is skipped.

- Input: `(series_name)/outputs/(series_name)_(index + 1).h5`
- Input: `(series_name)/data/(series_name)_(index + 1).pkl`
- Output: `(series_name)/samples/(series_name)_(index + 1).csv`
"""

# %%
parse_readdy_simulation_data(
    bucket,
    series_name,
    [""],
    n_replicates,
    n_timepoints,
    n_monomer_points,
    total_steps,
    str(temp_path),
)
