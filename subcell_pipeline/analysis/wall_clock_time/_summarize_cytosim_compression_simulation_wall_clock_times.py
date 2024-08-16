# %% [markdown]
# # Summarize Cytosim compression simulation wall clock times

# %% [markdown]
"""
Notebook contains steps for extracting wall clock time for Cytosim simulations
in which a single actin fiber is compressed at different compression
velocities.

- [Define simulation conditions](#define-simulation-conditions)
- [Extract wall clock time from logs](#extract-wall-clock-time-from-logs)
"""

# %%
if __name__ != "__main__":
    raise ImportError("This module is a notebook and is not meant to be imported")

# %%
from io_collection.save.save_dataframe import save_dataframe

from subcell_pipeline.analysis.wall_clock_time.log_data import (
    get_wall_clock_time_from_logs,
)

# %% [markdown]
"""
## Define simulation conditions

Defines the `ACTIN_COMPRESSION_VELOCITY` simulation series, which compresses a
single 500 nm actin fiber at four different velocities (4.7, 15, 47, and 150
μm/s) with five replicates each (random seeds 1, 2, 3, 4, and 5).
"""

# %%
# Name of the simulation series
series_name: str = "ACTIN_COMPRESSION_VELOCITY"

# S3 bucket for input and output files
bucket: str = "s3://cytosim-working-bucket"

# Random seeds for simulations
random_seeds: list[int] = [1, 2, 3, 4, 5]

# List of condition file keys for each velocity
condition_keys: list[str] = ["0047", "0150", "0470", "1500"]

# Job ARN
job_arns: dict = {
    "0047": "c1210f24-d2d4-47ee-84c5-c0bbcd6d6d1a",
    "0150": "bab2b1b2-8159-412b-bfee-9ed5f5ee53d8",
    "0470": "b57fdc74-a1f8-45df-867b-492c99009b46",
    "1500": "36d1198f-16cc-4564-ae30-a2944753c143",
}

# Pattern for finding wall clock time in logs
pattern = "end\s+([0-9]+) s"

# %% [markdown]
"""
## Extract wall clock time from logs

Iterate through all the logs and extract wall clock time using the specified
regex pattern.
"""

# %%
logs = get_wall_clock_time_from_logs(
    bucket, series_name, condition_keys, random_seeds, job_arns, pattern
)

# %%
wall_clock_time_key = f"{series_name}/analysis/{series_name}_wall_clock_times.csv"
save_dataframe(bucket, wall_clock_time_key, logs)

# %%
print(logs.groupby("condition")["time"].describe())
