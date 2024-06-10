# %% [markdown]
"""
# Run Cytosim single fiber simulations

Notebook contains steps for running Cytosim simulations for a baseline single
actin fiber.

This notebook uses [Cytosim](https://github.com/simularium/Cytosim) configs and
scripts. Clone a copy and set the environment variable
`CYTOSIM=/path/to/Cytosim/`.

This notebook provides an example of running a simulation series for a single
condition for multiple replicates. For an example of running a simulation series
for multiple conditions, each of which need to be run for multiple replicates,
see `run_cytosim_compression_batch_simulations.py`.

- [Define simulation conditions](#define-simulation-conditions)
- [Generate configs from template](#generate-configs-from-template)
- [Define simulation settings](#define-simulation-settings)
- [Create and run jobs](#create-and-run-jobs)
"""  # noqa: D400, D415

# %%
import getpass
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from subcell_pipeline.simulation.batch_simulations import (
    check_and_save_job_logs,
    generate_configs_from_file,
    register_and_run_simulations,
)

# %%
load_dotenv()
cytosim_path: Path = Path(os.getenv("CYTOSIM", "."))

# %% [markdown]
"""
## Define simulation conditions

Defines the `SINGLE_FIBER` simulation series, which simulates a single actin
fiber with a free barbed end across five replicates (random seeds 1, 2, 3, 4,
and 5).
"""

# %%
# Name of the simulation series
series_name: str = "SINGLE_FIBER"

# S3 bucket for input and output files
bucket: str = "s3://cytosim-working-bucket"

# Random seeds for simulations
random_seeds: list[int] = [1, 2, 3, 4, 5]

# Path to the config file
config_file: str = str(cytosim_path / "configs" / "free_barbed_end_final.cym")

# Current timestamp used to organize input and outfile files
timestamp: str = datetime.now().strftime("%Y-%m-%d")

# %% [markdown]
"""
## Generate configs from template

For the config file, separate configs are saved to S3 bucket for each replicate.
"""

# %%
generate_configs_from_file(
    bucket,
    series_name,
    timestamp,
    random_seeds,
    config_file,
)

# %% [markdown]
"""
## Define simulation settings

Defines the AWS Batch settings for the simulations. Note that these settings
will need to be modified for different AWS accounts.
"""

# %%
# AWS account number
aws_account: str = getpass.getpass()

# AWS region
aws_region: str = "us-west-2"

# Prefix for job name and image
aws_user: str = "jessicay"

# Image name and version
image: str = "cytosim:0.0.0"

# Number of vCPUs for each job
vcpus: int = 1

# Memory for each job
memory: int = 7000

# Job queue
job_queue: str = "general_on_demand"

# Job array size
job_size: int = len(random_seeds)

# %% [markdown]
"""
## Register and run jobs

For each velocity, we create a new job definition that specifies the input
configs and the output location. Each job definition is then registered (for job
definitions with the same name, a new revision will be created unless no
parameters have changed). All replicates for a given velocities are submitted as
a job array. Job status can be monitored via the AWS Console.
"""

# %%
job_arns = register_and_run_simulations(
    bucket,
    series_name,
    timestamp,
    [series_name],
    aws_account,
    aws_region,
    aws_user,
    image,
    vcpus,
    memory,
    job_queue,
    job_size,
)

# %% [markdown]
"""
## Check and save job logs

Iterates through the list of submitted job ARNs to check job status. If job does
not have the "SUCCEEDED" status, print the current status. If job does have the
"SUCCEEDED" status, then save a copy of the CloudWatch logs. The list of job
ARNs can be manually adjusted to limit which jobs are checks or avoid saving
logs that have already been saved.
"""

# %%
check_and_save_job_logs(bucket, series_name, job_arns, aws_region)
