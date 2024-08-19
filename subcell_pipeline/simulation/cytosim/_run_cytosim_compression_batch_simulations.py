# %% [markdown]
# # Run Cytosim compression simulations

# %% [markdown]
"""
Notebook contains steps for running Cytosim simulations in which a single actin
fiber is compressed at different compression velocities.

This notebook uses [Cytosim](https://github.com/simularium/Cytosim) templates
and scripts. Clone a copy and set the environment variable
`CYTOSIM=/path/to/Cytosim/`.

This notebook provides an example of running a simulation series in which there
are multiple conditions, each of which need to be run for multiple replicates.
For an example of running a simulation series for a single condition for
multiple replicates, see `run_cytosim_no_compression_batch_simulations.py`.

- [Define simulation conditions](#define-simulation-conditions)
- [Generate configs from template](#generate-configs-from-template)
- [Define simulation settings](#define-simulation-settings)
- [Register and run jobs](#register-and-run-jobs)
- [Check and save job logs](#check-and-save-job-logs)
"""

# %%
if __name__ != "__main__":
    raise ImportError("This module is a notebook and is not meant to be imported")

# %%
import getpass
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from subcell_pipeline.simulation.batch_simulations import (
    check_and_save_job_logs,
    generate_configs_from_template,
    register_and_run_simulations,
)

# %%
load_dotenv()
cytosim_path: Path = Path(os.getenv("CYTOSIM", "."))

# %% [markdown]
"""
Template generator uses the Preconfig class from Cytosim. Append the path to
`preconfig.py` from a copy of the `Cytosim` repository, or download a copy into
this location.
"""

# %%
sys.path.append(str(cytosim_path / "python" / "run"))
from preconfig import Preconfig  # noqa: E402

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

# Path to the config template file
path_to_template: Path = cytosim_path / "templates" / "vary_compress_rate.cym.tpl"

# Current timestamp used to organize input and outfile files
timestamp: str = datetime.now().strftime("%Y-%m-%d")

# File keys for each velocity
velocity_keys: dict[str, str] = {
    "4.73413649": "0047",
    "15": "0150",
    "47.4341649": "0470",
    "150": "1500",
}

# %% [markdown]
"""
## Generate configs from template

Use the `Preconfig` from Cytosim to convert the `.cym.tpl` template file into
local `.cym` config files. For each file, extract the compression velocity to
use as the simulation condition key. Save all config files to S3 bucket.
"""

# %%
preconfig = Preconfig()
config_files = preconfig.parse(path_to_template, {}, path=cytosim_path / "configs")

# %%
pattern = r"compression_velocity:([\s0-9\.]+)"
group_keys = generate_configs_from_template(
    bucket,
    series_name,
    timestamp,
    random_seeds,
    config_files,
    pattern,
    velocity_keys,
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
definitions with the same name, a new revisions will be created unless no
parameters have changed). All replicates for a given velocities are submitted as
a job array. Job status can be monitored via the AWS Console.
"""

# %%
job_arns = register_and_run_simulations(
    bucket,
    series_name,
    timestamp,
    group_keys,
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
