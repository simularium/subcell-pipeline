# %%
import boto3
import getpass
import numpy as np
from preconfig import Preconfig

# %% [markdown]
# # 1. Upload config files to S3
# %%
# Preconfig class allows us to parse a template file and generate a list of config files.
# Two loops puts the generated config files for a given number of repeats in S3.
preconfig = Preconfig()
path_to_template = "../templates/vary_compress_rate.cym.tpl"
configs = preconfig.parse(path_to_template, {})
s3_client = boto3.client("s3")
bucket = "cytosim-working-bucket"
num_repeats = 5
job_names = []
buffered = np.empty((len(configs)), dtype=object)
for index, config in enumerate(configs):
    # removed '.cym'
    job_name = config[:-4]

    job_names.append(job_name)
    for repeat in range(num_repeats):
        opened_config = open(config, "rb")
        config_name = f"{job_name}/config/{job_name}_{repeat}.cym"
        s3_client.put_object(Bucket=bucket, Key=config_name, Body=opened_config)
job_names

# %% [markdown]
# # 2a. Specify job definition

# %%
job_definition_arn = "job_definition_arn"

# %% [markdown]
# # 2b. Create and register job definition

# %%
# Parameters for a job definition
from container_collection.batch.register_batch_job import register_batch_job
from container_collection.batch.make_batch_job import make_batch_job

job_definition_name = "karthikv_cytosim_vary_compress_rate"
image = "simularium/cytosim:latest"
vcpus = 1
memory = 7000
bucket_name = "s3://cytosim-working-bucket/"
simulation_name = ""

# %%
account = getpass.getpass()

# %%
# Creating job definitions with make_batch_job
# Submitting job definitions with register_batch_job
jobs = np.empty(len(configs))
job_definitions = np.empty((len(configs)), dtype=object)
for index in range(len(configs)):
    print(index)
    simulation_name = job_names[index]
    print(simulation_name)
    job_definition = make_batch_job(
    name=f"{str(job_names[index])}",
    image="simularium/cytosim:latest",
    vcpus=1,
    memory=7000,
    job_role_arn=f"arn:aws:iam::{account}:role/BatchJobRole",
    environment=[
    {"name": "BATCH_WORKING_URL", "value": "s3://cytosim-working-bucket/"},
    {"name": "FILE_SET_NAME", "value": f"{job_names[index]}"},
    {"name": "SIMULATION_TYPE", "value": "AWS"}
]
)

    registered_jd = register_batch_job(job_definition)
    job_definitions[index] = registered_jd


# %% [markdown]
# # 3. Submit job

# %%
# Submit batch job allows us to submit a batch job with a given job definition and job name.
from container_collection.batch.submit_batch_job import submit_batch_job

# %%
new_configs = configs[:4]
new_configs

# %%
# Parameters for our batch job [size indicates our desired number of repeats]
job_name = "cytosim-varycompressrate"
user = "karthikv"
queue = "general_on_demand"
size = 5

# %%
# Loop to submit our batch jobs [index * size for total number of simulations]
for index in range(len(configs)):
    if configs[index] != 'vary_compress_rate0005.cym':
        continue
    else: 
        print(index)
        print(f"{job_names}")
        submit_batch_job(
            name=f"{job_names[index]}",
            job_definition_arn=job_definitions[index],
            user=user,
            queue=queue,
            size=size,
        )


# %% [markdown]
# # 4. Monitor job status

# %%
# TODO: check job status, print progress bar
from container_collection.batch.check_batch_job import check_batch_job


# %% [markdown]
# # 5. Load results

# %%
from subcell_analysis.cytosim.post_process_cytosim import create_dataframes_for_repeats
import pandas as pd

# %%
bucket_name = "cytosim-working-bucket"
num_repeats = 5
num_velocities = 7
configs = [f"vary_compress_rate000{num}" for num in range(3, num_velocities)]


# %%
from pathlib import Path

# %%
save_folder = Path("../data/dataframes")

# %%
create_dataframes_for_repeats(bucket_name, num_repeats, configs, save_folder)

# %%
from subcell_analysis.compression_workflow_runner import (
    compression_metrics_workflow,
    plot_metric,
    plot_metric_list,
)
from subcell_analysis.compression_analysis import (
    COMPRESSIONMETRIC,
)

# %%
config_inds = [3, 4]
outputs = [[None] * num_repeats] * len(config_inds)

# %%
# TODO: Run metric calculations on repeats.
num_repeats = 5
outputs = [None] * num_repeats
for repeat in range(num_repeats):
    all_output = pd.read_csv(f"dataframes/actin-forces0_{repeat}.csv")
    outputs[repeat] = compression_metrics_workflow(
        all_output,
        [
            COMPRESSIONMETRIC.PEAK_ASYMMETRY,
            COMPRESSIONMETRIC.AVERAGE_PERP_DISTANCE,
            COMPRESSIONMETRIC.NON_COPLANARITY,
            COMPRESSIONMETRIC.TOTAL_FIBER_TWIST,
            COMPRESSIONMETRIC.SUM_BENDING_ENERGY,
        ],
    )

# %%
import matplotlib.pyplot as plt

config_ind = 0
metrics = [
    COMPRESSIONMETRIC.AVERAGE_PERP_DISTANCE,
    COMPRESSIONMETRIC.TOTAL_FIBER_TWIST,
    COMPRESSIONMETRIC.SUM_BENDING_ENERGY,
    COMPRESSIONMETRIC.PEAK_ASYMMETRY,
    COMPRESSIONMETRIC.NON_COPLANARITY,
]
for metric in metrics:
    fig, ax = plt.subplots()
    for repeat in range(num_repeats):
        metric_by_time = (
            outputs[config_ind][repeat].groupby(["time"])[metric.value].mean()
        )
        ax.plot(metric_by_time, label=f"config ind {config_ind} repeat {repeat}")
    ax.legend()
    ax.set_xlabel("time")
    ax.set_ylabel(metric.value)
    ax.set_title(f"{metric.value} by time")

# %% [markdown]
# ### Plot pacmap embedding

# %%
import numpy as np
import pandas as pd
from subcell_analysis.compression_analysis import get_pacmap_embedding
from pacmap import PaCMAP
from scipy import interpolate as spinterp

# %% [markdown]
# #### create k x t x n x 3 numpy array of fiber points

# %%
num_repeats = 5
df_list = []
configs = [3, 4]
for config in configs:
    for repeat in range(num_repeats):
        df = pd.read_csv(f"dataframes/actin_forces{config}_{repeat}.csv")
        df["repeat"] = repeat
        df["config"] = config
        df_list.append(df)
df_all = pd.concat(df_list)

# %%
df_all.to_csv("dataframes/all_fibers_configs_3_4.csv")

# %%
num_monomers = 100
num_timepoints = 101
all_config_repeats = []
cols_to_interp = ["xpos", "ypos", "zpos"]
for config, df_config in df_all.groupby("config"):
    for repeat, df_repeat in df_config.groupby("repeat"):
        all_times = []
        for time, df_time in df_repeat.groupby("time"):
            # interpolate xpos, ypos, zpos to num_monomers
            X = df_time[cols_to_interp].values
            t = np.linspace(0, 1, X.shape[0])
            F = spinterp.interp1d(t, X.T, bounds_error=False, fill_value="extrapolate")
            u = np.linspace(0, 1, num_monomers)
            all_times.append(F(u).T)
        all_times = np.array(all_times)
        interp_timepoints = np.around(
            len(all_times) / num_timepoints * np.arange(num_timepoints)
        ).astype(int)
        all_config_repeats.append(np.array(all_times)[interp_timepoints, :, :])
all_config_repeats = np.array(all_config_repeats)

# %%
embedding = PaCMAP(n_components=2, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0)

# %%
reshaped_metrics = all_config_repeats.reshape(all_config_repeats.shape[0], -1)

# %%
embed_pos = embedding.fit_transform(reshaped_metrics)

# %% [markdown]
# Plot embeddings

# %%
import matplotlib.pyplot as plt

# %%
fig, ax = plt.subplots()
configs = [3, 4]
for ct, config in enumerate(configs):
    inds = ct * num_repeats + np.arange(num_repeats)
    ax.scatter(embed_pos[inds, 0], embed_pos[inds, 1], label=f"config {config}")
ax.set_xlabel("embedding 1")
ax.set_ylabel("embedding 2")
ax.set_title("PaCMAP embedding of all repeats")
ax.legend()
plt.show()
