# %%
import boto3
import getpass
import numpy as np
from preconfig import Preconfig

# %%
preconfig = Preconfig()
path_to_config = "../configs/free_barbed_end_final.cym"
config = "free_barbed_end_final.cym"
s3_client = boto3.client("s3")
bucket = "cytosim-working-bucket"
num_repeats = 5
job_names = []
# %%
for repeat in range(num_repeats):
    opened_config = open (path_to_config, "rb")
    job_name = config[:-4]
    config_name = f"{job_name}/config/{job_name}_{repeat}.cym"
    s3_client.put_object(Bucket=bucket, Key=config_name, Body=opened_config)
# %%
job_definition_arn = "job_definition_arn"
from container_collection.batch.register_batch_job import register_batch_job
from container_collection.batch.make_batch_job import make_batch_job

job_definition_name = "karthikv_cytosim"+config[:-4]
image = "simularium/cytosim:latest"
vcpus = 1
memory = 7000
bucket_name = "s3://cytosim-working-bucket/"
simulation_name = ""
account = getpass.getpass()
# %%

job_definition = make_batch_job(
    name="free_barbed_end_final",
    image="simularium/cytosim:latest",
    vcpus=1,
    memory=7000,
    job_role_arn=f"arn:aws:iam::{account}:role/BatchJobRole",
    environment=[
    {"name": "BATCH_WORKING_URL", "value": "s3://cytosim-working-bucket/"},
    {"name": "FILE_SET_NAME", "value": f"{config[:-4]}"},
    {"name": "SIMULATION_TYPE", "value": "AWS"}
    
]
)

registered_jd = register_batch_job(job_definition)

# %%
# Submit batch job allows us to submit a batch job with a given job definition and job name.
from container_collection.batch.submit_batch_job import submit_batch_job
# %%
# Parameters for our batch job [size indicates our desired number of repeats]
job_name = "free_barbed_end_final"
user = "karthikv"
queue = "general_on_demand"
size = 5

# %%
print(f"{job_name}")
submit_batch_job(
    name=f"{job_name}",
    job_definition_arn=registered_jd,
    user=user,
    queue=queue,
    size=size,
)
# %%
