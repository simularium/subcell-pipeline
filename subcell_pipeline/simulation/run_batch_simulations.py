import io
import re

import boto3
from container_collection.batch.get_batch_logs import get_batch_logs
from container_collection.batch.make_batch_job import make_batch_job
from container_collection.batch.register_batch_job import register_batch_job
from container_collection.batch.submit_batch_job import submit_batch_job


def generate_configs_from_file(
    bucket: str,
    series_name: str,
    timestamp: str,
    random_seeds: list[int],
    config_file: str,
) -> None:
    """
    Generate configs from given file for each seed and save to S3 bucket.

    Parameters
    ----------
    bucket
        Name of S3 bucket for input and output files.
    series_name
        Name of simulation series.
    timestamp
        Current timestamp used to organize input and outfile files.
    random_seeds
        Random seeds for simulations.
    config_file
       Path to the config file.
    """

    s3_client = boto3.client("s3")

    with open(config_file) as f:
        contents = f.read()

    for index, seed in enumerate(random_seeds):
        config_key = f"{series_name}/{timestamp}/configs/{series_name}_{index}.cym"
        config_contents = contents.replace("{{RANDOM_SEED}}", str(seed))
        print(f"Saving config for for seed {seed} to [ {config_key}]")

        with io.BytesIO() as buffer:
            buffer.write(config_contents.encode("utf-8"))
            s3_client.put_object(Bucket=bucket, Key=config_key, Body=buffer.getvalue())


def generate_configs_from_template(
    bucket: str,
    series_name: str,
    timestamp: str,
    random_seeds: list[int],
    config_files: list[str],
    pattern: str,
    key_map: dict[str, str],
) -> list[str]:
    """
    Generate configs for each given file for each seed and save to S3 bucket.

    Parameters
    ----------
    bucket
        Name of S3 bucket for input and output files.
    series_name
        Name of simulation series.
    timestamp
        Current timestamp used to organize input and outfile files.
    random_seeds
        Random seeds for simulations.
    config_files
        Path to the config files.
    pattern
        Regex pattern to find config condition value.
    key_map
        Map of condition values to file keys.

    Returns
    -------
    :
        List of config groups.
    """

    group_keys = []
    s3_client = boto3.client("s3")

    for config_file in config_files:
        with open(config_file) as f:
            contents = f.read()

        match = re.findall(pattern, contents)[0].strip()
        match_key = key_map[match]

        group_key = f"{series_name}_{match_key}"
        group_keys.append(group_key)

        for index, seed in enumerate(random_seeds):
            config_key = f"{series_name}/{timestamp}/configs/{group_key}_{index}.cym"
            config_contents = contents.replace("{{RANDOM_SEED}}", str(seed))
            print(f"Saving config for [ {match} ] for seed {seed} to [ {config_key}]")

            with io.BytesIO() as buffer:
                buffer.write(config_contents.encode("utf-8"))
                s3_client.put_object(
                    Bucket=bucket, Key=config_key, Body=buffer.getvalue()
                )

    return group_keys


def register_and_run_simulations(
    bucket: str,
    series_name: str,
    timestamp: str,
    group_keys: list[str],
    aws_account: str,
    aws_region: str,
    aws_user: str,
    image: str,
    vcpus: int,
    memory: int,
    job_queue: str,
    job_size: int,
) -> list[str]:
    """
    Register job definitions and submit jobs to AWS Batch.

    Parameters
    ----------
    bucket
        Name of S3 bucket for input and output files.
    series_name
        Name of simulation series.
    timestamp
        Current timestamp used to organize input and outfile files.
    group_keys
        List of config group keys.
    aws_account : str
        AWS account number.
    aws_region : str
        AWS region.
    aws_user : str
        User name prefix for job name and image.
    image : str
        Image name and version.
    vcpus : int
        Number of vCPUs for each job.
    memory : int
        Memory for each job.
    job_queue : str
        Job queue.
    job_size : int
        Job array size.

    Returns
    -------
    :
        List of job ARNs.
    """

    boto3.setup_default_session(region_name=aws_region)

    all_job_arns: list[str] = []
    registry = f"{aws_account}.dkr.ecr.{aws_region}.amazonaws.com"
    job_key = f"s3://{bucket}/{series_name}/{timestamp}/"

    for group_key in group_keys:
        job_definition = make_batch_job(
            f"{aws_user}_{group_key}",
            f"{registry}/{aws_user}/{image}",
            vcpus,
            memory,
            [
                {"name": "SIMULATION_TYPE", "value": "AWS"},
                {"name": "BATCH_WORKING_URL", "value": job_key},
                {"name": "FILE_SET_NAME", "value": group_key},
            ],
            f"arn:aws:iam::{aws_account}:role/BatchJobRole",
        )

        job_definition_arn = register_batch_job(job_definition)

        print(f"Create job definition [ {job_definition_arn} ] for [ {group_key} ]")

        job_arns = submit_batch_job(
            group_key,
            job_definition_arn,
            aws_user,
            job_queue,
            job_size,
        )

        for job_arn in job_arns:
            print(f"Submitted job [ {job_arn} ] for [ {group_key} ]")

        all_job_arns.extend(job_arns)

    return all_job_arns


def check_and_save_job_logs(
    bucket: str, series_name: str, job_arns: list[str], aws_region: str
):
    """
    Check job status and save CloudWatch logs for successfully completed jobs.

    Parameters
    ----------
    bucket
        Name of S3 bucket for input and output files.
    series_name
        Name of simulation series.
    job_arns
        List of job ARNs.
    aws_region : str
        AWS region.
    """

    boto3.setup_default_session(region_name=aws_region)

    s3_client = boto3.client("s3")
    batch_client = boto3.client("batch")

    responses = batch_client.describe_jobs(jobs=job_arns)["jobs"]

    for response in responses:
        if responses[0]["status"] != "SUCCEEDED":
            print(
                f"Job [ {response['jobId']} ] has status [ {responses[0]['status']} ]"
            )
        else:
            group_key = next(
                item
                for item in response["container"]["environment"]
                if item["name"] == "FILE_SET_NAME"
            )["value"]
            log_key = f"{series_name}/logs/{group_key}_{response['jobId']}.log"

            print(f"Saving logs for job [ {response['jobId']} ] to [ {log_key}]")

            logs = get_batch_logs(response["jobArn"], ".")

            with io.BytesIO() as buffer:
                buffer.write(logs.encode("utf-8"))
                s3_client.put_object(Bucket=bucket, Key=log_key, Body=buffer.getvalue())
