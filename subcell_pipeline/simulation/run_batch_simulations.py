import io
import re

import boto3
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
) -> None:
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

        print(f"Create job definition [ {job_definition_arn } ] for [ {group_key} ]")

        job_arns = submit_batch_job(
            group_key,
            job_definition_arn,
            aws_user,
            job_queue,
            job_size,
        )

        for job_arn in job_arns:
            print(f"Submitted job [ {job_arn } ] for [ {group_key} ]")
