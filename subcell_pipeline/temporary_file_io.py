
"""Methods for parsing ReaDDy simulations."""

import os

import boto3
from botocore.exceptions import ClientError

from .constants import LOCAL_DOWNLOADS_PATH


s3_client = boto3.client("s3")


def _make_download_dir() -> None:
    if not os.path.isdir(LOCAL_DOWNLOADS_PATH):
        os.makedirs(LOCAL_DOWNLOADS_PATH)


def _download_s3_file(
    bucket: str, 
    key: str, 
    dest_path: str,
) -> bool:
    """
    Download files from S3
    """
    if os.path.isfile(dest_path):
        # already downloaded
        return False
    try:
        s3_client.download_file(
            bucket,
            key,
            dest_path,
        )
        print(f"Downloaded {dest_path}")
        return True
    except ClientError:
        print(f"!!! Failed to download {key}")
        return False


def download_readdy_hdf5(
    bucket: str, 
    series_name: str, 
    series_key: str,
    rep_ix: int,
) -> bool:
    """
    Download files from S3
    (ReaDDy Python pkg currently requires a local file path)

    Parameters
    ----------
    bucket
        Name of S3 bucket for input and output files.
    series_name
        Name of simulation series.
    series_key
        Combination of series and condition names.
    replicate_ix
        Replicate index.
    """
    aws_h5_key = f"{series_name}/outputs/{series_key}_{rep_ix}.h5"
    local_h5_path = os.path.join(LOCAL_DOWNLOADS_PATH, f"{series_key}_{rep_ix}.h5")
    return _download_s3_file(bucket, aws_h5_key, local_h5_path)


def download_all_readdy_outputs(
    bucket: str,
    series_name: str,
    condition_keys: list[str],
    n_replicates: int,
) -> None:
    """
    Download ReaDDy simulation outputs for all conditions and replicates.

    Parameters
    ----------
    bucket
        Name of S3 bucket for input and output files.
    series_name
        Name of simulation series.
    condition_keys
        List of condition keys.
    n_replicates
        Number of simulation replicates.
    """
    _make_download_dir()
    
    for condition_key in condition_keys:
        series_key = f"{series_name}_{condition_key}" if condition_key else series_name

        for rep_ix in range(n_replicates):

            local_h5_path = os.path.join(LOCAL_DOWNLOADS_PATH, f"{series_key}_{rep_ix}.h5")
            
            # Skip if file already exists.
            if os.path.isfile(local_h5_path):
                print(f"ReaDDy file [ { local_h5_path } ] already downloaded. Skipping.")
                continue
            
            aws_h5_key = f"{series_name}/outputs/{series_key}_{rep_ix}.h5"
            download_s3_file(bucket, aws_h5_key, local_h5_path)

            print(f"Downloaded data for [ {condition_key} ] replicate [ {rep_ix} ]")


def upload_file_to_s3(bucket: str, src_path: str, s3_path: str) -> bool:
    """
    Upload a file to an S3 bucket

    Parameters
    ----------
    bucket
        Name of S3 bucket for input and output files.
    src_path
        Local path to file to upload
    s3_path
        S3 key for where to save in the bucket
    """
    if not os.path.isfile(src_path):
        print(f"!!! File does not exist to upload {src_path}")
        return False
    try:
        s3_client.upload_file(src_path, bucket, s3_path)
        print(f"Uploaded to {s3_path}")
        return True
    except ClientError:
        print(f"!!! Failed to upload {src_path}")
        return False
