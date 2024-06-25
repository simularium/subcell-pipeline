"""Methods for parsing ReaDDy simulations."""

import os
from typing import List, Union, Tuple

import boto3
from botocore.exceptions import ClientError
import numpy as np
import pandas as pd
from io_collection.keys.check_key import check_key
from io_collection.save.save_dataframe import save_dataframe

from .loader import ReaddyLoader
from .post_processor import ReaddyPostProcessor
from ..constants import COLUMN_NAMES, COLUMN_DTYPES
from .constants import (
    ACTIN_START_PARTICLE_PHRASE, 
    ACTIN_PARTICLE_TYPES,
    IDEAL_ACTIN_POSITIONS,
    IDEAL_ACTIN_VECTOR_TO_AXIS,
)


LOCAL_DOWNLOADS_PATH = "aws_downloads/"
READDY_TIMESTEP = 0.1  # ns
READDY_TOTAL_STEPS = {
    "ACTIN_NO_COMPRESSION" : 1e7,
    "ACTIN_COMPRESSION_VELOCITY_0047" : 3.2e8,
    "ACTIN_COMPRESSION_VELOCITY_0150" : 1e8, 
    "ACTIN_COMPRESSION_VELOCITY_0470" : 3.2e7, 
    "ACTIN_COMPRESSION_VELOCITY_1500" : 1e7,
}
BOX_SIZE = np.array(3 * [600.0])


s3_client = boto3.client("s3")


def _make_download_dir():
    if not os.path.isdir(LOCAL_DOWNLOADS_PATH):
        os.makedirs(LOCAL_DOWNLOADS_PATH)


def _download_s3_file(bucket_name, key, dest_path) -> bool:
    """
    Download files from S3 (skip files that already exist)
    
    (ReaDDy Python pkg currently requires a local file path)
    """
    if os.path.isfile(dest_path):
        # already downloaded
        return False
    try:
        s3_client.download_file(
            bucket_name,
            key,
            dest_path,
        )
        print(f"Downloaded {dest_path}")
        return True
    except ClientError:
        print(f"!!! Failed to download {key}")
        return False


def _load_readdy_fiber_points(
    series_key: str, 
    rep_ix: int,
    n_timepoints: int,
    n_monomer_points: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a ReaDDy trajectory, calculate the polymer trace from 
    the monomer particle positions (using measurements from x-ray crystallography), 
    and resample to get the requested number of points 
    along each linear fiber at each timestep.
    """
    h5_file_path = os.path.join(LOCAL_DOWNLOADS_PATH, f"{series_key}_{rep_ix}.h5")
    time_inc = READDY_TOTAL_STEPS[series_key] / n_timepoints
    readdy_loader = ReaddyLoader(
        h5_file_path=str(h5_file_path),
        time_inc=time_inc,
        timestep=READDY_TIMESTEP,
    )
    readdy_post_processor = ReaddyPostProcessor(
        readdy_loader.trajectory(),
        box_size=BOX_SIZE,
    )
    fiber_chain_ids = readdy_post_processor.linear_fiber_chain_ids(
        start_particle_phrases=[ACTIN_START_PARTICLE_PHRASE],
        other_particle_types=ACTIN_PARTICLE_TYPES,
        polymer_number_range=5,
    )
    axis_positions, _ = readdy_post_processor.linear_fiber_axis_positions(
        fiber_chain_ids=fiber_chain_ids,
        ideal_positions=IDEAL_ACTIN_POSITIONS,
        ideal_vector_to_axis=IDEAL_ACTIN_VECTOR_TO_AXIS,
    )
    fiber_points = readdy_post_processor.linear_fiber_control_points(
        axis_positions=axis_positions,
        n_points=n_monomer_points,
    )
    times = readdy_post_processor.times()
    return np.array(fiber_points), times


def _parse_readdy_simulation_trajectory(
    series_key: str, 
    rep_ix: int,
    n_timepoints: int,
    n_monomer_points: int,
) -> pd.DataFrame:
    """
    Parse ReaDDy trajectory data into tidy data format.
    (Assume one fiber)
    """
    fiber_points, times = _load_readdy_fiber_points(
        series_key, rep_ix, n_timepoints, n_monomer_points
    )
    
    point_data: list[list[Union[str, int, float]]] = []
    for time_ix in range(fiber_points.shape[0]):
        for pos_ix in range(fiber_points.shape[2]):
            point_data.append([
                1,  # fiber_id
                fiber_points[time_ix][0][pos_ix][0],  # xpos
                fiber_points[time_ix][0][pos_ix][1],  # ypos
                fiber_points[time_ix][0][pos_ix][2],  # zpos
                0.0,  # xforce
                0.0,  # yforce
                0.0,  # zforce
                0.0,  # segment_curvature
                times[time_ix],  # time
                pos_ix,  # fiber_point
            ])

    # Combine all data into dataframe and update data types.
    dataframe = pd.DataFrame(point_data, columns=COLUMN_NAMES)
    dataframe = dataframe.astype(dtype=COLUMN_DTYPES)

    # Add placeholders for features not calculated in ReaDDy
    dataframe["force_magnitude"] = np.array(len(point_data) * [0.0])
    dataframe["segment_energy"] = np.array(len(point_data) * [0.0])

    return dataframe


def parse_readdy_simulation_data(
    bucket: str,
    series_name: str,
    condition_keys: list[str],
    n_replicates: int,
    n_timepoints: int, 
    n_monomer_points: int,
) -> None:
    """
    Parse ReaDDy simulation data for select conditions and replicates.

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
            dataframe_key = f"{series_name}/data/{series_key}_{rep_ix}.csv"

            # Skip if dataframe file already exists.
            if check_key(bucket, dataframe_key):
                print(f"Dataframe [ { dataframe_key } ] already exists. Skipping.")
                continue

            print(f"Parsing data for [ {condition_key} ] replicate [ {rep_ix} ]")

            aws_h5_key = f"{series_name}/outputs/{series_key}_{rep_ix}.h5"
            local_h5_key = os.path.join(LOCAL_DOWNLOADS_PATH, f"{series_key}_{rep_ix}.h5")
            _download_s3_file(bucket, aws_h5_key, local_h5_key)
            
            data = _parse_readdy_simulation_trajectory(
                series_key, rep_ix, n_timepoints, n_monomer_points
            )

            save_dataframe(bucket, dataframe_key, data, index=False)
