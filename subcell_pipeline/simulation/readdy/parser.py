"""Methods for parsing ReaDDy simulations."""

import os
from math import floor, log10
from typing import Optional, Union

import boto3
import numpy as np
import pandas as pd
from botocore.exceptions import ClientError
from io_collection.keys.check_key import check_key
from io_collection.save.save_dataframe import save_dataframe

from subcell_pipeline.simulation.readdy.loader import ReaddyLoader
from subcell_pipeline.simulation.readdy.post_processor import ReaddyPostProcessor

COLUMN_NAMES: list[str] = [
    "fiber_id",
    "xpos",
    "ypos",
    "zpos",
    "xforce",
    "yforce",
    "zforce",
    "segment_curvature",
    "time",
    "fiber_point",
]
"""Parsed tidy data column names."""

COLUMN_DTYPES: dict[str, Union[type[float], type[int]]] = {
    "fiber_id": int,
    "xpos": float,
    "ypos": float,
    "zpos": float,
    "xforce": float,
    "yforce": float,
    "zforce": float,
    "segment_curvature": float,
    "time": float,
    "fiber_point": int,
}
"""Parsed tidy data column data types."""

READDY_TIMESTEP: float = 0.1
"""Simulation timestep (in ns)."""

BOX_SIZE: np.ndarray = np.array(3 * [600.0])
"""Default simulation volume dimensions (x, y, z)."""

COMPRESSION_DISTANCE: float = 150.0
"""Total distance the fiber end was displaced in nm."""


def _download_s3_file(bucket: str, key: str, dest_path: str) -> Optional[str]:
    """
    Download file from S3 to local path.

    Parameters
    ----------
    bucket
        Name of S3 bucket.
    key
        Source key.
    dest_path
        Target local path.
    """

    s3_client = boto3.client("s3")

    if os.path.isfile(dest_path):
        return dest_path
    try:
        s3_client.download_file(bucket, key, dest_path)
        print(f"Downloaded [ {key} ] to [ {dest_path} ].")
        return dest_path
    except ClientError:
        print(f"!!! Failed to download {key}")
        return None


def download_readdy_hdf5(
    bucket: str,
    series_name: str,
    series_key: str,
    rep_ix: int,
    download_path: str,
) -> Optional[str]:
    """
    Download ReaDDy h5 files from S3 to local path.

    The ReaDDy Python package currently requires a local file path.

    Parameters
    ----------
    bucket
        Name of S3 bucket for input and output files.
    series_name
        Name of simulation series.
    series_key
        Combination of series and condition names.
    rep_ix
        Replicate index.
    download_path
        Path for downloading temporary h5 files.
    """

    if bucket.startswith("s3://"):
        bucket = bucket.replace("s3://", "")

    aws_h5_key = f"{series_name}/outputs/{series_key}_{rep_ix}.h5"
    local_h5_path = os.path.join(download_path, f"{series_key}_{rep_ix}.h5")
    return _download_s3_file(bucket, aws_h5_key, local_h5_path)


def parse_readdy_simulation_single_fiber_trajectory(
    bucket: str,
    series_name: str,
    series_key: str,
    rep_ix: int,
    n_timepoints: int,
    n_monomer_points: int,
    total_steps: int,
    temp_path: str,
    timestep: float = READDY_TIMESTEP,
) -> pd.DataFrame:
    """
    Parse ReaDDy trajectory data into tidy data format.

    Note that this methods assumes there is only one fiber in the simulation.

    Parameters
    ----------
    bucket
        Name of S3 bucket for input and output files.
    series_name
        Name of simulation.
    series_key
        Series key.
    rep_ix
        Replicate index.
    n_timepoints
        Number of equally spaced timepoints to sample.
    n_monomer_points
        Number of equally spaced monomer points to sample.
    total_steps
        Total number of steps for each given simulation.
    temp_path
        Path for saving temporary h5 files.
    timestep
        Simulation timestep (in ns).
    """

    h5_file_path = download_readdy_hdf5(
        bucket, series_name, series_key, rep_ix, temp_path
    )

    assert isinstance(h5_file_path, str)

    rep_id = rep_ix + 1
    pickle_key = f"{series_name}/data/{series_key}_{rep_id:06d}.pkl"
    time_inc = total_steps // n_timepoints

    readdy_loader = ReaddyLoader(
        h5_file_path=h5_file_path,
        time_inc=time_inc,
        timestep=timestep,
        pickle_location=bucket,
        pickle_key=pickle_key,
    )

    post_processor = ReaddyPostProcessor(readdy_loader.trajectory(), box_size=BOX_SIZE)

    times = post_processor.times()
    fiber_chain_ids = post_processor.linear_fiber_chain_ids(polymer_number_range=5)
    axis_positions, _ = post_processor.linear_fiber_axis_positions(fiber_chain_ids)

    fiber_points = post_processor.linear_fiber_control_points(
        axis_positions=axis_positions,
        n_points=n_monomer_points,
    )

    point_data: list[list[Union[str, int, float]]] = []
    for time_ix in range(len(fiber_points)):
        for pos_ix in range(fiber_points[0][0].shape[0]):
            point_data.append(
                [
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
                ]
            )

    # Combine all data into dataframe and update data types.
    dataframe = pd.DataFrame(point_data, columns=COLUMN_NAMES)
    dataframe = dataframe.astype(dtype=COLUMN_DTYPES)

    # Add placeholders for features not calculated in ReaDDy
    dataframe["force_magnitude"] = np.array(len(point_data) * [0.0])
    dataframe["segment_energy"] = np.array(len(point_data) * [0.0])

    return dataframe


def round_2_sig_figs(x: float) -> int:
    return int(round(x, -int(floor(log10(abs(0.1 * x))))))


def velocity_for_cond(condition_key: str) -> float:
    """'NNNN' -> NNN.N."""
    return float(condition_key[:3] + "." + condition_key[-1])


def parse_readdy_simulation_data(
    bucket: str,
    series_name: str,
    condition_keys: list[str],
    n_replicates: int,
    n_timepoints: int,
    n_monomer_points: int,
    compression: bool,
    temp_path: str,
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
    n_timepoints
        Number of equally spaced timepoints to sample.
    n_monomer_points
        Number of equally spaced monomer points to sample.
    compression
        If True, parse compressed trajectories,
        If False, parse baseline uncompressed trajectories.
    temp_path
        Path for saving temporary h5 files.
    """
    total_steps: dict[str, int] = {}
    if compression:
        total_steps = {
            cond: round_2_sig_figs(
                (COMPRESSION_DISTANCE * 1e-3 / velocity_for_cond(cond)) * 1e10
            )
            for cond in condition_keys
        }
    else:
        total_steps = {"": int(1e7)}

    for condition_key in condition_keys:
        series_key = f"{series_name}_{condition_key}" if condition_key else series_name

        for rep_ix in range(n_replicates):
            rep_id = rep_ix + 1
            dataframe_key = f"{series_name}/samples/{series_key}_{rep_id:06d}.csv"

            # Skip if dataframe file already exists.
            if check_key(bucket, dataframe_key):
                print(f"Dataframe [ { dataframe_key } ] already exists. Skipping.")
                continue

            print(f"Parsing data for [ {condition_key} ] replicate [ {rep_ix} ]")

            data = parse_readdy_simulation_single_fiber_trajectory(
                bucket,
                series_name,
                series_key,
                rep_ix,
                n_timepoints,
                n_monomer_points,
                total_steps[condition_key],
                temp_path,
            )

            save_dataframe(bucket, dataframe_key, data, index=False)
