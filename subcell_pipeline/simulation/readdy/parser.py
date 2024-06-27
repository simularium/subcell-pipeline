"""Methods for parsing ReaDDy simulations."""

import os
from typing import List, Union, Tuple

import numpy as np
import pandas as pd
from io_collection.keys.check_key import check_key
from io_collection.save.save_dataframe import save_dataframe

from .loader import ReaddyLoader
from .post_processor import ReaddyPostProcessor
from ...constants import (
    COLUMN_NAMES, 
    COLUMN_DTYPES, 
    BOX_SIZE,
    READDY_TIMESTEP,
    READDY_TOTAL_STEPS,
    ACTIN_START_PARTICLE_PHRASE, 
    ACTIN_PARTICLE_TYPES,
    IDEAL_ACTIN_POSITIONS,
    IDEAL_ACTIN_VECTOR_TO_AXIS,
    LOCAL_DOWNLOADS_PATH,
)
from ...temporary_file_io import download_readdy_hdf5


def readdy_post_processor(
    bucket: str,
    series_name: str, 
    series_key: str, 
    rep_ix: int,
    n_timepoints: int,
) -> ReaddyPostProcessor:
    """
    Load a ReaddyPostProcessor from the specified ReaDDy trajectory.
    (Load from a pickle if it exists.)

    Parameters
    ----------
    bucket
        Name of S3 bucket for input and output files.
    series_name
        Name of simulation series.
    series_key
        Name of simulation series plus condition_key if applicable.
    rep_ix
        Replicate index.
    n_timepoints
        Number of timepoints to visualize.
    """
    h5_file_path = os.path.join(LOCAL_DOWNLOADS_PATH, f"{series_key}_{rep_ix}.h5")
    rep_id = rep_ix + 1
    pickle_key = f"{series_name}/data/{series_key}_{rep_id:06d}.pkl"
    time_inc = READDY_TOTAL_STEPS[series_key] / n_timepoints
    readdy_loader = ReaddyLoader(
        h5_file_path=str(h5_file_path),
        time_inc=time_inc,
        timestep=READDY_TIMESTEP,
        pickle_location=bucket,
        pickle_key=pickle_key,
    )
    return ReaddyPostProcessor(
        readdy_loader.trajectory(),  # this will load from a pickle if it exists
        box_size=BOX_SIZE,
    )


def load_readdy_fiber_points(
    bucket: str,
    series_name: str, 
    series_key: str, 
    rep_ix: int,
    n_timepoints: int,
    n_monomer_points: int,
) -> Tuple[List[List[List[int]]], List[List[np.ndarray]], np.ndarray, np.ndarray]:
    """
    Load a ReaDDy trajectory, calculate the polymer trace from 
    the monomer particle positions (using measurements from x-ray crystallography), 
    and resample to get the requested number of points 
    along each linear fiber at each timestep.

    Parameters
    ----------
    bucket
        Name of S3 bucket for input and output files.
    series_name
        Name of simulation series.
    series_key
        Name of simulation series plus condition_key if applicable.
    rep_ix
        Replicate index.
    n_timepoints
        Number of timepoints to visualize.
    n_monomer_points
        Number of control points for each polymer trace.

    Returns
    -------
    readdy_post_processor: ReaddyPostProcessor
        The ReaddyPostProcessor loaded with this trajectory 
        in case it is needed for additional analysis.
    fiber_chain_ids: List[List[List[int]]]
        Particle IDs for particles in each linear fiber at each timestep
        that match the axis_positions list.
    axis_positions: List[List[np.ndarray (shape = n x 3)]]
        List of lists of arrays containing the x,y,z positions
        of the closest point on the fiber axis to the position
        of each particle in each fiber at each time.
    fiber_points: np.ndarray (shape = n_timepoints x n_fibers (1) x n x 3)
        Array containing the x,y,z positions
        of control points for each fiber at each time.
    times: np.ndarray (shape = n_timepoints)
        Simulation time at each timestep.
    """
    readdy_post_processor = readdy_post_processor(
        bucket, series_name, series_key, rep_ix, n_timepoints
    )
    fiber_chain_ids = readdy_post_processor.linear_fiber_chain_ids(
        start_particle_phrases=[ACTIN_START_PARTICLE_PHRASE],
        other_particle_types=ACTIN_PARTICLE_TYPES,
        polymer_number_range=5,
    )
    axis_positions, fiber_chain_ids = readdy_post_processor.linear_fiber_axis_positions(
        fiber_chain_ids=fiber_chain_ids,
        ideal_positions=IDEAL_ACTIN_POSITIONS,
        ideal_vector_to_axis=IDEAL_ACTIN_VECTOR_TO_AXIS,
    )
    fiber_points = readdy_post_processor.linear_fiber_control_points(
        axis_positions=axis_positions,
        n_points=n_monomer_points,
    )
    times = readdy_post_processor.times()
    return readdy_post_processor, fiber_chain_ids, axis_positions, np.array(fiber_points), times


def _parse_readdy_simulation_trajectory(
    bucket: str,
    series_name: str, 
    series_key: str, 
    rep_ix: int,
    n_timepoints: int,
    n_monomer_points: int,
) -> pd.DataFrame:
    """
    Parse ReaDDy trajectory data into tidy data format.
    (Assume one fiber)
    """
    _, _, _, fiber_points, times = load_readdy_fiber_points(
        bucket, series_name, series_key, rep_ix, n_timepoints, n_monomer_points
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
    for condition_key in condition_keys:
        series_key = f"{series_name}_{condition_key}" if condition_key else series_name

        for rep_ix in range(n_replicates):
            rep_id = rep_ix + 1
            dataframe_key = f"{series_name}/data/{series_key}_{rep_id:06d}.csv"

            # Skip if dataframe file already exists.
            if check_key(bucket, dataframe_key):
                print(f"Dataframe [ { dataframe_key } ] already exists. Skipping.")
                continue

            print(f"Parsing data for [ {condition_key} ] replicate [ {rep_ix} ]")
            
            download_readdy_hdf5(bucket, series_name, series_key, rep_ix)
            
            data = _parse_readdy_simulation_trajectory(
                bucket, series_name, series_key, rep_ix, n_timepoints, n_monomer_points
            )

            save_dataframe(bucket, dataframe_key, data, index=False)
