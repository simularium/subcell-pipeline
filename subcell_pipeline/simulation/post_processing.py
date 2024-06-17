"""Methods for processing simulations."""

import numpy as np
import pandas as pd
from io_collection.keys.check_key import check_key
from io_collection.load.load_dataframe import load_dataframe
from io_collection.save.save_dataframe import save_dataframe

SAMPLE_COLUMNS = ["xpos", "ypos", "zpos"]


def sample_simulation_data(
    bucket: str,
    series_name: str,
    condition_keys: list[str],
    random_seeds: list[int],
    n_timepoints: int,
    n_monomer_points: int,
) -> None:
    """
    Sample simulation data for select conditions and seeds at given resolution.

    Parameters
    ----------
    bucket
        Name of S3 bucket for input and output files.
    series_name
        Name of simulation series.
    condition_keys
        List of condition keys.
    random_seeds
        Random seeds for simulations.
    n_timepoints
        Number of equally spaced timepoints to sample.
    n_monomer_points
        Number of equally spaced monomer points to sample.
    """

    for condition_key in condition_keys:
        series_key = f"{series_name}_{condition_key}" if condition_key else series_name

        for seed in random_seeds:
            data_key = f"{series_name}/data/{series_key}_{seed:06d}.csv"
            sampled_key = f"{series_name}/samples/{series_key}_{seed:06d}.csv"

            # Skip if dataframe file already exists.
            if check_key(bucket, sampled_key):
                print(
                    f"Sampled dataframe [ { sampled_key } ] already exists. Skipping."
                )
                continue

            print(f"Sampling data for [ {condition_key} ] seed [ {seed} ]")

            full_data = load_dataframe(bucket, data_key)
            sampled_data = sample_simulation_data_points(
                full_data, n_timepoints, n_monomer_points
            )

            save_dataframe(bucket, sampled_key, sampled_data, index=False)


def sample_simulation_data_points(
    data: pd.DataFrame,
    n_timepoints: int,
    n_monomer_points: int,
    sampled_columns: list[str] = SAMPLE_COLUMNS,
) -> pd.DataFrame:
    """
    Sample selected columns from simulation data at given resolution.

    Parameters
    ----------
    data
        Full simulation data.
    n_timepoints
        Number of equally spaced timepoints to sample.
    n_monomer_points
        Number of equally spaced monomer points to sample.
    sampled_columns
        List of column names to sample.

    Returns
    -------
    :
        Sampled simulation data.
    """

    all_sampled_points = []

    unique_timepoints = data["time"].unique()
    n_unique_timepoints = unique_timepoints.size

    time_indices = np.rint(
        np.interp(
            np.linspace(0, 1, n_timepoints),
            np.linspace(0, 1, n_unique_timepoints),
            np.arange(n_unique_timepoints),
        )
    ).astype(int)

    time_data = data[data["time"].isin(unique_timepoints[time_indices])]

    for time, group in time_data.groupby("time"):
        sampled_points = pd.DataFrame()
        sampled_points["monomer_ids"] = np.arange(n_monomer_points)
        sampled_points["time"] = time

        for column in sampled_columns:
            sampled_points[column] = np.interp(
                np.linspace(0, 1, n_monomer_points),
                np.linspace(0, 1, group.shape[0]),
                group[column].values,
            )

        all_sampled_points.append(sampled_points)

    return pd.concat(all_sampled_points)
