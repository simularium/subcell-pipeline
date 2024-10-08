"""Methods for analyzing tomography data."""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from io_collection.keys.check_key import check_key
from io_collection.load.load_dataframe import load_dataframe
from io_collection.save.save_dataframe import save_dataframe
from io_collection.save.save_figure import save_figure

TOMOGRAPHY_SAMPLE_COLUMNS: list[str] = ["xpos", "ypos", "zpos"]
"""Columns names used when sampling tomography data."""


def test_consecutive_segment_angles(polymer_trace: np.ndarray) -> bool:
    """
    Test if all angles between consecutive segments of a polymer trace are less
    than 90 degrees.

    Parameters
    ----------
    polymer_trace
        A 2D array where each row is a point in 3D space.

    Returns
    -------
    :
        True if all consecutive angles are less than 90 degrees, False
        otherwise.
    """
    vectors = polymer_trace[1:] - polymer_trace[:-1]

    vectors /= np.linalg.norm(vectors, axis=1)[:, np.newaxis]
    dot_products = np.dot(vectors[1:], vectors[:-1].T)

    return np.all(dot_products > 0).item()


def read_tomography_data(file: str, label: str = "fil") -> pd.DataFrame:
    """
    Read tomography data from file as dataframe.

    Parameters
    ----------
    file
        Path to tomography data.
    label
        Label for the filament id column.

    Returns
    -------
    :
        Dataframe of tomography data.
    """

    coordinates = pd.read_table(file, delim_whitespace=True)

    if len(coordinates.columns) == 4:
        coordinates.columns = [label, "xpos", "ypos", "zpos"]
    elif len(coordinates.columns) == 5:
        coordinates.columns = ["object", label, "xpos", "ypos", "zpos"]
    else:
        print(f"Data file [ {file} ] has an unexpected number of columns")

    return coordinates


def rescale_tomography_data(data: pd.DataFrame, scale_factor: float = 1.0) -> None:
    """
    Rescale tomography data from pixels to um.

    Parameters
    ----------
    data
        Unscaled tomography data.
    scale_factor
        Data scaling factor (pixels to um).
    """

    data["xpos"] = data["xpos"] * scale_factor
    data["ypos"] = data["ypos"] * scale_factor
    data["zpos"] = data["zpos"] * scale_factor


def get_branched_tomography_data(
    bucket: str,
    name: str,
    repository: str,
    datasets: list[tuple[str, str]],
    scale_factor: float = 1.0,
) -> pd.DataFrame:
    """
    Load or create merged branched actin tomography data for given datasets.

    Parameters
    ----------
    bucket
        Name of S3 bucket for input and output files.
    name
        Name of dataset.
    repository
        Data repository for downloading tomography data.
    datasets
        Folders and names of branched actin datasets.
    scale_factor
        Data scaling factor (pixels to um).

    Returns
    -------
    :
        Merged branched tomography data.
    """

    return get_tomography_data(
        bucket, name, repository, datasets, "branched", scale_factor
    )


def get_unbranched_tomography_data(
    bucket: str,
    name: str,
    repository: str,
    datasets: list[tuple[str, str]],
    scale_factor: float = 1.0,
) -> pd.DataFrame:
    """
    Load or create merged unbranched actin tomography data for given datasets.

    Parameters
    ----------
    bucket
        Name of S3 bucket for input and output files.
    name
        Name of dataset.
    repository
        Data repository for downloading tomography data.
    datasets
        Folders and names of branched actin datasets.
    scale_factor
        Data scaling factor (pixels to um).

    Returns
    -------
    :
        Merged unbranched tomography data.
    """

    return get_tomography_data(
        bucket, name, repository, datasets, "unbranched", scale_factor
    )


def get_tomography_data(
    bucket: str,
    name: str,
    repository: str,
    datasets: list[tuple[str, str]],
    group: str,
    scale_factor: float = 1.0,
) -> pd.DataFrame:
    """
    Load or create merged tomography data for given datasets.

    Parameters
    ----------
    bucket
        Name of S3 bucket for input and output files.
    name
        Name of dataset.
    repository
        Data repository for downloading tomography data.
    datasets
        Folders and names of branched actin datasets.
    group
        Actin filament group ("branched" or "unbranched").
    scale_factor
        Data scaling factor (pixels to um).

    Returns
    -------
    :
        Merged tomography data.
    """

    data_key = f"{name}/{name}_coordinates_{group}.csv"

    if check_key(bucket, data_key):
        print(f"Loading existing combined tomogram data from [ { data_key } ]")
        return load_dataframe(bucket, data_key)
    else:
        all_tomogram_dfs = []

        for folder, name in datasets:
            print(f"Loading tomogram data for [ { name } ]")
            tomogram_file = f"{repository}/{folder}/{group.title()}Actin_{name}.txt"
            tomogram_df = read_tomography_data(tomogram_file)
            tomogram_df["dataset"] = name
            tomogram_df["id"] = tomogram_df["fil"].apply(
                lambda row, name=name: f"{row:02d}_{name}"
            )
            rescale_tomography_data(tomogram_df, scale_factor)
            all_tomogram_dfs.append(tomogram_df)

        all_tomogram_df = pd.concat(all_tomogram_dfs)

        print(f"Saving combined tomogram data to [ { data_key } ]")
        save_dataframe(bucket, data_key, all_tomogram_df, index=False)

        return all_tomogram_df


def sample_tomography_data(
    data: pd.DataFrame,
    save_location: str,
    save_key: str,
    n_monomer_points: int,
    minimum_points: int,
    sampled_columns: list[str] = TOMOGRAPHY_SAMPLE_COLUMNS,
    recalculate: bool = False,
) -> pd.DataFrame:
    """
    Sample selected columns from tomography data at given resolution.

    Parameters
    ----------
    data
        Tomography data to sample.
    save_location
        Location to save sampled data.
    save_key
        File key for sampled data.
    n_monomer_points
        Number of equally spaced monomer points to sample.
    minimum_points
        Minimum number of points for valid fiber.
    sampled_columns
        List of column names to sample.
    recalculate
        True to recalculate the sampled tomography data, False otherwise.

    Returns
    -------
    :
        Sampled tomography data.
    """

    if check_key(save_location, save_key) and not recalculate:
        print(f"Loading existing sampled tomogram data from [ { save_key } ]")
        return load_dataframe(save_location, save_key)
    else:
        all_sampled_points = []

        # TODO sort experimental samples in order along the fiber before resampling
        # (see simularium visualization)
        for fiber_id, group in data.groupby("id"):
            if len(group) < minimum_points:
                continue

            sampled_points = pd.DataFrame()
            sampled_points["monomer_ids"] = np.arange(n_monomer_points)
            sampled_points["dataset"] = group["dataset"].unique()[0]
            sampled_points["id"] = fiber_id

            for column in sampled_columns:
                sampled_points[column] = np.interp(
                    np.linspace(0, 1, n_monomer_points),
                    np.linspace(0, 1, group.shape[0]),
                    group[column].to_numpy(),
                )

            sampled_points["ordered"] = test_consecutive_segment_angles(
                sampled_points[sampled_columns].to_numpy()
            )

            all_sampled_points.append(sampled_points)

        all_sampled_df = pd.concat(all_sampled_points)

        print(f"Saving sampled tomogram data to [ { save_key } ]")
        save_dataframe(save_location, save_key, all_sampled_df, index=False)

        return all_sampled_df


def plot_tomography_data_by_dataset(
    data: pd.DataFrame,
    save_location: Optional[str] = None,
    save_key_template: str = "tomography_data_%s.png",
) -> None:
    """
    Plot tomography data for each dataset.

    Parameters
    ----------
    data
        Tomography data.
    save_location
        Location for output file (local path or S3 bucket).
    save_key_template
        Name key template for output file.
    """

    for dataset, group in data.groupby("dataset"):
        figure, ax = plt.subplots(1, 3, figsize=(6, 2))
        ax[1].set_title(dataset)

        views = ["XY", "XZ", "YZ"]
        for index, view in enumerate(views):
            ax[index].set_xticks([])
            ax[index].set_yticks([])
            ax[index].set_xlabel(view[0])
            ax[index].set_ylabel(view[1], rotation=0)

        for _, fiber in group.groupby("id"):
            ax[0].plot(fiber["xpos"], fiber["ypos"], marker="o", ms=1, lw=1)
            ax[1].plot(fiber["xpos"], fiber["zpos"], marker="o", ms=1, lw=1)
            ax[2].plot(fiber["ypos"], fiber["zpos"], marker="o", ms=1, lw=1)

        if save_location is not None:
            save_key = save_key_template % dataset
            save_figure(save_location, save_key, figure)
