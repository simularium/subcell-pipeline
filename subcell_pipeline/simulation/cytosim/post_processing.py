"""Methods for processing Cytosim simulations."""

from typing import Union

import numpy as np
import pandas as pd
from io_collection.keys.check_key import check_key
from io_collection.load.load_text import load_text
from io_collection.save.save_dataframe import save_dataframe

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

CYTOSIM_SCALE_FACTOR: int = 1000
"""Default Cytosim position scaling factor."""

CYTOSIM_RIGIDITY: float = 0.041
"""Default Cytosim rigidity."""


def parse_cytosim_simulation_data(
    bucket: str,
    series_name: str,
    condition_keys: list[str],
    random_seeds: list[int],
) -> None:
    """
    Parse Cytosim simulation data for select conditions and seeds.

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
    """

    for condition_key in condition_keys:
        series_key = f"{series_name}_{condition_key}" if condition_key else series_name

        for index, seed in enumerate(random_seeds):
            dataframe_key = f"{series_name}/data/{series_key}_{seed:06d}.csv"

            # Skip if dataframe file already exists.
            if check_key(bucket, dataframe_key):
                print(f"Dataframe [ { dataframe_key } ] already exists. Skipping.")
                continue

            print(f"Parsing data for [ {condition_key} ] seed [ {seed} ]")

            output_key_template = f"{series_name}/outputs/{series_key}_{index}/%s"
            segment_curvature = load_text(
                bucket, output_key_template % "fiber_segment_curvature.txt"
            )
            data = parse_cytosim_simulation_curvature_data(segment_curvature)

            save_dataframe(bucket, dataframe_key, data, index=False)


def parse_cytosim_simulation_curvature_data(
    data: str,
    rigidity: float = CYTOSIM_RIGIDITY,
    scale_factor: int = CYTOSIM_SCALE_FACTOR,
) -> pd.DataFrame:
    """
    Parse Cytosim fiber segment curvature data into tidy data format.

    Parameters
    ----------
    data
        Output data from Cytosim report fiber:segment_energy.
    rigidity
        Fiber rigidity used to calculate segment energy.
    scale_factor
        Scaling factor for fiber points.

    Returns
    -------
    :
        Data for individual fiber points.
    """

    point_data: list[list[Union[str, int, float]]] = []

    fiber_index = 0
    point_index = 0

    # Iterate through each row of the data and extract data for each point.
    for line in data.splitlines():
        line_split = line.strip().split()

        if line.startswith("%"):
            if line.startswith("% time"):
                time = float(line.split(" ")[-1])
                print(f"Parsing timepoint [ {time} ]")
            elif line.startswith("% end"):
                fiber_index = 0
                point_index = 0
        elif len(line_split) > 0:
            if int(fiber_index) == int(line_split[0]):
                point_index += 1
            else:
                point_index = 0
                fiber_index += 1

            point_data.append([*line_split, time, point_index])

    # Combine all data into dataframe and update data types.
    dataframe = pd.DataFrame(point_data, columns=COLUMN_NAMES)
    dataframe = dataframe.astype(dtype=COLUMN_DTYPES)

    dataframe["xpos"] = dataframe["xpos"] * scale_factor
    dataframe["ypos"] = dataframe["ypos"] * scale_factor
    dataframe["zpos"] = dataframe["zpos"] * scale_factor

    # Calculate force magnitude
    dataframe["force_magnitude"] = np.sqrt(
        np.square(dataframe["xforce"])
        + np.square(dataframe["yforce"])
        + np.square(dataframe["zforce"])
    )

    # Calculate segment bending energy in pN nm
    dataframe["segment_energy"] = dataframe["segment_curvature"] * rigidity * 1000

    return dataframe
