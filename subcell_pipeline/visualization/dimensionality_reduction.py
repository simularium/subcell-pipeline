"""Visualization methods for dimensionality reduction analysis."""

import os

import matplotlib.pyplot as plt
import numpy as np
from io_collection.load.load_buffer import load_buffer
from io_collection.load.load_dataframe import load_dataframe
from io_collection.load.load_pickle import load_pickle
from io_collection.save.save_buffer import save_buffer
from matplotlib.colors import Colormap
from simulariumio import DISPLAY_TYPE, CameraData, DisplayData, MetaData, UnitData
from sklearn.decomposition import PCA

from subcell_pipeline.visualization.fiber_points import (
    generate_trajectory_converter_for_fiber_points,
)

BOX_SIZE: np.ndarray = np.array(3 * [600.0])
"""Bounding box size for dimensionality reduction trajectory."""


def _rgb_to_hex_color(color: tuple[float, float, float]) -> str:
    """
    Convert RGB color to hexadecimal format.

    Parameters
    ----------
    color
        Red, green, and blue colors (between 0.0 and 1.0).

    Returns
    -------
    :
        Color in hexadecimal format.
    """
    rgb = (int(255 * color[0]), int(255 * color[1]), int(255 * color[2]))
    return f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"


def _pca_fiber_points_over_time(
    samples: list[np.ndarray],
    pca: PCA,
    pc_ix: int,
    simulator_name: str = "Combined",
    color: str = "#eaeaea",
) -> tuple[list[np.ndarray], list[str], dict[str, DisplayData]]:
    """
    Get fiber_points for samples of the PC distributions in order to visualize
    the samples over time.
    """
    if simulator_name == "Combined":
        simulator_name = ""
    if simulator_name:
        simulator_name += "#"
    fiber_points: list[np.ndarray] = []
    display_data: dict[str, DisplayData] = {}
    for sample_ix in range(len(samples[0])):
        if pc_ix < 1:
            data = [samples[0][sample_ix], 0]
        else:
            data = [0, samples[1][sample_ix]]
        fiber_points.append(pca.inverse_transform(data).reshape(-1, 3))
    fiber_points_arr: np.ndarray = np.array(fiber_points)
    type_name: str = f"{simulator_name}PC{pc_ix + 1}"
    display_data[type_name] = DisplayData(
        name=type_name,
        display_type=DISPLAY_TYPE.FIBER,
        color=color,
    )
    return [fiber_points_arr], [type_name], display_data


def _pca_fiber_points_one_timestep(
    samples: list[np.ndarray],
    pca: PCA,
    color_maps: dict[str, Colormap],
    pc_ix: int,
    simulator_name: str = "Combined",
) -> tuple[list[np.ndarray], list[str], dict[str, DisplayData]]:
    """
    Get fiber_points for samples of the PC distributions in order to visualize
    the samples together in one timestep.
    """
    color_map = color_maps[simulator_name]
    if simulator_name == "Combined":
        simulator_name = ""
    if simulator_name:
        simulator_name += "_"

    fiber_points = []
    type_names = []
    display_data = {}
    for sample_ix in range(len(samples[0])):
        data = [
            [samples[0][sample_ix], 0],
            [0, samples[1][sample_ix]],
        ]
        fiber_points.append(pca.inverse_transform(data[pc_ix]).reshape(1, -1, 3))
        sample = samples[pc_ix][sample_ix]
        sample_name = str(round(sample))
        type_name = f"{simulator_name}PC{pc_ix + 1}#{sample_name}"
        type_names.append(type_name)
        if type_name not in display_data:
            color_range = -samples[pc_ix][0]
            display_data[type_name] = DisplayData(
                name=type_name,
                display_type=DISPLAY_TYPE.FIBER,
                color=_rgb_to_hex_color(color_map(abs(sample) / color_range)),
            )
    return fiber_points, type_names, display_data


def _generate_simularium_and_save(
    name: str,
    fiber_points: list[np.ndarray],
    type_names: list[str],
    display_data: dict[str, DisplayData],
    distribution_over_time: bool,
    simulator_detail: bool,
    bucket: str,
    temp_path: str,
    pc: str,
) -> None:
    """Generate a simulariumio object for the fiber points and save it."""
    meta_data = MetaData(
        box_size=BOX_SIZE,
        camera_defaults=CameraData(
            position=np.array([70.0, 70.0, 300.0]),
            look_at_position=np.array([70.0, 70.0, 0.0]),
            fov_degrees=60.0,
        ),
        trajectory_title="Actin Compression Dimensionality Reduction",
    )
    time_units = UnitData("count")  # frames
    spatial_units = UnitData("nm")  # nanometers
    converter = generate_trajectory_converter_for_fiber_points(
        fiber_points,
        type_names,
        meta_data,
        display_data,
        time_units,
        spatial_units,
        fiber_radius=1.0,
    )

    # Save locally and copy to bucket.
    output_key = name
    output_key += "_time" if distribution_over_time else ""
    output_key += "_simulators" if simulator_detail else ""
    output_key += f"_pc{pc}" if pc else ""
    local_file_path = os.path.join(temp_path, output_key)
    converter.save(output_path=local_file_path)
    output_key = f"{output_key}.simularium"
    save_buffer(bucket, f"{name}/{output_key}", load_buffer(temp_path, output_key))


def visualize_dimensionality_reduction(
    bucket: str,
    pca_results_key: str,
    pca_pickle_key: str,
    distribution_over_time: bool,
    simulator_detail: bool,
    sample_ranges: dict[str, list[list[float]]],
    separate_pcs: bool,
    sample_resolution: int,
    temp_path: str,
) -> None:
    """
    Visualize PCA space for actin fibers.

    Parameters
    ----------
    bucket
        Name of S3 bucket for input and output files.
    pca_results_key
        File key for PCA results dataframe.
    pca_pickle_key
        File key for PCA object pickle.
    distribution_over_time
        True to scroll through the PC distributions over time, False otherwise.
    simulator_detail
        True to show individual simulator ranges, False otherwise.
    sample_ranges
        Min and max values to visualize for each PC (and each simulator if
        simulator_detail).
    separate_pcs
        True to Visualize PCs in separate files, False otherwise.
    sample_resolution
        Number of samples for each PC distribution.
    temp_path
        Local path for saving visualization output files.
    """
    pca_results = load_dataframe(bucket, pca_results_key)
    pca = load_pickle(bucket, pca_pickle_key)

    fiber_points: list[list[np.ndarray]] = [[], []]
    type_names: list[list[str]] = [[], []]
    display_data: list[dict[str, DisplayData]] = [{}, {}]
    pca_results_simulators = {
        "Combined": pca_results,
    }
    if simulator_detail:
        pca_results_simulators["ReaDDy"] = pca_results.loc[
            pca_results["SIMULATOR"] == "READDY"
        ]
        pca_results_simulators["Cytosim"] = pca_results.loc[
            pca_results["SIMULATOR"] == "CYTOSIM"
        ]
    color_maps = {
        "Combined": plt.colormaps.get_cmap("RdPu"),
        "ReaDDy": plt.colormaps.get_cmap("YlOrRd"),
        "Cytosim": plt.colormaps.get_cmap("GnBu"),
    }
    over_time_colors = {
        "Combined": "#ffffff",
        "ReaDDy": "#ff8f52",
        "Cytosim": "#1cbfaa",
    }
    dataset_name = os.path.splitext(pca_pickle_key)[0]
    pc_ixs = list(range(2))
    for simulator in pca_results_simulators:
        samples = [
            np.arange(
                sample_ranges[simulator][0][0],
                sample_ranges[simulator][0][1],
                (sample_ranges[simulator][0][1] - sample_ranges[simulator][0][0])
                / float(sample_resolution),
            ),
            np.arange(
                sample_ranges[simulator][1][0],
                sample_ranges[simulator][1][1],
                (sample_ranges[simulator][1][1] - sample_ranges[simulator][1][0])
                / float(sample_resolution),
            ),
        ]
        for pc_ix in pc_ixs:
            if distribution_over_time:
                _fiber_points, _type_names, _display_data = _pca_fiber_points_over_time(
                    samples, pca, pc_ix, simulator, over_time_colors[simulator]
                )
            else:
                _fiber_points, _type_names, _display_data = (
                    _pca_fiber_points_one_timestep(
                        samples, pca, color_maps, pc_ix, simulator
                    )
                )
            if separate_pcs:
                fiber_points[pc_ix] += _fiber_points
                type_names[pc_ix] += _type_names
                display_data[pc_ix] = {**display_data[pc_ix], **_display_data}
            else:
                fiber_points[0] += _fiber_points
                type_names[0] += _type_names
                display_data[0] = {**display_data[0], **_display_data}
    if separate_pcs:
        for pc_ix in pc_ixs:
            _generate_simularium_and_save(
                dataset_name,
                fiber_points[pc_ix],
                type_names[pc_ix],
                display_data[pc_ix],
                distribution_over_time,
                simulator_detail,
                bucket,
                temp_path,
                str(pc_ix + 1),
            )
    else:
        _generate_simularium_and_save(
            dataset_name,
            fiber_points[0],
            type_names[0],
            display_data[0],
            distribution_over_time,
            simulator_detail,
            bucket,
            temp_path,
            "",
        )
