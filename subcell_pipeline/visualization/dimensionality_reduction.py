import os
from typing import Tuple

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


def rgb_to_hex_color(color):
    rgb = (int(255 * color[0]), int(255 * color[1]), int(255 * color[2]))
    return "#%02x%02x%02x" % rgb


def pca_fiber_points_over_time(
    samples: list[np.ndarray],
    pca: PCA,
    pc_ix: int,
    simulator_name: str = "Combined",
) -> Tuple[list[np.ndarray], list[str], dict[str, DisplayData]]:
    """
    Get fiber_points for samples of the PC distributions
    in order to visualize the samples over time
    """
    if simulator_name == "Combined":
        simulator_name = ""
    if simulator_name:
        simulator_name += "#"
    fiber_points = []
    display_data = {}
    for sample_ix in range(len(samples[0])):
        if pc_ix < 1:
            data = [samples[0][sample_ix], 0]
        else:
            data = [0, samples[1][sample_ix]]
        fiber_points.append(pca.inverse_transform(data).reshape(-1, 3))
    fiber_points = np.array(fiber_points)
    type_name = f"{simulator_name}PC{pc_ix + 1}"
    display_data[type_name] = DisplayData(
        name=type_name,
        display_type=DISPLAY_TYPE.FIBER,
        color="#eaeaea",
    )
    return [fiber_points], [type_name], display_data


def pca_fiber_points_one_timestep(
    samples: list[np.ndarray],
    pca: PCA,
    color_maps: list[Colormap],
    pc_ix: int,
    simulator_name: str = "Combined",
) -> Tuple[list[np.ndarray], list[str], dict[str, DisplayData]]:
    """
    Get fiber_points for samples of the PC distributions
    in order to visualize the samples together in one timestep.
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
                color=rgb_to_hex_color(color_map(abs(sample) / color_range)),
            )
    return fiber_points, type_names, display_data


def generate_simularium_and_save(
    name: str,
    fiber_points: list[np.ndarray],
    type_names: list[str],
    display_data: dict[str, DisplayData],
    distribution_over_time: bool,
    simulator_detail: bool,
    bucket: str,
    temp_path: str,
    pc: str,
) -> Tuple[list[np.ndarray], list[str], dict[str, DisplayData]]:
    """
    Generate a Simulariumio object for the fiber points and save it.
    """
    meta_data = MetaData(
        box_size=BOX_SIZE,
        camera_defaults=CameraData(
            position=np.array([-20.0, 350.0, 200.0]),
            look_at_position=np.array([50.0, 0.0, 0.0]),
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
        fiber_radius=8.0,
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
    range_pc1: list[float],
    range_pc2: list[float],
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
    range_pc1
        Min and max values of PC1 to visualize.
    range_pc2
        Min and max values of PC2 to visualize.
    separate_pcs
        True to Visualize PCs in separate files, False otherwise.
    sample_resolution
        Number of samples for each PC distribution.
    temp_path
        Local path for saving visualization output files.
    """
    pca_results = load_dataframe(bucket, pca_results_key)
    pca = load_pickle(bucket, pca_pickle_key)

    fiber_points = [[], []] if separate_pcs else []
    type_names = [[], []] if separate_pcs else []
    display_data = [{}, {}] if separate_pcs else {}
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
    dataset_name = os.path.splitext(pca_pickle_key)[0]
    pc_ixs = list(range(2))
    for simulator in pca_results_simulators:
        samples = [
            np.arange(
                range_pc1[0],
                range_pc1[1],
                (range_pc1[1] - range_pc1[0]) / float(sample_resolution),
            ),
            np.arange(
                range_pc2[0],
                range_pc2[1],
                (range_pc2[1] - range_pc2[0]) / float(sample_resolution),
            ),
        ]
        for pc_ix in pc_ixs:
            if distribution_over_time:
                _fiber_points, _type_names, _display_data = pca_fiber_points_over_time(
                    samples, pca, pc_ix, simulator
                )
            else:
                _fiber_points, _type_names, _display_data = (
                    pca_fiber_points_one_timestep(
                        samples, pca, color_maps, pc_ix, simulator
                    )
                )
            if separate_pcs:
                fiber_points[pc_ix] += _fiber_points
                type_names[pc_ix] += _type_names
                display_data[pc_ix] = {**display_data[pc_ix], **_display_data}
            else:
                fiber_points += _fiber_points
                type_names += _type_names
                display_data = {**display_data, **_display_data}
    if separate_pcs:
        for pc_ix in pc_ixs:
            generate_simularium_and_save(
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
        generate_simularium_and_save(
            dataset_name,
            fiber_points,
            type_names,
            display_data,
            distribution_over_time,
            simulator_detail,
            bucket,
            temp_path,
            "",
        )
