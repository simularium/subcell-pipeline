import os
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap
from sklearn.decomposition import PCA
from io_collection.load.load_buffer import load_buffer
from io_collection.load.load_dataframe import load_dataframe
from io_collection.load.load_pickle import load_pickle
from io_collection.save.save_buffer import save_buffer
from simulariumio import DISPLAY_TYPE, CameraData, DisplayData, MetaData, UnitData

from subcell_pipeline.visualization.fiber_points import (
    generate_trajectory_converter_for_fiber_points,
)

BOX_SIZE: np.ndarray = np.array(3 * [600.0])
"""Bounding box size for dimensionality reduction trajectory."""


def rgb_to_hex_color(color):
    rgb = (int(255 * color[0]), int(255 * color[1]), int(255 * color[2]))
    return "#%02x%02x%02x" % rgb


def pca_fiber_points_over_time(
    stdev_pc1: float, 
    stdev_pc2: float, 
    samples: np.ndarray, 
    pca: PCA,
    color_maps: list[Colormap],
    simulator_name: str = "Combined",
) -> Tuple[list[np.ndarray], list[str], dict[str, DisplayData]]:
    """
    Get fiber_points for samples of the PC distributions
    in order to visualize the samples over time
    """
    color_map = color_maps[simulator_name]
    if simulator_name == "Combined":
        simulator_name = ""
    if simulator_name:
        simulator_name += "#"
    fiber_points = []
    type_names = []
    display_data = {}
    for pc_ix in range(2):
        fiber_points.append([])
        for sample in samples:
            if pc_ix < 1:
                data = [sample * stdev_pc1, 0]
            else:
                data = [0, sample * stdev_pc2]
            fiber_points[pc_ix].append(pca.inverse_transform(data).reshape(-1, 3))
        fiber_points[pc_ix] = np.array(fiber_points[pc_ix])
        type_name = f"{simulator_name}PC{pc_ix + 1}"
        type_names.append(type_name)
        display_data[type_name] = DisplayData(
            name=type_name,
            display_type=DISPLAY_TYPE.FIBER,
            color=rgb_to_hex_color(color_map(0.8)),
        )
    return fiber_points, type_names, display_data


def pca_fiber_points_one_timestep(
    stdev_pc1: float, 
    stdev_pc2: float, 
    samples: np.ndarray, 
    pca: PCA,
    color_maps: list[Colormap],
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
    std_devs = np.max(samples)
    for sample in samples:
        data = [
            [sample * stdev_pc1, 0],
            [0, sample * stdev_pc2],
        ]
        for pc_ix in range(2):
            fiber_points.append(pca.inverse_transform(data[pc_ix]).reshape(1, -1, 3))
            type_name = f"{simulator_name}PC{pc_ix + 1}#{sample}"
            type_names.append(type_name)
            if type_name not in display_data:
                display_data[type_name] = DisplayData(
                    name=type_name,
                    display_type=DISPLAY_TYPE.FIBER,
                    color=rgb_to_hex_color(color_map(abs(sample) / std_devs)),
                )
    return fiber_points, type_names, display_data 


def visualize_dimensionality_reduction(
    bucket: str,
    pca_results_key: str,
    pca_pickle_key: str,
    distribution_over_time: bool,
    simulator_detail: bool,
    std_devs: float,
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
    std_devs
        Number of standard deviations to visualize.
    sample_resolution
        Number of samples for each PC distribution. Should be odd.
    temp_path
        Local path for saving visualization output files.
    """
    pca_results = load_dataframe(bucket, pca_results_key)
    pca = load_pickle(bucket, pca_pickle_key)
    
    fiber_points = []
    type_names = []
    display_data = {}
    pca_results_simulators = {
        "Combined" : pca_results,
    }
    if simulator_detail:
        pca_results_simulators["ReaDDy"] = pca_results.loc[pca_results["SIMULATOR"] == "READDY"]
        pca_results_simulators["Cytosim"] = pca_results.loc[pca_results["SIMULATOR"] == "CYTOSIM"]
    color_maps = {
        "Combined" : plt.colormaps.get_cmap("RdPu"),
        "ReaDDy" : plt.colormaps.get_cmap("YlOrRd"),
        "Cytosim" : plt.colormaps.get_cmap("GnBu"),
    }
    
    for simulator in pca_results_simulators:
        inc = 2 * std_devs / (sample_resolution - 1)
        samples = np.arange(-std_devs, std_devs + inc, inc)
        results = pca_results_simulators[simulator]
        stdev_pc1 = float(results["PCA1"].std(ddof=0))
        stdev_pc2 = float(results["PCA2"].std(ddof=0))
        if distribution_over_time:
            _fiber_points, _type_names, _display_data = pca_fiber_points_over_time(
                stdev_pc1, stdev_pc2, samples, pca, color_maps, simulator
            )
        else: 
            _fiber_points, _type_names, _display_data = pca_fiber_points_one_timestep(
                stdev_pc1, stdev_pc2, samples, pca, color_maps, simulator
            )
        fiber_points += _fiber_points
        type_names += _type_names
        display_data = {**display_data, **_display_data}
    
    meta_data = MetaData(
        box_size=BOX_SIZE,
        camera_defaults=CameraData(
            position=np.array([10.0, 0.0, 200.0]),
            look_at_position=np.array([10.0, 0.0, 0.0]),
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
    name = os.path.splitext(pca_pickle_key)[0]
    output_key = name
    output_key += "_time" if distribution_over_time else ""
    output_key += "_simulators" if simulator_detail else ""
    local_file_path = os.path.join(temp_path, output_key)
    converter.save(output_path=local_file_path)
    output_key = f"{output_key}.simularium"
    save_buffer(bucket, f"{name}/{output_key}", load_buffer(temp_path, output_key))
