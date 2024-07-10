import os

import numpy as np
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

    if sample_resolution % 2 == 0:
        sample_resolution += 1

    pca_results = load_dataframe(bucket, pca_results_key)
    pca = load_pickle(bucket, pca_pickle_key)

    fiber_points = []
    type_names = []
    display_data = {}

    inc = 2 * std_devs / (sample_resolution - 1)
    samples = np.arange(-std_devs, std_devs + inc, inc)
    stdev_pc1 = pca_results["PCA1"].std(ddof=0)
    stdev_pc2 = pca_results["PCA2"].std(ddof=0)
    data = {
        "PC1": [samples * stdev_pc1, 0],
        "PC2": [0, samples * stdev_pc2],
    }

    if distribution_over_time:
        for pc_ix, pc in enumerate(data):
            fiber_points.append([])
            pca.inverse_transform(data[pc]).reshape(-1, 3)
            for _ in samples:
                fiber_points[pc_ix].append()
            fiber_points[pc_ix] = np.array(fiber_points[pc_ix])
    else:
        for sample in samples:
            for pc in data:

                import ipdb

                ipdb.set_trace()

                fiber_points.append(pca.inverse_transform(data[pc]).reshape(1, -1, 3))
                type_name = f"{pc}#{sample}"
                type_names.append(type_name)
                if type_name not in display_data:
                    display_data[type_name] = DisplayData(
                        name=type_name,
                        display_type=DISPLAY_TYPE.FIBER,
                    )

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
    )

    # Save locally and copy to bucket.
    name = os.path.splitext(pca_pickle_key)[0]
    local_file_path = os.path.join(temp_path, name)
    converter.save(output_path=local_file_path)
    output_key = f"{name}/{name}.simularium"
    save_buffer(bucket, output_key, load_buffer(temp_path, f"{name}.simularium"))
