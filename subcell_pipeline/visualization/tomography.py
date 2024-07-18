import os
from typing import Optional

import numpy as np
import pandas as pd
from io_collection.load.load_buffer import load_buffer
from io_collection.load.load_dataframe import load_dataframe
from io_collection.save.save_buffer import save_buffer
from simulariumio import CameraData, MetaData, TrajectoryConverter, UnitData

from subcell_pipeline.analysis.compression_metrics.compression_metric import (
    CompressionMetric,
)
from subcell_pipeline.visualization.fiber_points import (
    generate_trajectory_converter_for_fiber_points,
)
from subcell_pipeline.visualization.histogram_plots import make_empty_histogram_plots
from subcell_pipeline.visualization.spatial_annotator import SpatialAnnotator

TOMOGRAPHY_SAMPLE_COLUMNS: list[str] = ["xpos", "ypos", "zpos"]

TOMOGRAPHY_VIZ_SCALE: float = 100.0


def _add_tomography_plots(
    converter: TrajectoryConverter,
    metrics: list[CompressionMetric],
    fiber_points: list[np.ndarray],
) -> None:
    """Add plots to tomography data with calculated metrics."""

    histogram_plots = make_empty_histogram_plots(metrics)

    for metric, plot in histogram_plots.items():
        values = [
            metric.calculate_metric(polymer_trace=fiber[0, :, :])
            for fiber in fiber_points
        ]

        if metric == CompressionMetric.COMPRESSION_RATIO:
            plot.traces["actin"] = np.array(values) * 100
        else:
            plot.traces["actin"] = np.array(values)

        converter.add_plot(plot, "histogram")


def _get_tomography_spatial_center_and_size(
    tomo_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Get the center and size of the tomography dataset in 3D space."""

    all_mins = []
    all_maxs = []

    for column in TOMOGRAPHY_SAMPLE_COLUMNS:
        all_mins.append(tomo_df[column].min())
        all_maxs.append(tomo_df[column].max())

    mins = np.array(all_mins)
    maxs = np.array(all_maxs)

    return mins + 0.5 * (maxs - mins), maxs - mins


def visualize_tomography(
    bucket: str,
    name: str,
    temp_path: str,
    metrics: Optional[list[CompressionMetric]] = None,
) -> None:
    """
    Visualize segmented tomography data for actin fibers.

    Parameters
    ----------
    bucket
        Name of S3 bucket for input and output files.
    name
        Name of tomography dataset.
    temp_path
        Local path for saving visualization output files.
    metrics
        List of metrics to include in visualization plots.
    """

    tomo_key = f"{name}/{name}_coordinates_sampled.csv"
    tomo_df = load_dataframe(bucket, tomo_key)
    tomo_df = tomo_df.sort_values(by=["id", "monomer_ids"])
    tomo_df = tomo_df.reset_index(drop=True)

    time_units = UnitData("count")
    spatial_units = UnitData("um", 0.003)

    center, box_size = _get_tomography_spatial_center_and_size(tomo_df)

    all_fiber_points = []
    all_type_names = []

    for fiber_id, fiber_df in tomo_df.groupby("id"):
        fiber_index, dataset = fiber_id.split("_", 1)
        fiber_points = TOMOGRAPHY_VIZ_SCALE * (
            np.array([fiber_df[TOMOGRAPHY_SAMPLE_COLUMNS]]) - center
        )
        all_fiber_points.append(fiber_points)
        all_type_names.append(f"{dataset}#{fiber_index}")

    converter = generate_trajectory_converter_for_fiber_points(
        all_fiber_points,
        all_type_names,
        MetaData(
            box_size=TOMOGRAPHY_VIZ_SCALE * box_size,
            camera_defaults=CameraData(position=np.array([0.0, 0.0, 70.0])),
        ),
        {},
        time_units,
        spatial_units,
    )

    converter._data = SpatialAnnotator.add_sphere_agents(
        converter._data,
        fiber_points,
        type_name="point",
        radius=0.8,
        rainbow_colors=True,
    )

    if metrics:
        _add_tomography_plots(converter, metrics, all_fiber_points)

    # Save locally and copy to bucket.
    local_file_path = os.path.join(temp_path, name)
    converter.save(output_path=local_file_path)
    output_key = f"{name}/{name}.simularium"
    save_buffer(bucket, output_key, load_buffer(temp_path, f"{name}.simularium"))
