import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from io_collection.load.load_dataframe import load_dataframe
from simulariumio import (
    DISPLAY_TYPE,
    AgentData,
    CameraData,
    DisplayData,
    HistogramPlotData,
    MetaData,
    TrajectoryConverter,
    TrajectoryData,
    UnitData,
)

from subcell_pipeline.analysis.compression_metrics.compression_metric import (
    CompressionMetric,
)

from ..constants import (
    TOMOGRAPHY_SAMPLE_COLUMNS,
    TOMOGRAPHY_VIZ_SCALE,
    WORKING_DIR_PATH,
)
from ..temporary_file_io import make_working_directory
from .spatial_annotator import SpatialAnnotator


def _save_and_upload_simularium_file(
    converter: TrajectoryConverter, bucket: str, output_key: str
) -> None:
    """
    Save a local simularium file and upload it to s3.
    """
    local_key = os.path.splitext(os.path.basename(output_key))[0]
    local_output_path = os.path.join(WORKING_DIR_PATH, local_key)
    make_working_directory()

    converter.save(local_output_path)

    # upload_file_to_s3(bucket, f"{local_output_path}.simularium", output_key) TODO


def _generate_simularium_for_fiber_points(
    fiber_points: list[np.ndarray],
    type_names: list[str],
    meta_data: MetaData,
    display_data: Dict[str, DisplayData],
    time_units: UnitData,
    spatial_units: UnitData,
) -> TrajectoryConverter:
    """
    Generate a TrajectoryConverter for the fiber_points
    (list of fibers, each = timesteps X points X 3)
    """
    # build subpoints array with correct dimensions
    n_fibers = len(fiber_points)
    total_steps = fiber_points[0].shape[0]
    n_points = fiber_points[0].shape[1]
    subpoints = np.zeros((total_steps, n_fibers, n_points, 3))
    for time_ix in range(total_steps):
        for fiber_ix in range(n_fibers):
            subpoints[time_ix][fiber_ix] = fiber_points[fiber_ix][time_ix]
    subpoints = subpoints.reshape((total_steps, n_fibers, 3 * n_points))
    # convert to simularium
    traj_data = TrajectoryData(
        meta_data=meta_data,
        agent_data=AgentData(
            times=np.arange(total_steps),
            n_agents=n_fibers * np.ones((total_steps)),
            viz_types=1001 * np.ones((total_steps, n_fibers)),  # fiber viz type = 1001
            unique_ids=np.array(total_steps * [list(range(n_fibers))]),
            types=total_steps * [type_names],
            positions=np.zeros((total_steps, n_fibers, 3)),
            radii=0.5 * np.ones((total_steps, n_fibers)),
            n_subpoints=3 * n_points * np.ones((total_steps, n_fibers)),
            subpoints=subpoints,
            display_data=display_data,
        ),
        time_units=time_units,
        spatial_units=spatial_units,
    )
    return TrajectoryConverter(traj_data)


def _empty_tomography_plots() -> Dict[CompressionMetric, HistogramPlotData]:
    return {
        CompressionMetric.CONTOUR_LENGTH: HistogramPlotData(
            title="Contour Length",
            xaxis_title="filament contour length (nm)",
            traces={},
        ),
        CompressionMetric.COMPRESSION_RATIO: HistogramPlotData(
            title="Compression Percentage",
            xaxis_title="percent (%)",
            traces={},
        ),
        CompressionMetric.AVERAGE_PERP_DISTANCE: HistogramPlotData(
            title="Average Perpendicular Distance",
            xaxis_title="distance (nm)",
            traces={},
        ),
        CompressionMetric.CALC_BENDING_ENERGY: HistogramPlotData(
            title="Bending Energy",
            xaxis_title="energy",
            traces={},
        ),
        CompressionMetric.NON_COPLANARITY: HistogramPlotData(
            title="Non-coplanarity",
            xaxis_title="3rd component variance from PCA",
            traces={},
        ),
        CompressionMetric.PEAK_ASYMMETRY: HistogramPlotData(
            title="Peak Asymmetry",
            xaxis_title="normalized peak distance",
            traces={},
        ),
    }


def _add_tomography_plots(
    fiber_points: list[np.ndarray], converter: TrajectoryConverter
) -> None:
    """
    Add plots to tomography data using pre-calculated metrics.
    """
    plots = _empty_tomography_plots()
    for metric in plots:
        values = []
        for fiber in fiber_points:
            values.append(metric.calculate_metric(polymer_trace=fiber))
        plots[metric].traces["actin"] = np.array(values)
        if metric == CompressionMetric.COMPRESSION_RATIO:
            plots[metric].traces["actin"] *= 100.0
        converter.add_plot(plots[metric], "histogram")


def _get_tomography_spatial_center_and_size(
    tomo_df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the center and size of the tomography dataset in 3D space.
    """
    ixs = [
        list(tomo_df.columns).index(TOMOGRAPHY_SAMPLE_COLUMNS[0]),
        list(tomo_df.columns).index(TOMOGRAPHY_SAMPLE_COLUMNS[1]),
        list(tomo_df.columns).index(TOMOGRAPHY_SAMPLE_COLUMNS[2]),
    ]
    unique_values = list(map(set, tomo_df.values.T))
    mins = []
    maxs = []
    for dim_ix in range(3):
        d_values = np.array(list(unique_values[ixs[dim_ix]]))
        mins.append(np.amin(d_values))
        maxs.append(np.amax(d_values))
    mins = np.array(mins)
    maxs = np.array(maxs)
    return mins + 0.5 * (maxs - mins), maxs - mins


def visualize_tomography(bucket: str, name: str) -> None:
    """
    Visualize segmented tomography data for actin fibers.

    Parameters
    ----------
    bucket
        Name of S3 bucket for input and output files.
    name
        Name of tomography dataset.
    """
    tomo_key = f"{name}/{name}_coordinates_sampled.csv"
    tomo_df = load_dataframe(bucket, tomo_key)
    tomo_df = tomo_df.sort_values(by=["id", "monomer_ids"])
    tomo_df = tomo_df.reset_index(drop=True)
    time_units = UnitData("count")
    spatial_units = UnitData("um", 0.003)
    names, ids = np.unique(np.array(list(tomo_df["id"])), return_index=True)
    traj_ids = names[np.argsort(ids)]
    for traj_id in traj_ids:
        fiber_df = tomo_df.loc[tomo_df["id"] == traj_id]
        center, box_size = _get_tomography_spatial_center_and_size(fiber_df)
        fiber_points = TOMOGRAPHY_VIZ_SCALE * (
            np.array([fiber_df[["xpos", "ypos", "zpos"]]]) - center
        )
        type_names = ["Raw data"]
        display_data = {
            "Raw data": DisplayData(
                name="Raw data",
                display_type=DISPLAY_TYPE.FIBER,
                color="#888888",
            )
        }
        converter = _generate_simularium_for_fiber_points(
            [fiber_points],
            type_names,
            MetaData(
                box_size=TOMOGRAPHY_VIZ_SCALE * box_size,
                camera_defaults=CameraData(position=np.array([0.0, 0.0, 70.0])),
            ),
            display_data,
            time_units,
            spatial_units,
        )

        # TODO remove after debugging fiber point order
        converter._data = SpatialAnnotator.add_sphere_agents(
            converter._data,
            [fiber_points[0]],
            type_name="point",
            radius=0.8,
        )

        _add_tomography_plots([fiber_points[0]], converter)
        _save_and_upload_simularium_file(
            converter, bucket, f"{name}/{name}_{traj_id}.simularium"
        )
