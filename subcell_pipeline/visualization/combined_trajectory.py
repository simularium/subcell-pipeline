import os
from typing import Optional

import numpy as np
import pandas as pd
from io_collection.keys.check_key import check_key
from io_collection.load.load_buffer import load_buffer
from io_collection.load.load_dataframe import load_dataframe
from io_collection.save.save_buffer import save_buffer
from simulariumio import (
    DISPLAY_TYPE,
    AgentData,
    CameraData,
    DisplayData,
    MetaData,
    TrajectoryConverter,
    TrajectoryData,
    UnitData,
)

from subcell_pipeline.analysis.compression_metrics.compression_analysis import (
    get_compression_metric_data,
)
from subcell_pipeline.analysis.compression_metrics.compression_metric import (
    CompressionMetric,
)
from subcell_pipeline.analysis.dimensionality_reduction.fiber_data import align_fibers
from subcell_pipeline.visualization.scatter_plots import make_empty_scatter_plots

BOX_SIZE: np.ndarray = np.array(3 * [600.0])


def _load_fiber_points_from_dataframe(
    dataframe: pd.DataFrame, n_timepoints: int
) -> np.ndarray:
    """
    Load and reshape fiber points from sampled dataframe.

    Sampled dataframe is in the shape (n_timepoints x n_fiber_points, 3); method
    returns the dataframe reshaped to (n_timepoints, n_fiber_points x 3). If the
    sampled dataframe does not have the expected number of timepoints, method
    will raise an exception.
    """

    dataframe.sort_values(by=["time", "fiber_point"])
    total_steps = dataframe.time.unique().shape[0]

    if total_steps != n_timepoints:
        raise Exception(
            f"Requested number of timesteps [ {n_timepoints} ] does not match "
            f"number of timesteps in dataset [ {total_steps} ]."
        )

    align_fibers(dataframe)

    fiber_points = []
    for _, group in dataframe.groupby("time"):
        fiber_points.append(group[["xpos", "ypos", "zpos"]].values.flatten())

    return np.array(fiber_points)


def get_combined_trajectory_converter(
    fiber_points: list[np.ndarray],
    type_names: list[str],
    display_data: dict[str, DisplayData],
) -> TrajectoryConverter:
    """
    Generate a TrajectoryConverter to visualize simulations from ReaDDy and
    Cytosim together.
    """

    total_conditions = len(fiber_points)
    total_steps = fiber_points[0].shape[0]
    total_subpoints = fiber_points[0].shape[1]

    traj_data = TrajectoryData(
        meta_data=MetaData(
            box_size=BOX_SIZE,
            camera_defaults=CameraData(
                position=np.array([10.0, 0.0, 200.0]),
                look_at_position=np.array([10.0, 0.0, 0.0]),
                fov_degrees=60.0,
            ),
            trajectory_title="Actin compression in Cytosim and Readdy",
        ),
        agent_data=AgentData(
            times=np.arange(total_steps),
            n_agents=total_conditions * np.ones(total_steps),
            viz_types=1001
            * np.ones((total_steps, total_conditions)),  # fiber viz type = 1001
            unique_ids=np.array(total_steps * [list(range(total_conditions))]),
            types=total_steps * [type_names],
            positions=np.zeros((total_steps, total_conditions, 3)),
            radii=np.ones((total_steps, total_conditions)),
            n_subpoints=total_subpoints * np.ones((total_steps, total_conditions)),
            subpoints=np.moveaxis(np.array(fiber_points), [0, 1], [1, 0]),
            display_data=display_data,
        ),
        time_units=UnitData("count"),  # frames
        spatial_units=UnitData("nm"),  # nanometer
    )
    return TrajectoryConverter(traj_data)


def _add_combined_plots(
    converter: TrajectoryConverter,
    metrics: list[CompressionMetric],
    metrics_data: pd.DataFrame,
    n_timepoints: int,
    plot_names: list[tuple[str, str, int]],
    type_names: list[str],
) -> None:
    """Add plots for combined trajectories with calculated metrics."""
    scatter_plots = make_empty_scatter_plots(metrics, total_steps=n_timepoints)

    for metric, plot in scatter_plots.items():
        for plot_name, type_name in zip(plot_names, type_names):
            simulator, key, seed = plot_name
            simulator_data = metrics_data[simulator]
            data = simulator_data[
                (simulator_data["key"] == key) & (simulator_data["seed"] == seed)
            ]
            plot.ytraces[type_name] = np.array(data[metric.value])
        converter.add_plot(plot, "scatter")


def visualize_combined_trajectories(
    buckets: dict[str, str],
    series_names: dict[str, str],
    condition_keys: list[str],
    replicates: list[int],
    n_timepoints: int,
    simulator_colors: dict[str, str],
    temp_path: str,
    metrics: Optional[list[CompressionMetric]] = None,
) -> None:
    """
    Visualize combined simulations from ReaDDy and Cytosim for select conditions
    and number of replicates.

    Parameters
    ----------
    buckets
        Names of S3 buckets for input and output files for each simulator and
        visualization.
    series_names
        Names of simulation series for each simulator.
    condition_keys
        List of condition keys.
    replicates
        Simulation replicates ids.
    n_timepoints
        Number of equally spaced timepoints to visualize.
    simulator_colors
        Map of simulator name to color.
    temp_path
        Local path for saving visualization output files.
    metrics
        List of metrics to include in visualization plots.
    recalculate
        True to recalculate visualization files, False otherwise.
    """

    fiber_points = []
    type_names = []
    plot_names = []
    display_data = {}
    all_metrics_data = {}

    for simulator, color in simulator_colors.items():
        bucket = buckets[simulator]
        series_name = series_names[simulator]

        # Load calculated compression metric data.
        if metrics is not None:
            all_metrics_data[simulator] = get_compression_metric_data(
                bucket,
                series_name,
                condition_keys,
                replicates,
                metrics,
                recalculate=False,
            )
        else:
            metrics = []
            all_metrics_data[simulator] = pd.DataFrame(columns=["key", "seed"])

        for condition_key in condition_keys:
            series_key = (
                f"{series_name}_{condition_key}" if condition_key else series_name
            )

            for replicate in replicates:
                dataframe_key = (
                    f"{series_name}/samples/{series_key}_{replicate:06d}.csv"
                )

                # Skip if input dataframe does not exist.
                if not check_key(bucket, dataframe_key):
                    print(
                        f"Dataframe not available for {simulator} "
                        f"[ { dataframe_key } ]. Skipping."
                    )
                    continue

                print(
                    f"Loading data for [ {simulator} ] "
                    f"condition [ { dataframe_key } ] "
                    f"replicate [ {replicate} ]"
                )

                dataframe = load_dataframe(bucket, dataframe_key)
                fiber_points.append(
                    _load_fiber_points_from_dataframe(dataframe, n_timepoints)
                )

                condition = int(condition_key) / 10
                condition = round(condition) if condition_key[-1] == "0" else condition

                type_names.append(f"{simulator}#{condition} um/s {replicate}")
                plot_names.append((simulator, condition_key, replicate))
                display_data[type_names[-1]] = DisplayData(
                    name=type_names[-1],
                    display_type=DISPLAY_TYPE.FIBER,
                    color=color,
                )

    converter = get_combined_trajectory_converter(
        fiber_points, type_names, display_data
    )

    if metrics:
        _add_combined_plots(
            converter, metrics, all_metrics_data, n_timepoints, plot_names, type_names
        )

    output_key = "actin_compression_cytosim_readdy.simularium"
    local_file_path = os.path.join(temp_path, output_key)
    converter.save(output_path=local_file_path.replace(".simularium", ""))
    output_bucket = buckets["combined"]
    save_buffer(output_bucket, output_key, load_buffer(temp_path, output_key))
