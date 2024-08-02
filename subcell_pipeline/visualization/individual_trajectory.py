"""Visualization methods for individual simulators."""

from typing import Optional

import numpy as np
import pandas as pd
from io_collection.keys.check_key import check_key
from io_collection.load.load_buffer import load_buffer
from io_collection.load.load_text import load_text
from io_collection.save.save_buffer import save_buffer
from pint import UnitRegistry
from simulariumio import (
    DISPLAY_TYPE,
    CameraData,
    DisplayData,
    InputFileData,
    MetaData,
    TrajectoryConverter,
    UnitData,
)
from simulariumio.cytosim import CytosimConverter, CytosimData, CytosimObjectInfo
from simulariumio.filters import EveryNthTimestepFilter
from simulariumio.readdy import ReaddyConverter, ReaddyData

from subcell_pipeline.analysis.compression_metrics.compression_analysis import (
    get_compression_metric_data,
)
from subcell_pipeline.analysis.compression_metrics.compression_metric import (
    CompressionMetric,
)
from subcell_pipeline.analysis.dimensionality_reduction.fiber_data import align_fiber
from subcell_pipeline.simulation.cytosim.post_processing import CYTOSIM_SCALE_FACTOR
from subcell_pipeline.simulation.readdy.loader import ReaddyLoader
from subcell_pipeline.simulation.readdy.parser import BOX_SIZE as READDY_BOX_SIZE
from subcell_pipeline.simulation.readdy.parser import (
    READDY_TIMESTEP,
    download_readdy_hdf5,
)
from subcell_pipeline.simulation.readdy.post_processor import ReaddyPostProcessor
from subcell_pipeline.visualization.display_data import get_readdy_display_data
from subcell_pipeline.visualization.scatter_plots import make_empty_scatter_plots
from subcell_pipeline.visualization.spatial_annotator import SpatialAnnotator

READDY_SAVED_FRAMES: int = 1000

BOX_SIZE: np.ndarray = np.array(3 * [600.0])

UNIT_REGISTRY = UnitRegistry()


def _add_individual_plots(
    converter: TrajectoryConverter,
    metrics: list[CompressionMetric],
    metrics_data: pd.DataFrame,
    times: np.ndarray,
    time_units: UnitData,
) -> None:
    """Add plots to individual trajectory with calculated metrics."""
    scatter_plots = make_empty_scatter_plots(
        metrics, times=times, time_units=time_units
    )
    for metric, plot in scatter_plots.items():
        plot.ytraces["filament"] = np.array(metrics_data[metric.value])
        converter.add_plot(plot, "scatter")


def _add_readdy_spatial_annotations(
    converter: TrajectoryConverter,
    post_processor: ReaddyPostProcessor,
    n_monomer_points: int,
) -> None:
    """
    Add visualizations of edges, normals, and control points to the ReaDDy
    Simularium data.
    """
    fiber_chain_ids = post_processor.linear_fiber_chain_ids(polymer_number_range=5)
    axis_positions, _ = post_processor.linear_fiber_axis_positions(fiber_chain_ids)
    fiber_points = post_processor.linear_fiber_control_points(
        axis_positions=axis_positions,
        n_points=n_monomer_points,
    )
    converter._data.agent_data.positions, fiber_points = (
        post_processor.align_trajectory(fiber_points)
    )
    axis_positions, _ = post_processor.linear_fiber_axis_positions(fiber_chain_ids)
    edges = post_processor.edge_positions()

    # edges
    converter._data = SpatialAnnotator.add_fiber_agents(
        converter._data,
        fiber_points=edges,
        type_name="edge",
        fiber_width=0.5,
        color="#eaeaea",
    )

    # normals
    normals = post_processor.linear_fiber_normals(
        fiber_chain_ids=fiber_chain_ids,
        axis_positions=axis_positions,
        normal_length=10.0,
    )
    converter._data = SpatialAnnotator.add_fiber_agents(
        converter._data,
        fiber_points=normals,
        type_name="normal",
        fiber_width=0.5,
        color="#685bf3",
    )

    # control points
    sphere_positions = []
    for time_ix in range(len(fiber_points)):
        sphere_positions.append(fiber_points[time_ix][0])
    converter._data = SpatialAnnotator.add_sphere_agents(
        converter._data,
        sphere_positions,
        type_name="fiber point",
        radius=0.8,
        rainbow_colors=True,
    )


def _get_readdy_simularium_converter(
    path_to_readdy_h5: str,
    total_steps: int,
    n_timepoints: int,
) -> TrajectoryConverter:
    """
    Load from ReaDDy outputs and generate a TrajectoryConverter to visualize an
    actin trajectory in Simularium.
    """
    converter = ReaddyConverter(
        ReaddyData(
            timestep=1e-6 * (READDY_TIMESTEP * total_steps / READDY_SAVED_FRAMES),
            path_to_readdy_h5=path_to_readdy_h5,
            meta_data=MetaData(
                box_size=READDY_BOX_SIZE,
                camera_defaults=CameraData(
                    position=np.array([70.0, 70.0, 300.0]),
                    look_at_position=np.array([70.0, 70.0, 0.0]),
                    fov_degrees=60.0,
                ),
                scale_factor=1.0,
            ),
            display_data=get_readdy_display_data(),
            time_units=UnitData("ms"),
            spatial_units=UnitData("nm"),
        )
    )
    return _filter_time(converter, n_timepoints)


def visualize_individual_readdy_trajectory(
    bucket: str,
    series_name: str,
    series_key: str,
    rep_ix: int,
    n_timepoints: int,
    n_monomer_points: int,
    total_steps: int,
    temp_path: str,
    metrics: list[CompressionMetric],
    metrics_data: pd.DataFrame,
) -> None:
    """
    Save a Simularium file for a single ReaDDy trajectory with plots and spatial
    annotations.

    Parameters
    ----------
    bucket
        Name of S3 bucket for input and output files.
    series_name
        Name of simulation series.
    series_key
        Combination of series and condition names.
    rep_ix
        Replicate index.
    n_timepoints
        Number of equally spaced timepoints to visualize.
    n_monomer_points
        Number of equally spaced monomer points to visualize.
    total_steps
        Total number of steps for each simulation key.
    temp_path
        Local path for saving visualization output files.
    metrics
        List of metrics to include in visualization plots.
    metrics_data
        Calculated compression metrics data.
    """

    h5_file_path = download_readdy_hdf5(
        bucket, series_name, series_key, rep_ix, temp_path
    )

    assert isinstance(h5_file_path, str)

    converter = _get_readdy_simularium_converter(
        h5_file_path, total_steps, n_timepoints
    )

    if metrics:
        times = 2 * metrics_data["time"].values  # "time" seems to range (0, 0.5)
        times *= 1e-6 * (READDY_TIMESTEP * total_steps / n_timepoints)
        _add_individual_plots(
            converter, metrics, metrics_data, times, converter._data.time_units
        )

    assert isinstance(h5_file_path, str)

    rep_id = rep_ix + 1
    pickle_key = f"{series_name}/data/{series_key}_{rep_id:06d}.pkl"
    time_inc = total_steps // n_timepoints

    readdy_loader = ReaddyLoader(
        h5_file_path=h5_file_path,
        time_inc=time_inc,
        timestep=READDY_TIMESTEP,
        pickle_location=bucket,
        pickle_key=pickle_key,
    )

    post_processor = ReaddyPostProcessor(
        readdy_loader.trajectory(), box_size=READDY_BOX_SIZE
    )

    _add_readdy_spatial_annotations(converter, post_processor, n_monomer_points)

    # Save simularium file. Turn off validate IDs for performance.
    converter.save(output_path=h5_file_path, validate_ids=False)


def visualize_individual_readdy_trajectories(
    bucket: str,
    series_name: str,
    condition_keys: list[str],
    n_replicates: int,
    n_timepoints: int,
    n_monomer_points: int,
    total_steps: dict[str, int],
    temp_path: str,
    metrics: Optional[list[CompressionMetric]] = None,
    recalculate: bool = True,
) -> None:
    """
    Visualize individual ReaDDy simulations for select conditions and
    replicates.

    Parameters
    ----------
    bucket
        Name of S3 bucket for input and output files.
    series_name
        Name of simulation series.
    condition_keys
        List of condition keys.
    n_replicates
        Number of simulation replicates.
    n_timepoints
        Number of equally spaced timepoints to visualize.
    n_monomer_points
        Number of equally spaced monomer points to visualize.
    total_steps
        Total number of steps for each simulation key.
    temp_path
        Local path for saving visualization output files.
    metrics
        List of metrics to include in visualization plots.
    recalculate
        True to recalculate visualization files, False otherwise.
    """

    if metrics is not None:
        all_metrics_data = get_compression_metric_data(
            bucket,
            series_name,
            condition_keys,
            list(range(1, n_replicates + 1)),
            metrics,
            recalculate=False,
        )
    else:
        metrics = []
        all_metrics_data = pd.DataFrame(columns=["key", "seed"])

    for condition_key in condition_keys:
        series_key = f"{series_name}_{condition_key}" if condition_key else series_name

        for rep_ix in range(n_replicates):
            rep_id = rep_ix + 1
            output_key = f"{series_name}/viz/{series_key}_{rep_id:06d}.simularium"

            # Skip if output file already exists.
            if not recalculate and check_key(bucket, output_key):
                print(
                    f"Simularium file for [ { output_key } ] already exists. Skipping."
                )
                continue

            print(f"Visualizing data for [ {condition_key} ] replicate [ {rep_ix} ]")

            # Filter metrics data for specific conditon and replicate.
            if condition_key:
                metrics_data = all_metrics_data[
                    (all_metrics_data["key"] == condition_key)
                    & (all_metrics_data["seed"] == rep_id)
                ]
            else:
                metrics_data = all_metrics_data[(all_metrics_data["seed"] == rep_id)]

            visualize_individual_readdy_trajectory(
                bucket,
                series_name,
                series_key,
                rep_ix,
                n_timepoints,
                n_monomer_points,
                total_steps[condition_key],
                temp_path,
                metrics,
                metrics_data,
            )

            # Upload saved file to S3.
            temp_key = f"{series_key}_{rep_ix}.h5.simularium"
            save_buffer(bucket, output_key, load_buffer(temp_path, temp_key))


def _find_time_units(raw_time: float, units: str = "s") -> tuple[str, float]:
    """Get compact time units and a multiplier to put the times in those units."""
    time = UNIT_REGISTRY.Quantity(raw_time, units)
    time_compact = time.to_compact()
    return f"{time_compact.units:~}", time_compact.magnitude / raw_time


def _filter_time(
    converter: TrajectoryConverter, n_timepoints: int
) -> TrajectoryConverter:
    """Filter times using simulariumio time filter."""
    time_inc = int(converter._data.agent_data.times.shape[0] / n_timepoints)
    if time_inc < 2:
        return converter
    converter._data = converter.filter_data([EveryNthTimestepFilter(n=time_inc)])
    return converter


def _align_cytosim_fiber(converter: TrajectoryConverter) -> None:
    """
    Align the fiber subpoints so that the furthest point from the x-axis
    is aligned with the positive y-axis at the last time point.
    """
    fiber_points = converter._data.agent_data.subpoints[:, 0, :]
    n_timesteps = fiber_points.shape[0]
    n_points = int(fiber_points.shape[1] / 3)
    fiber_points = fiber_points.reshape((n_timesteps, n_points, 3))
    _, rotation = align_fiber(fiber_points[-1])
    for time_ix in range(n_timesteps):
        rotated = np.dot(fiber_points[time_ix][:, 1:], rotation)
        converter._data.agent_data.subpoints[time_ix, 0, :] = np.concatenate(
            (fiber_points[time_ix][:, 0:1], rotated), axis=1
        ).reshape(n_points * 3)


def _get_cytosim_simularium_converter(
    fiber_points_data: str,
    singles_data: str,
    n_timepoints: int,
) -> TrajectoryConverter:
    """
    Load from Cytosim outputs and generate a TrajectoryConverter to visualize an
    actin trajectory in Simularium.
    """
    singles_display_data = DisplayData(
        name="linker",
        radius=0.004,
        display_type=DISPLAY_TYPE.SPHERE,
        color="#eaeaea",
    )
    converter = CytosimConverter(
        CytosimData(
            meta_data=MetaData(
                box_size=BOX_SIZE,
                camera_defaults=CameraData(
                    position=np.array([70.0, 70.0, 300.0]),
                    look_at_position=np.array([70.0, 70.0, 0.0]),
                    fov_degrees=60.0,
                ),
                scale_factor=1,
            ),
            object_info={
                "fibers": CytosimObjectInfo(
                    cytosim_file=InputFileData(
                        file_contents=fiber_points_data,
                    ),
                    display_data={
                        1: DisplayData(
                            name="actin",
                            radius=0.002,
                            display_type=DISPLAY_TYPE.FIBER,
                            color="#1cbfaa",
                        )
                    },
                ),
                "singles": CytosimObjectInfo(
                    cytosim_file=InputFileData(
                        file_contents=singles_data,
                    ),
                    display_data={
                        1: singles_display_data,
                        2: singles_display_data,
                        3: singles_display_data,
                        4: singles_display_data,
                    },
                ),
            },
        )
    )
    _align_cytosim_fiber(converter)
    converter._data.agent_data.radii *= CYTOSIM_SCALE_FACTOR
    converter._data.agent_data.positions *= CYTOSIM_SCALE_FACTOR
    converter._data.agent_data.subpoints *= CYTOSIM_SCALE_FACTOR
    converter = _filter_time(converter, n_timepoints)
    time_units, time_multiplier = _find_time_units(converter._data.agent_data.times[-1])
    converter._data.agent_data.times *= time_multiplier
    converter._data.time_units = UnitData(time_units)
    return converter


def visualize_individual_cytosim_trajectory(
    bucket: str,
    series_name: str,
    series_key: str,
    index: int,
    n_timepoints: int,
    temp_path: str,
    metrics: list[CompressionMetric],
    metrics_data: pd.DataFrame,
) -> None:
    """
    Save a Simularium file for a single Cytosim trajectory with plots and
    spatial annotations.

    Parameters
    ----------
    bucket
        Name of S3 bucket for input and output files.
    series_name
        Name of simulation series.
    series_key
        Combination of series and condition names.
    index
        Simulation replicate index.
    n_timepoints
        Number of equally spaced timepoints to visualize.
    temp_path
        Local path for saving visualization output files.
    metrics
        List of metrics to include in visualization plots.
    metrics_data
        Calculated compression metrics data.
    """

    output_key_template = f"{series_name}/outputs/{series_key}_{index}/%s"
    fiber_points_data = load_text(bucket, output_key_template % "fiber_points.txt")
    singles_data = load_text(bucket, output_key_template % "singles.txt")

    converter = _get_cytosim_simularium_converter(
        fiber_points_data, singles_data, n_timepoints
    )

    if metrics:
        times = 1e3 * metrics_data["time"].values  # s --> ms
        _add_individual_plots(
            converter, metrics, metrics_data, times, converter._data.time_units
        )

    # Save simularium file. Turn off validate IDs for performance.
    local_file_path = f"{temp_path}/{series_key}_{index}"
    converter.save(output_path=local_file_path, validate_ids=False)


def visualize_individual_cytosim_trajectories(
    bucket: str,
    series_name: str,
    condition_keys: list[str],
    random_seeds: list[int],
    n_timepoints: int,
    temp_path: str,
    metrics: Optional[list[CompressionMetric]] = None,
    recalculate: bool = True,
) -> None:
    """
    Visualize individual Cytosim simulations for select conditions and
    replicates.

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
        Number of equally spaced timepoints to visualize.
    temp_path
        Local path for saving visualization output files.
    metrics
        List of metrics to include in visualization plots.
    recalculate
        True to recalculate visualization files, False otherwise.
    """

    if metrics is not None:
        all_metrics_data = get_compression_metric_data(
            bucket,
            series_name,
            condition_keys,
            random_seeds,
            metrics,
            recalculate=False,
        )
    else:
        metrics = []
        all_metrics_data = pd.DataFrame(columns=["key", "seed"])

    for condition_key in condition_keys:
        series_key = f"{series_name}_{condition_key}" if condition_key else series_name

        for index, seed in enumerate(random_seeds):
            output_key = f"{series_name}/viz/{series_key}_{seed:06d}.simularium"

            # Skip if output file already exists.
            if not recalculate and check_key(bucket, output_key):
                print(
                    f"Simularium file for [ { output_key } ] already exists. Skipping."
                )
                continue

            print(f"Visualizing data for [ {condition_key} ] seed [ {seed} ]")

            # Filter metrics data for specific conditon and replicate.
            if condition_key:
                metrics_data = all_metrics_data[
                    (all_metrics_data["key"] == condition_key)
                    & (all_metrics_data["seed"] == seed)
                ]
            else:
                metrics_data = all_metrics_data[(all_metrics_data["seed"] == seed)]

            visualize_individual_cytosim_trajectory(
                bucket,
                series_name,
                series_key,
                index,
                n_timepoints,
                temp_path,
                metrics,
                metrics_data,
            )

            temp_key = f"{series_key}_{index}.simularium"
            save_buffer(bucket, output_key, load_buffer(temp_path, temp_key))
