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
    ScatterPlotData,
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
from subcell_pipeline.simulation.cytosim.post_processing import CYTOSIM_SCALE_FACTOR
from subcell_pipeline.simulation.readdy.loader import ReaddyLoader
from subcell_pipeline.simulation.readdy.parser import BOX_SIZE as READDY_BOX_SIZE
from subcell_pipeline.simulation.readdy.parser import (
    READDY_TIMESTEP,
    download_readdy_hdf5,
)
from subcell_pipeline.simulation.readdy.post_processor import ReaddyPostProcessor
from subcell_pipeline.visualization.display_data import get_readdy_display_data
from subcell_pipeline.visualization.spatial_annotator import SpatialAnnotator

READDY_SAVED_FRAMES: int = 1000

BOX_SIZE: np.ndarray = np.array(3 * [600.0])

UNIT_REGISTRY = UnitRegistry()


def _empty_scatter_plots(
    metrics: list[CompressionMetric],
    total_steps: int = -1,
    times: Optional[np.ndarray] = None,
    time_units: Optional[str] = None,
) -> dict[CompressionMetric, ScatterPlotData]:
    """Create empty scatter plot placeholders for list of metrics."""

    if total_steps < 0 and times is None:
        raise Exception("Either total_steps or times array is required for plots")
    elif times is None:
        # use normalized time
        xlabel = "T (normalized)"
        xtrace = (1 / float(total_steps)) * np.arange(total_steps)
    else:
        # use actual time
        xlabel = f"T ({time_units})"
        xtrace = times
        total_steps = times.shape[0]

    plots = {}

    for metric in metrics:
        lower_bound, upper_bound = metric.bounds()
        plots[metric] = ScatterPlotData(
            title=metric.label(),
            xaxis_title=xlabel,
            yaxis_title=metric.description(),
            xtrace=xtrace,
            ytraces={
                "<<<": lower_bound * np.ones(total_steps),
                ">>>": upper_bound * np.ones(total_steps),
            },
            render_mode="lines",
        )

    return plots


def _add_individual_plots(
    converter: TrajectoryConverter,
    metrics: list[CompressionMetric],
    metrics_data: pd.DataFrame,
) -> None:
    """Add plots to individual trajectory with calculated metrics."""
    times = metrics_data["time"].values
    scatter_plots = _empty_scatter_plots(metrics, times=times)
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
    # edges
    edges = post_processor.edge_positions()
    converter._data = SpatialAnnotator.add_fiber_agents(
        converter._data,
        fiber_points=edges,
        type_name="edge",
        fiber_width=0.5,
        color="#eaeaea",
    )

    fiber_chain_ids = post_processor.linear_fiber_chain_ids(polymer_number_range=5)
    axis_positions, _ = post_processor.linear_fiber_axis_positions(fiber_chain_ids)
    fiber_points = post_processor.linear_fiber_control_points(
        axis_positions=axis_positions,
        n_points=n_monomer_points,
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
        color="#eaeaea",
    )


def get_readdy_simularium_converter(
    path_to_readdy_h5: str, total_steps: int
) -> TrajectoryConverter:
    """
    Load from ReaDDy outputs and generate a TrajectoryConverter to visualize an
    actin trajectory in Simularium.
    """
    return ReaddyConverter(
        ReaddyData(
            timestep=1e-6 * (READDY_TIMESTEP * total_steps / READDY_SAVED_FRAMES),
            path_to_readdy_h5=path_to_readdy_h5,
            meta_data=MetaData(
                box_size=READDY_BOX_SIZE,
                camera_defaults=CameraData(
                    position=np.array([0.0, 0.0, 300.0]),
                    look_at_position=np.zeros(3),
                    up_vector=np.array([0.0, 1.0, 0.0]),
                    fov_degrees=120.0,
                ),
                scale_factor=1.0,
            ),
            display_data=get_readdy_display_data(),
            time_units=UnitData("ms"),
            spatial_units=UnitData("nm"),
        )
    )


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
    """

    h5_file_path = download_readdy_hdf5(
        bucket, series_name, series_key, rep_ix, temp_path
    )

    assert isinstance(h5_file_path, str)

    converter = get_readdy_simularium_converter(h5_file_path, total_steps)

    if metrics:
        _add_individual_plots(converter, metrics, metrics_data)

    assert isinstance(h5_file_path, str)

    # TODO: fix temporal scaling? it looks like the actual data, metrics, and
    # the annotations are drawing at different time scales

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
        Path for saving temporary h5 files.
    metrics
        List of metrics to include in visualization plots.
    recalculate
        True to recalculate visualization files, False otherwise.
    """

    if metrics is not None:
        print(bucket, series_name, condition_keys)
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


def get_cytosim_simularium_converter(
    fiber_points_data: str,
    singles_data: str,
    n_timepoints: int,
) -> TrajectoryConverter:
    """
    Load from Cytosim outputs and generate a TrajectoryConverter to visualize an
    actin trajectory in Simularium.
    """

    # TODO: fix converter not showing fiber, possible scaling issue

    singles_display_data = DisplayData(
        name="linker",
        radius=0.01,
        display_type=DISPLAY_TYPE.SPHERE,
        color="#fff",
    )

    converter = CytosimConverter(
        CytosimData(
            meta_data=MetaData(
                box_size=BOX_SIZE,
                scale_factor=CYTOSIM_SCALE_FACTOR,
            ),
            object_info={
                "fibers": CytosimObjectInfo(
                    cytosim_file=InputFileData(
                        file_contents=fiber_points_data,
                    ),
                    display_data={
                        1: DisplayData(
                            name="actin",
                            radius=0.02,
                            display_type=DISPLAY_TYPE.FIBER,
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
    """Save a Simularium file for a single Cytosim trajectory with plots."""

    output_key_template = f"{series_name}/outputs/{series_key}_{index}/%s"
    fiber_points_data = load_text(bucket, output_key_template % "fiber_points.txt")
    singles_data = load_text(bucket, output_key_template % "singles.txt")

    converter = get_cytosim_simularium_converter(
        fiber_points_data, singles_data, n_timepoints
    )

    if metrics:
        _add_individual_plots(converter, metrics, metrics_data)

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
        print(bucket, series_name, condition_keys)
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
            break
        break
