#!/usr/bin/env python

import os
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
from pint import UnitRegistry
from io_collection.keys.check_key import check_key
from io_collection.load.load_text import load_text
from io_collection.load.load_dataframe import load_dataframe
from simulariumio import (
    TrajectoryConverter,
    MetaData,
    InputFileData,
    DisplayData,
    DISPLAY_TYPE,
    UnitData,
    ScatterPlotData,
    HistogramPlotData,
    CameraData,
    TrajectoryData,
    AgentData,
    DimensionData,
)
from simulariumio.filters import EveryNthTimestepFilter
from simulariumio.cytosim import CytosimConverter, CytosimData, CytosimObjectInfo
from simulariumio.readdy import ReaddyConverter, ReaddyData
from ..constants import (
    BOX_SIZE, 
    WORKING_DIR_PATH, 
    READDY_TIMESTEP, 
    READDY_TOTAL_STEPS, 
    READDY_SAVED_FRAMES,
    READDY_DISPLAY_DATA,
    SIMULATOR_COLORS,
    TOMOGRAPHY_SAMPLE_COLUMNS,
    TOMOGRAPHY_VIZ_SCALE,
    TOMOGRAPHY_MIN_COMPRESSION,
    TOMOGRAPHY_SCALE_FACTOR,
)

from ..temporary_file_io import (
    download_readdy_hdf5, 
    upload_file_to_s3,
    make_working_directory,
)
from ..constants import (
    BOX_SIZE, 
    READDY_TOTAL_STEPS, 
    CYTOSIM_SCALE_FACTOR,
)
from subcell_pipeline.analysis.compression_metrics.compression_metric import (
    CompressionMetric,
)
from ..simulation.readdy import ReaddyPostProcessor, load_readdy_fiber_points
from .spatial_annotator import SpatialAnnotator

def _empty_scatter_plots(
    total_steps: int = -1,
    times: np.ndarray = None,
    time_units: str = None,
) -> Dict[CompressionMetric, ScatterPlotData]:
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
    return {
        CompressionMetric.AVERAGE_PERP_DISTANCE: ScatterPlotData(
            title="Average Perpendicular Distance",
            xaxis_title=xlabel,
            yaxis_title="distance (nm)",
            xtrace=xtrace,
            ytraces={
                "<<<": np.zeros(total_steps),
                ">>>": 85.0 * np.ones(total_steps),
            },
            render_mode="lines",
        ),
        CompressionMetric.CALC_BENDING_ENERGY: ScatterPlotData(
            title="Bending Energy",
            xaxis_title=xlabel,
            yaxis_title="energy",
            xtrace=xtrace,
            ytraces={
                "<<<": np.zeros(total_steps),
                ">>>": 10.0 * np.ones(total_steps),
            },
            render_mode="lines",
        ),
        CompressionMetric.NON_COPLANARITY: ScatterPlotData(
            title="Non-coplanarity",
            xaxis_title=xlabel,
            yaxis_title="3rd component variance from PCA",
            xtrace=xtrace,
            ytraces={
                "<<<": np.zeros(total_steps),
                ">>>": 0.03 * np.ones(total_steps),
            },
            render_mode="lines",
        ),
        CompressionMetric.PEAK_ASYMMETRY: ScatterPlotData(
            title="Peak Asymmetry",
            xaxis_title=xlabel,
            yaxis_title="normalized peak distance",
            xtrace=xtrace,
            ytraces={
                "<<<": np.zeros(total_steps),
                ">>>": 0.5 * np.ones(total_steps),
            },
            render_mode="lines",
        ),
        CompressionMetric.CONTOUR_LENGTH: ScatterPlotData(
            title="Contour Length",
            xaxis_title=xlabel,
            yaxis_title="filament contour length (nm)",
            xtrace=xtrace,
            ytraces={
                "<<<": 480 * np.ones(total_steps),
                ">>>": 505 * np.ones(total_steps),
            },
            render_mode="lines",
        ),
    }


def _generate_plot_data(fiber_points: np.ndarray) -> Dict[CompressionMetric, list[float]]:
    """
    Calculate plot traces from fiber_points.
    """
    n_points = int(fiber_points.shape[2] / 3.0)
    result = {
        CompressionMetric.AVERAGE_PERP_DISTANCE: [],
        CompressionMetric.CALC_BENDING_ENERGY: [],
        CompressionMetric.NON_COPLANARITY: [],
        CompressionMetric.PEAK_ASYMMETRY: [],
        CompressionMetric.CONTOUR_LENGTH: [],
    }
    total_steps = fiber_points.shape[0]
    for time_ix in range(total_steps):
        points = fiber_points[time_ix][0].reshape((n_points, 3))
        for metric in result.keys():
            result[metric].append(
                metric.calculate_metric(
                    polymer_trace=points
                )
            )
    return result


def _add_individual_plots(
    converter: TrajectoryConverter, 
    fiber_points: np.ndarray,
    times: np.ndarray,
) -> None:
    """
    Add plots to an individual trajectory 
    using fiber_points to calculate metrics.
    """
    scatter_plots = _empty_scatter_plots(times)
    plot_data = _generate_plot_data(fiber_points)
    for metric, plot in scatter_plots.items():
        plot.ytraces["filament"] = np.array(plot_data[metric])
        converter.add_plot(plot, "scatter")


def _add_readdy_spatial_annotations(
    converter: TrajectoryConverter,
    post_processor: ReaddyPostProcessor,
    fiber_chain_ids: List[List[List[int]]],
    axis_positions: List[List[np.ndarray]],
    fiber_points: np.ndarray,
) -> None:
    """
    Add visualizations of edges, normals, and control points
    to the ReaDDy Simularium data.
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


def _load_readdy_simularium(path_to_readdy_h5: str, series_key: str) -> TrajectoryConverter:
    """
    Load from ReaDDy outputs and generate a TrajectoryConverter 
    to visualize an actin trajectory in Simularium.
    """
    total_steps = READDY_TOTAL_STEPS[series_key]
    return ReaddyConverter(ReaddyData(
        timestep=1e-6 * (READDY_TIMESTEP * total_steps / READDY_SAVED_FRAMES),
        path_to_readdy_h5=path_to_readdy_h5,
        meta_data=MetaData(
            box_size=BOX_SIZE,
            camera_defaults=CameraData(
                position=np.array([0.0, 0.0, 300.0]),
                look_at_position=np.zeros(3),
                up_vector=np.array([0.0, 1.0, 0.0]),
                fov_degrees=120.0,
            ),
            scale_factor=1.0,
        ),
        display_data=READDY_DISPLAY_DATA(),
        time_units=UnitData("ms"),
        spatial_units=UnitData("nm"),
    ))

def _visualize_readdy_trajectory(
    bucket: str,
    series_name: str,
    series_key: str,
    rep_ix: int,
    n_timepoints: int,
    n_monomer_points: int,
) -> None:
    """
    Save a Simularium file for a single ReaDDy trajectory with plots and spatial annotations.
    """
    path_to_readdy_h5 = os.path.join(WORKING_DIR_PATH, f"{series_key}_{rep_ix}.h5")
    converter = _load_readdy_simularium(path_to_readdy_h5, series_key)
    
    # load data shaped for analysis from a pickle if it exists, otherwise save one
    post_processor, fiber_chain_ids, axis_positions, fiber_points, times = load_readdy_fiber_points(
        bucket, series_name, series_key, rep_ix, n_timepoints, n_monomer_points
    )  
    _add_individual_plots(converter, fiber_points, times)
    _add_readdy_spatial_annotations(
        converter, post_processor, fiber_chain_ids, axis_positions, fiber_points
    )

    # save simularium file
    converter.save(
        output_path=path_to_readdy_h5,
        validate_ids=False,  # for performance
    )


def visualize_individual_readdy_trajectories(
    bucket: str,
    series_name: str,
    condition_keys: list[str],
    n_replicates: int,
    n_timepoints: int,
    n_monomer_points: int,
    recalculate: bool = True,
) -> None:
    """
    Visualize individual ReaDDy simulations for select conditions and replicates.

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
        Number of timepoints to visualize.
    n_monomer_points
        Number of control points for each polymer trace.
    recalculate
        Overwrite any outputs that already exist?
    """
    for condition_key in condition_keys:
        series_key = f"{series_name}_{condition_key}" if condition_key else series_name

        for rep_ix in range(n_replicates):
            local_h5_path = os.path.join(WORKING_DIR_PATH, f"{series_key}_{rep_ix}.h5")
            rep_id = rep_ix + 1
            output_key = f"{series_name}/viz/{series_key}_{rep_id:06d}.simularium"

            # Skip if output file already exists.
            if not recalculate and check_key(bucket, output_key):
                print(f"Simularium visualization [ { output_key } ] already exists. Skipping.")
                continue

            print(f"Visualizing data for [ {condition_key} ] replicate [ {rep_ix} ]")
            
            download_readdy_hdf5(bucket, series_name, series_key, rep_ix)
    
            _visualize_readdy_trajectory(
                bucket,
                series_name,
                series_key,
                rep_ix,
                n_timepoints,
                n_monomer_points,
            )
            
            upload_file_to_s3(bucket, f"{local_h5_path}.simularium", output_key)


ureg = UnitRegistry()

def _find_time_units(raw_time: float, units: str = "s") -> Tuple[str, float]:
    """
    Get the compact time units and a multiplier to put the times in those units
    """
    time = ureg.Quantity(raw_time, units)
    time = time.to_compact()
    return "{:~}".format(time.units), time.magnitude / raw_time


def _filter_time(converter: TrajectoryConverter, n_timepoints: int) -> TrajectoryConverter:
    """
    Use Simulariumio time filter
    """
    time_inc = int(converter._data.agent_data.times.shape[0] / n_timepoints)
    if time_inc < 2:
        return converter
    converter._data = converter.filter_data(
        [
            EveryNthTimestepFilter(
                n=time_inc,
            ),
        ]
    )
    return converter
            
            
def _load_cytosim_simularium(
    fiber_points_data: str,
    singles_data: str,
    n_timepoints: int,
) -> TrajectoryConverter:
    """
    Load from Cytosim outputs and generate a TrajectoryConverter 
    to visualize an actin trajectory in Simularium.
    """
    singles_display_data = DisplayData(
        name="linker",
        radius=0.01,
        display_type=DISPLAY_TYPE.SPHERE,
        color="#fff",
    )
    converter = CytosimConverter(CytosimData(
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
                        name=f"actin",
                        radius=0.02,
                        display_type=DISPLAY_TYPE.FIBER,
                    )
                },
            ),
            "singles" : CytosimObjectInfo(
                cytosim_file=InputFileData(
                    file_contents=singles_data,
                ),
                display_data={
                    1 : singles_display_data,
                    2 : singles_display_data,
                    3 : singles_display_data,
                    4 : singles_display_data,
                }
            ),
        },
    ))
    converter = _filter_time(converter, n_timepoints)
    time_units, time_multiplier = _find_time_units(converter._data.agent_data.times[-1])
    converter._data.agent_data.times *= time_multiplier
    converter._data.time_units = UnitData(time_units)
    return converter
            
            
def _visualize_cytosim_trajectory(
    fiber_points_data: str,
    singles_data: str,
    local_output_path: str,
    n_timepoints: int,
) -> None:
    """
    Save a Simularium file for a single Cytosim trajectory with plots.
    """
    converter = _load_cytosim_simularium(fiber_points_data, singles_data, n_timepoints)
    _add_individual_plots(
        converter, 
        converter._data.agent_data.subpoints, 
        converter._data.agent_data.times
    )
    converter.save(local_output_path)


def visualize_individual_cytosim_trajectories(
    bucket: str,
    series_name: str,
    condition_keys: list[str],
    random_seeds: list[int],
    n_timepoints: int,
    recalculate: bool = True,
) -> None:
    """
    Visualize individual Cytosim simulations for select conditions and replicates.

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
        Number of timepoints to visualize.
    recalculate
        Overwrite any outputs that already exist?
    """
    for condition_key in condition_keys:
        series_key = f"{series_name}_{condition_key}" if condition_key else series_name

        for index, seed in enumerate(random_seeds):
            output_key = f"{series_name}/viz/{series_key}_{seed:06d}.simularium"
            
            # Skip if output file already exists.
            if not recalculate and check_key(bucket, output_key):
                print(f"Simularium visualization [ { output_key } ] already exists. Skipping.")
                continue
            
            output_key_template = f"{series_name}/outputs/{series_key}_{index}/%s"
            fiber_points_data = load_text(
                bucket, output_key_template % "fiber_points.txt"
            )
            singles_data = load_text(
                bucket, output_key_template % "singles.txt"
            )
            local_output_path = os.path.join(WORKING_DIR_PATH, output_key)
            _visualize_cytosim_trajectory(
                fiber_points_data, singles_data, local_output_path, n_timepoints
            )
            
            upload_file_to_s3(bucket, local_output_path, output_key)


def _load_fiber_points_from_dataframe(
    simulator: str,
    dataframe: pd.DataFrame, 
    n_timepoints: int
) -> np.ndarray:
    """
    Load fiber points from pre-calculated dataframes
    and generate a TrajectoryConverter to visualize 
    all actin trajectories together in Simularium.
    """
    dataframe.sort_values(by=["time", "fiber_point"])
    total_steps = dataframe.time.unique().shape[0]
    n_points = dataframe.fiber_point.unique().shape[0]
    if total_steps != n_timepoints:
        raise Exception(
            f"Requested number of timesteps [ {n_timepoints} ] does not match "
            f"number of timesteps in dataset [ {total_steps} ]."
        )
    result = []
    for time_ix in range(total_steps):
        result.append([])
        result[time_ix].append(
            (CYTOSIM_SCALE_FACTOR if simulator == "cytosim" else 1) * np.array(
                dataframe[time_ix * n_points : (time_ix + 1) * n_points][["xpos", "ypos", "zpos"]]
            )
        )
    return np.array(result)
     
            
def _load_all_together_simularium(
    fiber_points: list[np.ndarray],
    type_names: list[str],
    display_data: Dict[str, DisplayData],
) -> TrajectoryConverter:
    """
    Generate a TrajectoryConverter with all simulations from ReaDDy and Cytosim together.
    """
    total_conditions = len(fiber_points)
    total_steps = fiber_points[0].shape[0]
    n_monomer_points = fiber_points[0].shape[1]
    subpoints = []
    traj_data = TrajectoryData(
        meta_data=MetaData(
            box_size=np.array([BOX_SIZE, BOX_SIZE, BOX_SIZE]),
            camera_defaults=CameraData(
                position=np.array([10.0, 0.0, 200.0]),
                look_at_position=np.array([10.0, 0.0, 0.0]),
                fov_degrees=60.0,
            ),
            trajectory_title="Actin compression in Cytosim and Readdy",
        ),
        agent_data=AgentData(
            times=np.arange(total_steps),
            n_agents=total_conditions * np.ones((total_steps)),
            viz_types=1001
            * np.ones((total_steps, total_conditions)),  # fiber viz type = 1001
            unique_ids=np.array(total_steps * [list(range(total_conditions))]),
            types=total_steps * [type_names],
            positions=np.zeros((total_steps, total_conditions, 3)),
            radii=np.ones((total_steps, total_conditions)),
            n_subpoints=3 * n_monomer_points * np.ones((total_steps, total_conditions)),
            subpoints=align(subpoints),
            display_data=display_data,
        ),
        time_units=UnitData("count"),  # frames
        spatial_units=UnitData("nm"),  # nanometer
    )
    return TrajectoryConverter(traj_data)


def _add_combined_plots(
    converter: TrajectoryConverter, 
    fiber_points: np.ndarray,
    type_names: list[str],
    n_timepoints: int,
) -> None:
    """
    Add plots for all trajectories together  
    using fiber_points to calculate metrics.
    """
    scatter_plots = _empty_scatter_plots(total_steps=n_timepoints)
    for traj_ix in range(len(fiber_points)):
        plot_data = _generate_plot_data(fiber_points[traj_ix])
        for metric, plot in scatter_plots.items():
            plot.ytraces[type_names[traj_ix]] = np.array(plot_data[metric])
    for metric, plot in scatter_plots.items():
        converter.add_plot(plot, "scatter")


def visualize_all_compressed_trajectories_together(
    subcell_bucket: str,
    readdy_bucket: str,
    readdy_series_name: str,
    cytosim_bucket: str,
    cytosim_series_name: str,
    condition_keys: list[str],
    n_replicates: int,
    n_timepoints: int,
) -> None:
    """
    Visualize simulations from ReaDDy and Cytosim together
    for select conditions and number of replicates.

    Parameters
    ----------
    subcell_bucket
        Name of S3 bucket for combined input and output files.
    readdy_bucket
        Name of S3 bucket for ReaDDy input and output files.
    readdy_series_name
        Name of ReaDDy simulation series.
    cytosim_bucket
        Name of S3 bucket for Cytosim input and output files.
    cytosim_series_name
        Name of Cytosim simulation series.
    condition_keys
        List of condition keys.
    n_replicates
        How many replicates to visualize.
    n_timepoints
        Number of timepoints to visualize.
    """
    fiber_points = []
    type_names = []
    display_data = {}
    for condition_key in condition_keys:
        for index in range(n_replicates):
            for simulator in SIMULATOR_COLORS:
                
                # get path of dataframe from simulation post-processing to use as input
                rep_id = index + 1
                if simulator == "readdy":
                    bucket = readdy_bucket
                    df_key = f"{readdy_series_name}/data/{readdy_series_name}_{condition_key}_{rep_id:06d}.csv"
                else:
                    bucket = cytosim_bucket
                    df_key = f"{cytosim_series_name}/samples/{cytosim_series_name}_{condition_key}_{rep_id:06d}.csv"
                
                # Skip if input dataframe does not exist.
                if not check_key(bucket, df_key):
                    print(f"Dataframe not available for {simulator} [ { df_key } ]. Skipping.")
                    continue
                
                dataframe = load_dataframe(bucket, df_key)
                fiber_points.append(_load_fiber_points_from_dataframe(simulator, dataframe, n_timepoints))
                condition = float(condition_key[:3] + "." + condition_key[-1])
                condition = round(condition) if condition_key[-1] == "0" else condition
                type_names.append(f"{simulator}#{condition} um/s {index}")
                display_data[type_names[-1]] = DisplayData(
                    name=type_names[-1],
                    display_type=DISPLAY_TYPE.FIBER,
                    color=SIMULATOR_COLORS[simulator],
                )
    
    converter = _load_all_together_simularium(fiber_points, type_names, display_data)
    _add_combined_plots(converter, fiber_points, type_names, n_timepoints)
    output_key = "actin_compression_cytosim_readdy.simularium"
    local_output_path = os.path.join(WORKING_DIR_PATH, output_key)
    converter.save(local_output_path)
            
    upload_file_to_s3(subcell_bucket, local_output_path, output_key)


def _empty_tomography_plots() -> Dict[CompressionMetric, HistogramPlotData]:
    return {
        CompressionMetric.CONTOUR_LENGTH : HistogramPlotData(
            title="Contour Length",
            xaxis_title="filament contour length (nm)",
            traces={},
        ),
        CompressionMetric.COMPRESSION_RATIO : HistogramPlotData(
            title="Compression Percentage",
            xaxis_title="percent (%)",
            traces={},
        ),
        CompressionMetric.AVERAGE_PERP_DISTANCE : HistogramPlotData(
            title="Average Perpendicular Distance",
            xaxis_title="distance (nm)",
            traces={},
        ),
        CompressionMetric.CALC_BENDING_ENERGY : HistogramPlotData(
            title="Bending Energy",
            xaxis_title="energy",
            traces={},
        ),
        CompressionMetric.NON_COPLANARITY : HistogramPlotData(
            title="Non-coplanarity",
            xaxis_title="3rd component variance from PCA",
            traces={},
        ),
        CompressionMetric.PEAK_ASYMMETRY : HistogramPlotData(
            title="Peak Asymmetry",
            xaxis_title="normalized peak distance",
            traces={},
        ),
    }


def _add_tomography_plots(tomo_df: pd.DataFrame, converter: TrajectoryConverter) -> None:
    """
    Add plots to tomography data using pre-calculated metrics.
    """
    plots = _empty_tomography_plots()
    for metric in plots:
        values = []
        for _, fiber in tomo_df.groupby("id"):
            polymer_trace = fiber[["xpos", "ypos", "zpos"]].values
            values.append(metric.calculate_metric(
                polymer_trace=polymer_trace
            ))
        plots[metric].traces["actin"] = np.array(values)
        if metric == CompressionMetric.COMPRESSION_RATIO:
            plots[metric].traces["actin"] *= 100.
        converter.add_plot(plots[metric], "histogram")


def _get_tomography_spatial_center_and_size(tomo_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
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
    
    
def _load_tomography_simularium(bucket: str, name: str) -> TrajectoryConverter:
    """
    Load sampled tomography data and generate a TrajectoryConverter 
    to visualize it in Simularium.
    """
    tomo_key = f"{name}/{name}_coordinates_sampled.csv"
    tomo_df = load_dataframe(bucket, tomo_key)
    tomo_df = tomo_df.sort_values(by=["id", "monomer_ids"])
    tomo_df = tomo_df.reset_index(drop=True)
    names, ids = np.unique(np.array(list(tomo_df["id"])), return_index=True)
    traj_ids = names[np.argsort(ids)]
    center, box_size = _get_tomography_spatial_center_and_size(tomo_df)
    max_points = 0
    subpoints = []
    compression_ratios = []
    test_points = []
    for traj_id in traj_ids:
        fiber_df = tomo_df.loc[tomo_df["id"] == traj_id]
        points = TOMOGRAPHY_VIZ_SCALE * (np.array(fiber_df[["xpos", "ypos", "zpos"]]) - center)
        subpoints.append(points.flatten())
        compression_ratio = CompressionMetric.COMPRESSION_RATIO.calculate_metric(points)
        compression_ratios.append(compression_ratio)
        if compression_ratio > 0.75:
            test_points.append(points)
        if len(fiber_df) > max_points:
            max_points = len(fiber_df)
    n_agents = len(subpoints)
    compression_percents = 100. * np.array(compression_ratios)
    min_compression_ratio = np.amin(compression_percents)
    max_compression_ratio = np.amax(compression_percents)
    bins = np.linspace(min_compression_ratio, max_compression_ratio, 20)
    digitized = np.digitize(compression_percents, bins)
    type_names = []
    display_data = {}
    type_name_min = f"actin less than {TOMOGRAPHY_MIN_COMPRESSION} percent compressed"
    for agent_ix in range(n_agents):
        bin_percent = int(10 * bins[digitized[agent_ix] - 1]) / 10.
        if bin_percent < TOMOGRAPHY_MIN_COMPRESSION:
            type_name = type_name_min
        else:
            bin_percent_name = str(round(bin_percent))
            type_name = f"actin {bin_percent_name} percent compressed"
        type_names.append(type_name)
        if type_name not in display_data:
            display_data[type_name] = DisplayData(
                name=type_name,
                display_type=DISPLAY_TYPE.FIBER,
            )
    display_data[type_name_min] = DisplayData(
        name=type_name_min,
        display_type=DISPLAY_TYPE.FIBER,
        color="#222222",
    )
    agent_data = AgentData.from_dimensions(DimensionData(
        total_steps=1,
        max_agents=n_agents,
        max_subpoints=3 * max_points,
    ))
    agent_data.n_agents[0] = n_agents
    agent_data.viz_types[0] = 1001.0 * np.ones(n_agents)
    agent_data.unique_ids[0] = np.arange(n_agents)
    agent_data.types[0] = type_names
    agent_data.radii *= 0.5
    for agent_ix in range(n_agents):
        n_subpoints = subpoints[agent_ix].shape[0]
        agent_data.n_subpoints[0][agent_ix] = n_subpoints
        agent_data.subpoints[0][agent_ix][:n_subpoints] = subpoints[agent_ix]
    agent_data.display_data = display_data
    UNIT_SCALE_FACTOR = 1 / 20.  # TODO not sure where this factor came from or if it is still needed
    traj_data = TrajectoryData(
        meta_data=MetaData(
            box_size=TOMOGRAPHY_VIZ_SCALE * box_size,
            camera_defaults=CameraData(position=np.array([0.0, 0.0, 70.0]))
        ),
        agent_data=agent_data,
        spatial_units=UnitData("um", UNIT_SCALE_FACTOR * TOMOGRAPHY_SCALE_FACTOR / TOMOGRAPHY_VIZ_SCALE),  # 0.003?
    )
    converter = TrajectoryConverter(traj_data)
    _add_tomography_plots(tomo_df, converter)
    
    # TODO remove after debugging fiber point sampling order
    for ix, points in enumerate(test_points):
        converter._data = SpatialAnnotator.add_sphere_agents(
            converter._data,
            [points],
            type_name=f"fiber point",
            radius=0.8,
        )
    
    return converter


def visualize_tomography(bucket: str, name: str) -> None:
    """
    Visualize segmented tomography data for actin fibers.

    Parameters
    ----------
    data
        Tomography data.
    """
    output_key = f"{name}/{name}"
    local_output_path = os.path.join(WORKING_DIR_PATH, name)
    make_working_directory()
    
    converter = _load_tomography_simularium(bucket, name)
    converter.save(local_output_path)
            
    upload_file_to_s3(bucket, f"{local_output_path}.simularium", f"{output_key}.simularium")
                