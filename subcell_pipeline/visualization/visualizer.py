#!/usr/bin/env python

import os
from typing import Tuple, Dict, List

import numpy as np
from pint import UnitRegistry
from io_collection.keys.check_key import check_key
from io_collection.load.load_text import load_text
from simulariumio import (
    TrajectoryConverter,
    MetaData,
    InputFileData,
    DisplayData,
    DISPLAY_TYPE,
    UnitData,
    EveryNthTimestepFilter,
    ScatterPlotData,
    CameraData,
)
from simulariumio.cytosim import CytosimConverter, CytosimData, CytosimObjectInfo
from simulariumio.readdy import ReaddyConverter, ReaddyData
from ..constants import (
    BOX_SIZE, 
    LOCAL_DOWNLOADS_PATH, 
    READDY_TIMESTEP, 
    READDY_TOTAL_STEPS, 
    READDY_SAVED_FRAMES,
    READDY_DISPLAY_DATA,
)

from ..temporary_file_io import (
    download_readdy_hdf5, 
    upload_file_to_s3
)
from ..constants import (
    BOX_SIZE, 
    READDY_TOTAL_STEPS, 
    CYTOSIM_SCALE_FACTOR,
)
from ..analysis.compression_metrics.compression_analysis import (
    COMPRESSIONMETRIC,
    get_asymmetry_of_peak,
    get_average_distance_from_end_to_end_axis,
    get_bending_energy_from_trace,
    get_contour_length_from_trace,
    get_third_component_variance,
)
from ..simulation.readdy import ReaddyPostProcessor, load_readdy_fiber_points
from .spatial_annotator import SpatialAnnotator


def _empty_scatter_plots(
    total_steps: int = -1,
    times: np.ndarray = None,
    time_units: str = None,
) -> Dict[COMPRESSIONMETRIC, ScatterPlotData]:
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
        COMPRESSIONMETRIC.AVERAGE_PERP_DISTANCE: ScatterPlotData(
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
        COMPRESSIONMETRIC.CALC_BENDING_ENERGY: ScatterPlotData(
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
        COMPRESSIONMETRIC.NON_COPLANARITY: ScatterPlotData(
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
        COMPRESSIONMETRIC.PEAK_ASYMMETRY: ScatterPlotData(
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
        COMPRESSIONMETRIC.CONTOUR_LENGTH: ScatterPlotData(
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


def _generate_plot_data(fiber_points):
    """
    Calculate plot traces from fiber_points.
    """
    n_points = int(fiber_points.shape[2] / 3.0)
    result = {
        COMPRESSIONMETRIC.AVERAGE_PERP_DISTANCE: [],
        COMPRESSIONMETRIC.CALC_BENDING_ENERGY: [],
        COMPRESSIONMETRIC.NON_COPLANARITY: [],
        COMPRESSIONMETRIC.PEAK_ASYMMETRY: [],
        COMPRESSIONMETRIC.CONTOUR_LENGTH: [],
    }
    total_steps = fiber_points.shape[0]
    for time_ix in range(total_steps):
        points = fiber_points[time_ix][0].reshape((n_points, 3))
        result[COMPRESSIONMETRIC.AVERAGE_PERP_DISTANCE].append(
            get_average_distance_from_end_to_end_axis(
                polymer_trace=points,
            )
        )
        result[COMPRESSIONMETRIC.CALC_BENDING_ENERGY].append(
            CYTOSIM_SCALE_FACTOR
            * get_bending_energy_from_trace(
                polymer_trace=points,
            )
        )
        result[COMPRESSIONMETRIC.NON_COPLANARITY].append(
            get_third_component_variance(
                polymer_trace=points,
            )
        )
        result[COMPRESSIONMETRIC.PEAK_ASYMMETRY].append(
            get_asymmetry_of_peak(
                polymer_trace=points,
            )
        )
        result[COMPRESSIONMETRIC.CONTOUR_LENGTH].append(
            get_contour_length_from_trace(
                polymer_trace=points,
            )
        )
    return result


def _add_plots(
    converter: TrajectoryConverter, 
    fiber_points: np.ndarry,
    times: np.ndarray,
):
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
    Get a TrajectoryData to visualize an actin trajectory in Simularium.
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
    path_to_readdy_h5 = os.path.join(LOCAL_DOWNLOADS_PATH, f"{series_key}_{rep_ix}.h5")
    converter = _load_readdy_simularium(path_to_readdy_h5, series_key)
    
    # load data shaped for analysis from a pickle if it exists, otherwise save one
    post_processor, fiber_chain_ids, axis_positions, fiber_points, times = load_readdy_fiber_points(
        bucket, series_name, series_key, rep_ix, n_timepoints, n_monomer_points
    )  
    _add_plots(converter, fiber_points, times)
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
    overwrite_existing: bool = True,
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
    overwrite_existing
        Overwrite any outputs that already exist?
    """
    for condition_key in condition_keys:
        series_key = f"{series_name}_{condition_key}" if condition_key else series_name

        for rep_ix in range(n_replicates):
            local_h5_path = os.path.join(LOCAL_DOWNLOADS_PATH, f"{series_key}_{rep_ix}.h5")
            output_key = f"{series_name}/viz/{series_key}_{rep_ix}.simularium"

            # Skip if output file already exists.
            if not overwrite_existing and check_key(bucket, output_key):
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
    Build a converter from a single Cytosim trajectory to Simularium.
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
    _add_plots(
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
    overwrite_existing: bool = True,
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
    overwrite_existing
        Overwrite any outputs that already exist?
    """
    for condition_key in condition_keys:
        series_key = f"{series_name}_{condition_key}" if condition_key else series_name

        for index, seed in enumerate(random_seeds):
            output_key = f"{series_name}/viz/{series_key}_{seed:06d}.simularium"
            
            # Skip if output file already exists.
            if not overwrite_existing and check_key(bucket, output_key):
                print(f"Simularium visualization [ { output_key } ] already exists. Skipping.")
                continue
            
            output_key_template = f"{series_name}/outputs/{series_key}_{index}/%s"
            fiber_points_data = load_text(
                bucket, output_key_template % "fiber_points.txt"
            )
            singles_data = load_text(
                bucket, output_key_template % "singles.txt"
            )
            local_output_path = os.path.join(LOCAL_DOWNLOADS_PATH, output_key)
            _visualize_cytosim_trajectory(
                fiber_points_data, singles_data, local_output_path, n_timepoints
            )
            
            upload_file_to_s3(bucket, local_output_path, output_key)
