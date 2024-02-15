import argparse
import math
import os
from typing import Dict, Tuple

import boto3
import numpy as np
import pandas as pd
from pint import UnitRegistry
from botocore.exceptions import ClientError
from scipy.spatial.transform import Rotation
from simulariumio import (DISPLAY_TYPE, AgentData, CameraData, DisplayData,
                          MetaData, ScatterPlotData, TrajectoryConverter,
                          TrajectoryData, UnitData)
from simulariumio.filters import EveryNthTimestepFilter

from subcell_analysis.compression_analysis import COMPRESSIONMETRIC
from subcell_analysis.compression_workflow_runner import \
    compression_metrics_workflow
from subcell_analysis.cytosim.post_process_cytosim import cytosim_to_simularium
from subcell_analysis.compression_analysis import (
    get_average_distance_from_end_to_end_axis,
    get_bending_energy_from_trace,
    get_third_component_variance,
    get_asymmetry_of_peak,
    get_contour_length_from_trace,
)


CYTOSIM_CONDITIONS = {
    "0001" : 0.48,
    "0002" : 1.5,
    "0003" : 4.7,
    "0004" : 15,
    "0005" : 47,
    "0006" : 150,
}
READDY_CONDITIONS = [
    4.7,
    15,
    47,
    150,
]
NUM_REPEATS = 5
COMBINED_CSV_PATH = "combined_actin_compression_dataset_subsampled.csv"
TOTAL_STEPS = 200
POINTS_PER_FIBER = 200
CYTOSIM_SCALE_FACTOR = 1000.
BOX_SIZE = 600.


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualizes ReaDDy and Cytosim actin simulations"
    )
    parser.add_argument('--combined', action=argparse.BooleanOptionalAction)
    parser.set_defaults(combined=False)
    parser.add_argument('--cytosim', action=argparse.BooleanOptionalAction)
    parser.set_defaults(cytosim=False)
    parser.add_argument('--upload', action=argparse.BooleanOptionalAction)
    parser.set_defaults(upload=False)
    return parser.parse_args()
    
s3_client = boto3.client("s3")
def download_s3_file(bucket_name, s3_path, dest_path) -> bool:
    """
    Download files (skip files that already exist)
    """
    if os.path.isfile(dest_path):
        # already downloaded
        return False
    try:
        s3_client.download_file(
            bucket_name,
            s3_path,
            dest_path,
        )
        print(f"Downloaded {dest_path}")
        return True
    except ClientError:
        print(f"!!! Failed to download {s3_path}")
        return False
        
def upload_file_to_s3(bucket_name, src_path, s3_path) -> bool:
    """
    Upload a file to an S3 bucket
    """
    if not os.path.isfile(src_path):
        print(f"!!! File does not exist to upload {src_path}")
        return False
    try:
        s3_client.upload_file(
            src_path, 
            bucket_name, 
            s3_path
        )
        print(f"Uploaded to {s3_path}")
        return True
    except ClientError:
        print(f"!!! Failed to upload {src_path}")
        return False

def make_download_dirs():
    if not os.path.isdir("data"):
        os.makedirs("data")
    if not os.path.isdir("data/aws_downloads"):
        os.makedirs("data/aws_downloads")

def download_combined_csv_data():
    make_download_dirs()
    # combined csv is in ReaDDy bucket for now
    download_s3_file(
        bucket_name="readdy-working-bucket",
        s3_path=f"outputs/{COMBINED_CSV_PATH}",
        dest_path=f"data/aws_downloads/{COMBINED_CSV_PATH}",
    )

def download_cytosim_trajectory_data():
    make_download_dirs()
    for condition in CYTOSIM_CONDITIONS.keys():
        for repeat in range(NUM_REPEATS):
            download_s3_file(
                bucket_name="cytosim-working-bucket",
                s3_path=f"vary_compress_rate{condition}/outputs/{repeat}/fiber_points.txt",
                dest_path=f"data/aws_downloads/fiber_points_{condition}_{repeat}.txt",
            )
            download_s3_file(
                bucket_name="cytosim-working-bucket",
                s3_path=f"vary_compress_rate{condition}/outputs/{repeat}/singles.txt",
                dest_path=f"data/aws_downloads/singles_{condition}_{repeat}.txt",
            )

def empty_scatter_plots(
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
        COMPRESSIONMETRIC.AVERAGE_PERP_DISTANCE : ScatterPlotData(
            title="Average Perpendicular Distance",
            xaxis_title=xlabel,
            yaxis_title="distance (nm)",
            xtrace=xtrace,
            ytraces={
                "<<<": np.zeros(total_steps),
                ">>>": 85.0 * np.ones(total_steps),
                "filament": [],
            },
            render_mode="lines"
        ),
        COMPRESSIONMETRIC.CALC_BENDING_ENERGY : ScatterPlotData(
            title="Bending Energy",
            xaxis_title=xlabel,
            yaxis_title="energy",
            xtrace=xtrace,
            ytraces={
                "<<<": np.zeros(total_steps),
                ">>>": 10.0 * np.ones(total_steps),
                "filament": [],
            },
            render_mode="lines"
        ),
        COMPRESSIONMETRIC.NON_COPLANARITY : ScatterPlotData(
            title="Non-coplanarity",
            xaxis_title=xlabel,
            yaxis_title="3rd component variance from PCA",
            xtrace=xtrace,
            ytraces={
                "<<<": np.zeros(total_steps),
                ">>>": 0.03 * np.ones(total_steps),
                "filament": [],
            },
            render_mode="lines"
        ),
        COMPRESSIONMETRIC.PEAK_ASYMMETRY : ScatterPlotData(
            title="Peak Asymmetry",
            xaxis_title=xlabel,
            yaxis_title="normalized peak distance",
            xtrace=xtrace,
            ytraces={
                "<<<": np.zeros(total_steps),
                ">>>": 0.5 * np.ones(total_steps),
                "filament": [],
            },
            render_mode="lines"
        ),
        COMPRESSIONMETRIC.CONTOUR_LENGTH : ScatterPlotData(
            title="Contour Length",
            xaxis_title=xlabel,
            yaxis_title="filament contour length (nm)",
            xtrace=xtrace,
            ytraces={
                "<<<": 480 * np.ones(total_steps),
                ">>>": 505 * np.ones(total_steps),
                "filament": [],
            },
            render_mode="lines"
        ),
    }

def rmsd(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
    return np.sqrt(((((vec1 - vec2) ** 2)) * 3).mean())

def align(fibers: np.ndarray) -> np.ndarray:
    """
    Rotationally align the given fibers around the x-axis.

    Parameters
    ----------
    fiber_points: np.ndarray (shape = time x fiber x (3 * points_per_fiber))
        Array containing the flattened x,y,z positions of control points
        for each fiber at each time.
        
    Returns
    ----------
    aligned_data: np.ndarray
        The given data aligned.        
    """
    # get angle to align each fiber at the last time point
    align_by = []
    points_per_fiber = int(fibers.shape[2] / 3)
    ref = fibers[-1][0].copy().reshape((points_per_fiber, 3))
    for fiber_ix in range(len(fibers[-1])):
        best_rmsd = math.inf
        for angle in np.linspace(0, 2 * np.pi, 1000):
            rot = Rotation.from_rotvec(angle * np.array([1, 0, 0]))
            new_vec = Rotation.apply(
                rot, fibers[-1][fiber_ix].copy().reshape((points_per_fiber, 3))
            )
            test_rmsd = rmsd(new_vec, ref)
            if test_rmsd < best_rmsd:
                best_angle = angle
                best_rmsd = test_rmsd
        align_by.append(best_angle)
    # align all the fibers to ref across all time points
    aligned = np.zeros_like(fibers)
    for fiber_ix in range(fibers.shape[1]):
        rot = Rotation.from_rotvec(align_by[fiber_ix] * np.array([1, 0, 0]))
        for time_ix in range(fibers.shape[0]):
            fiber = fibers[time_ix][fiber_ix].copy().reshape((points_per_fiber, 3))
            new_fiber = Rotation.apply(rot, fiber)
            aligned[time_ix][fiber_ix] = new_fiber.flatten()
    return aligned

def save_combined_simularium():
    df = pd.read_csv(f"data/aws_downloads/{COMBINED_CSV_PATH}")
    simulators = ["cytosim", "readdy"]
    colors = {
        "cytosim" : [
            "#4DFE8A",
            "#c1fe4d",
            "#fee34d",
            "#fe8b4d",
        ],
        "readdy" : [
            "#94dbfc",
            "#627EFB",
            "#b594fc",
            "#e994fc",
        ],
    }
    total_conditions = NUM_REPEATS * len(simulators) * len(READDY_CONDITIONS)
    subpoints = np.zeros((TOTAL_STEPS, total_conditions, 3 * POINTS_PER_FIBER))
    types_per_step = []
    display_data={}
    scatter_plots = empty_scatter_plots(total_steps=TOTAL_STEPS)
    # these metrics need to be multiplied by 1000 in cytosim because of different units
    cytosim_metrics_to_scale = [
        COMPRESSIONMETRIC.AVERAGE_PERP_DISTANCE,
        COMPRESSIONMETRIC.CONTOUR_LENGTH,
    ]
    for sim_ix, simulator in enumerate(simulators):
        sim_df = df.loc[df["simulator"] == simulator]
        sim_df.sort_values(by=["repeat", "simulator", "velocity", "time", "monomer_ids"])
        for condition_ix, condition in enumerate(READDY_CONDITIONS):
            condition_df = sim_df.loc[sim_df["velocity"] == condition]
            for repeat_ix in range(NUM_REPEATS):
                rep_df = condition_df.loc[condition_df["repeat"] == repeat_ix]
                for time_ix in range(TOTAL_STEPS):
                    ix = (
                        (sim_ix * len(READDY_CONDITIONS) * NUM_REPEATS) 
                        + (condition_ix * NUM_REPEATS) + repeat_ix
                    )
                    subpoints[time_ix][ix] = (
                        (CYTOSIM_SCALE_FACTOR if simulator == "cytosim" else 1) * 
                        np.array(rep_df[time_ix * TOTAL_STEPS:(time_ix + 1) * TOTAL_STEPS][["xpos", "ypos", "zpos"]]
                        ).flatten()
                    )
                types_per_step.append(f"{simulator}#{condition} um/s {repeat_ix}")
                display_data[types_per_step[-1]] = DisplayData(
                    name=types_per_step[-1],
                    display_type=DISPLAY_TYPE.FIBER,
                    color=colors[simulator][condition_ix],
                )
                metrics_df = compression_metrics_workflow(rep_df.copy(), list(scatter_plots.keys()))
                metrics_df = metrics_df[metrics_df["monomer_ids"] == 0]
                for metric in scatter_plots:
                    scale_factor = CYTOSIM_SCALE_FACTOR if (
                        (simulator == "cytosim" and metric in cytosim_metrics_to_scale) 
                        or metric == COMPRESSIONMETRIC.CALC_BENDING_ENERGY
                    ) else 1.
                    scatter_plots[metric].ytraces[types_per_step[-1]] = (
                        scale_factor * np.array(metrics_df[metric.value])
                    )
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
            times=np.arange(TOTAL_STEPS),
            n_agents=total_conditions * np.ones((TOTAL_STEPS)),
            viz_types=1001 * np.ones((TOTAL_STEPS, total_conditions)),  # fiber viz type = 1001
            unique_ids=np.array(TOTAL_STEPS * [list(range(total_conditions))]),
            types=TOTAL_STEPS * [types_per_step],
            positions=np.zeros((TOTAL_STEPS, total_conditions, 3)),
            radii=np.ones((TOTAL_STEPS, total_conditions)),
            n_subpoints=3 * POINTS_PER_FIBER * np.ones((TOTAL_STEPS, total_conditions)),
            subpoints=align(subpoints),
            display_data=display_data,
        ),
        time_units=UnitData("count"),  # frames
        spatial_units=UnitData("nm"),  # nanometer
    )
    converter = TrajectoryConverter(traj_data)
    for metric, plot in scatter_plots.items():
        converter.add_plot(plot, "scatter")
    converter.save(f"data/actin_compression")

def time_increment(raw_total_steps):
    """
    Find a time increment to get the total steps close to 1000
    """
    if raw_total_steps < 2000:
        return 1
    magnitude = math.floor(math.log(raw_total_steps, 10))
    amount = raw_total_steps / 10 ** magnitude
    if amount > 5:
        return 5 * 10 ** (magnitude - 3)
    return 10 ** (magnitude - 3)

ureg = UnitRegistry()
def find_time_units(raw_time: float) -> Tuple[str, float]:
    """
    Get the compact time units and a multiplier to put the times in those units
    given a time value in seconds
    """
    time = raw_time * ureg.second
    time = time.to_compact()
    return "{:~}".format(time.units), time.magnitude / raw_time

def save_cytosim_simularium():
    if not os.path.isdir("data/cytosim_outputs"):
        os.makedirs("data/cytosim_outputs")
    for condition in CYTOSIM_CONDITIONS.keys():
        velocity = CYTOSIM_CONDITIONS[condition]
        for repeat_ix in range(NUM_REPEATS):
            fiber_points_path = f"data/aws_downloads/fiber_points_{condition}_{repeat_ix}.txt"
            singles_path = f"data/aws_downloads/singles_{condition}_{repeat_ix}.txt"
            output_path = f"data/cytosim_outputs/actin_compression_velocity={velocity}_{repeat_ix}"
            if os.path.isfile(f"{output_path}.simularium"):
                print(f"Skipping v={velocity} #{repeat_ix}, output file already exists")
                continue
            if not os.path.isfile(fiber_points_path):
                continue
            if not os.path.isfile(singles_path):
                singles_path = None
            print(f"Converting Cytosim v={velocity} #{repeat_ix}")
            traj_data = cytosim_to_simularium(
                title=f"Actin Compression v={velocity} {repeat_ix}",
                fiber_points_path=fiber_points_path,
                singles_path=singles_path,
                scale_factor=CYTOSIM_SCALE_FACTOR,
            )
            converter = TrajectoryConverter(traj_data)
            time_inc = time_increment(converter._data.agent_data.times.shape[0])
            if time_inc > 1:
                converter._data = converter.filter_data([
                    EveryNthTimestepFilter(
                        n=time_inc,
                    ),
                ])
            time_units, time_multiplier = find_time_units(converter._data.agent_data.times[-1])
            converter._data.agent_data.times *= time_multiplier
            converter._data.time_units = UnitData(time_units)
            # plots
            total_steps = converter._data.agent_data.subpoints.shape[0]
            n_points = int(converter._data.agent_data.subpoints.shape[2] / 3.)
            perp_dist = []
            bending_energy = []
            non_coplanarity = []
            peak_asym = []
            contour_length = []
            for time_ix in range(total_steps):
                points = converter._data.agent_data.subpoints[time_ix][0].reshape((n_points, 3))
                perp_dist.append(
                    get_average_distance_from_end_to_end_axis(
                    polymer_trace=points,
                ))
                bending_energy.append(
                    CYTOSIM_SCALE_FACTOR * get_bending_energy_from_trace(
                    polymer_trace=points,
                ))
                non_coplanarity.append(
                    get_third_component_variance(
                    polymer_trace=points,
                ))
                peak_asym.append(
                    get_asymmetry_of_peak(
                    polymer_trace=points,
                ))
                contour_length.append(
                    get_contour_length_from_trace(
                    polymer_trace=points,
                ))
            scatter_plots = empty_scatter_plots(
                times=converter._data.agent_data.times, 
                time_units=time_units,
            )
            scatter_plots[COMPRESSIONMETRIC.AVERAGE_PERP_DISTANCE
            ].ytraces["filament"] = np.array(perp_dist)
            scatter_plots[
                COMPRESSIONMETRIC.CALC_BENDING_ENERGY
            ].ytraces["filament"] = np.array(bending_energy)
            scatter_plots[
                COMPRESSIONMETRIC.NON_COPLANARITY
            ].ytraces["filament"] = np.array(non_coplanarity)
            scatter_plots[
                COMPRESSIONMETRIC.PEAK_ASYMMETRY
            ].ytraces["filament"] = np.array(peak_asym)
            scatter_plots[
                COMPRESSIONMETRIC.CONTOUR_LENGTH
            ].ytraces["filament"] = np.array(contour_length)
            for metric, plot in scatter_plots.items():
                converter.add_plot(plot, "scatter")
            converter.save(output_path)

def upload_cytosim_trajectories():
    for condition in CYTOSIM_CONDITIONS.keys():
        velocity = CYTOSIM_CONDITIONS[condition]
        for repeat in range(NUM_REPEATS):
            upload_file_to_s3(
                bucket_name="cytosim-working-bucket",
                src_path=f"data/cytosim_outputs/actin_compression_velocity={velocity}_{repeat}.simularium",
                s3_path=f"simularium/actin_compression_velocity={velocity}_{repeat}.simularium",
            )

def main():
    args = parse_args()
    if not (args.combined or args.cytosim):
        print("Please specify either --combined or --cytosim arguments")
    if args.combined:
        # save one simularium file with all cytosim and readdy trajectories
        download_combined_csv_data()
        save_combined_simularium()
        if args.upload:
            upload_file_to_s3(
                bucket_name="readdy-working-bucket",
                src_path=f"data/actin_compression.simularium",
                s3_path=f"outputs/actin_compression_cytosim_readdy.simularium",
            )
    elif args.cytosim:
        # save an individual simularium file for each cytosim trajectory
        download_cytosim_trajectory_data()
        save_cytosim_simularium()
        if args.upload:
            upload_cytosim_trajectories()


if __name__ == "__main__":
    main()