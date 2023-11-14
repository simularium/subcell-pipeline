import argparse
import math
import os
from typing import Dict, List, Tuple

import boto3
import numpy as np
import pandas as pd
from botocore.exceptions import ClientError
from scipy.spatial.transform import Rotation
from simularium_readdy_models.visualization import ActinVisualization
from simulariumio import (DISPLAY_TYPE, AgentData, CameraData, DisplayData,
                          InputFileData, MetaData, ScatterPlotData,
                          TrajectoryConverter, TrajectoryData, UnitData)
from simulariumio.cytosim import (CytosimConverter, CytosimData,
                                  CytosimObjectInfo)

from subcell_analysis.compression_analysis import COMPRESSIONMETRIC
from subcell_analysis.cytosim.post_process_cytosim import cytosim_to_simularium
from subcell_analysis.spatial_aligner import SpatialAligner


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualizes ReaDDy and Cytosim actin simulations"
    )
    # parser.add_argument(
    #     "params_path", help="the file path of an excel file with parameters"
    # )
    parser.add_argument('--sub_sampled', action=argparse.BooleanOptionalAction)
    parser.set_defaults(sub_sampled=True)
    return parser.parse_args()
    
s3_client = boto3.client("s3")
def download_s3_file(bucket_name, s3_path, dest_path):
    """
    Download files (skip files that already exist)
    """
    if os.path.isfile(dest_path):
        return
    try:
        s3_client.download_file(
            bucket_name,
            s3_path,
            dest_path,
        )
        print(f"Downloaded {dest_path}")
    except ClientError:
        print(f"!!! Failed to download {s3_path}")

def download_data(sub_sampled):
    if not os.path.isdir("data"):
        os.makedirs("data")
    if not os.path.isdir("data/aws_downloads"):
        os.makedirs("data/aws_downloads")
    if sub_sampled:
        download_s3_file(
            bucket_name="cytosim-working-bucket",
            s3_path="cytosim_actin_compression_all_velocities_and_repeats.csv",
            dest_path="data/aws_downloads/cytosim_actin_compression_all_velocities_and_repeats.csv",
        )
        download_s3_file(
            bucket_name="readdy-working-bucket",
            s3_path="outputs/readdy_actin_compression_all_velocities_and_repeats.csv",
            dest_path="data/aws_downloads/readdy_actin_compression_all_velocities_and_repeats.csv",
        )
    else:
        for condition in cytosim_conditions.keys():
            for repeat in range(num_repeats):
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
        for condition in readdy_conditions:
            for repeat in range(num_repeats):
                download_s3_file(
                    bucket_name="readdy-working-bucket",
                    s3_path=f"outputs/actin_compression_velocity={condition}_{repeat}.h5",
                    dest_path=f"data/aws_downloads/readdy_velocity={condition}_{repeat}.h5",
                )

def rmsd(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
    return np.sqrt(((((vec1 - vec2) ** 2)) * 3).mean())

def align(fibers: np.ndarray) -> np.ndarray:
    # find the best alignment angle for each fiber
    align_by = []
    ref = fibers[-1][0].copy()
    for fiber_ix in range(len(fibers[-1])):
        best_rmsd = math.inf
        for j in np.linspace(0, 2 * np.pi, 1000):
            r = Rotation.from_rotvec(j * np.array([1, 0, 0]))
            new_vec = Rotation.apply(
                r, fibers[-1][fiber_ix].copy()
            )
            if rmsd(new_vec, ref) < best_rmsd:
                best_rot = j
                best_rmsd = rmsd(new_vec, ref)
        align_by.append(best_rot)
    # align all the different curves to ref
    aligned = np.zeros_like(fibers)
    for fiber_ix in range(len(fibers[-1])):
        r = Rotation.from_rotvec(align_by[fiber_ix] * np.array([1, 0, 0]))
        for time_ix in range(len(fibers)):
            a = fibers[time_ix][fiber_ix].copy()
            new_vec = Rotation.apply(r, a)
            aligned[time_ix][fiber_ix] = new_vec
    return aligned
                
def convert_sub_sampled_to_simularium() -> TrajectoryData:
    df = pd.read_csv("data/aws_downloads/combined_actin_compression_dataset_subsampled.csv")
    first_rep_df = df.loc[df["repeat"] == 0]
    first_rep_df.sort_values(by=["simulator", "velocity", "time", "monomer_ids"])
    total_steps = 200
    simulators = ["cytosim", "readdy"]
    conditions = [
        4.7,
        15,
        47,
        150,
    ]
    total_conditions = 2 * len(conditions)
    points_per_fiber = 200
    subpoints = np.zeros((total_steps, total_conditions, 3 * points_per_fiber))
    types_per_step = []
    display_data={}
    for sim_ix, simulator in enumerate(simulators):
        sim_df = first_rep_df.loc[first_rep_df["simulator"] == simulator]
        for condition_ix, condition in enumerate(conditions):
            condition_df = sim_df.loc[sim_df["velocity"] == condition]
            for time_ix in range(total_steps):
                subpoints[time_ix][(sim_ix * len(conditions)) + condition_ix] = (
                    (1000. if simulator == "cytosim" else 1) * 
                    np.array(condition_df[time_ix * 200:(time_ix + 1) * 200][["xpos", "ypos", "zpos"]]).flatten()
                )
            types_per_step.append(f"{condition} um/s#{simulator}")
            display_data[types_per_step[-1]] = DisplayData(
                name=types_per_step[-1],
                display_type=DISPLAY_TYPE.FIBER,
            )
    box_size = 600.
    return TrajectoryData(
        meta_data=MetaData(
            box_size=np.array([box_size, box_size, box_size]),
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
            viz_types=1001 * np.ones((total_steps, total_conditions)),  # fiber viz type = 1001
            unique_ids=np.array(total_steps * [list(range(total_conditions))]),
            types=total_steps * [types_per_step],
            positions=np.zeros((total_steps, total_conditions, 3)),
            radii=np.ones((total_steps, total_conditions)),
            n_subpoints=3 * points_per_fiber * np.ones((total_steps, total_conditions)),
            subpoints=subpoints, #TODO SpatialAligner.align_fibers_y(subpoints),
            display_data=display_data,
        ),
        time_units=UnitData("count"),  # frames
        spatial_units=UnitData("nm"),  # nanometer
    )

def convert_raw_to_simularium(
    cytosim_conditions: Dict[str,str], 
    readdy_conditions: List[str], 
    num_repeats: int,
) -> Tuple[Dict[str,List[TrajectoryData]], Dict[str,List[TrajectoryData]]]:
    cytosim_traj_data = {}
    for condition in cytosim_conditions.keys():
        cytosim_traj_data[condition] = []
        for repeat in range(num_repeats):
            velocity = cytosim_conditions[condition]
            fiber_points_path=f"data/aws_downloads/fiber_points_{condition}_{repeat}.txt"
            singles_path=f"data/aws_downloads/singles_{condition}_{repeat}.txt"
            if not os.path.isfile(fiber_points_path):
                continue
            if not os.path.isfile(singles_path):
                singles_path = None
            print(f"Converting Cytosim v={velocity} #{repeat}")
            data = cytosim_to_simularium(
                title=f"Actin Compression v={velocity} {repeat}",
                fiber_points_path=fiber_points_path,
                singles_path=singles_path,
                agent_state=f"v={velocity} {repeat}",
            )
            cytosim_traj_data[condition].append(data)
    readdy_traj_data = {}
    box_size = 600.
    total_steps = {
        "4.7" : 3.17e8,
        "15" : 1e8,
        "47" : 3.17e7,
        "150" : 1e7,
    }
    for condition in readdy_conditions:
        readdy_traj_data[condition] = []
        for repeat in range(num_repeats):
            readdy_h5_path=f"data/aws_downloads/readdy_velocity={condition}_{repeat}.h5"
            if not os.path.isfile(readdy_h5_path):
                continue
            print(f"Converting ReaDDy v={condition} #{repeat}")
            readdy_traj_data.append(ActinVisualization.simularium_trajectory(
                path_to_readdy_h5=readdy_h5_path,
                box_size=np.array([box_size, box_size, box_size]),
                total_steps=total_steps[condition],
                time_multiplier=1e-3,  # assume 1e3 recorded steps
                longitudinal_bonds=True,
            ))
    return cytosim_traj_data, readdy_traj_data

def combine_trajectories(
    cytosim_conditions: Dict[str,str], 
    cytosim_traj_data: Dict[str,List[TrajectoryData]],
    readdy_traj_data: Dict[str,List[TrajectoryData]],
    num_repeats: int,
) -> Dict[str,TrajectoryData]:
    traj_data = {}
    for condition in cytosim_conditions.keys():
        velocity = cytosim_conditions[condition]
        print(f"Combining v={velocity}")
        traj_data[condition] = None
        # Cytosim
        if condition in cytosim_traj_data:
            for repeat in range(num_repeats):
                if len(cytosim_traj_data[condition]) <= repeat:
                    continue
                if traj_data[condition] is None:
                    traj_data[condition] = cytosim_traj_data[condition][repeat]
                else:
                    traj_data[condition].append_agents(cytosim_traj_data[condition][repeat].agent_data)
        # ReaDDy
        readdy_condition = cytosim_conditions[condition]
        if readdy_condition in readdy_traj_data:
            for repeat in range(num_repeats):
                if len(readdy_traj_data[readdy_condition]) > repeat:
                    continue
                if traj_data[condition] is None:
                    traj_data[condition] = readdy_traj_data[readdy_condition][repeat]
                else:
                    traj_data[condition].append_agents(readdy_traj_data[readdy_condition][repeat].agent_data)
    return traj_data

def save_trajectories(
    cytosim_conditions: Dict[str,str], 
    traj_data: Dict[str,TrajectoryData]
):
    for condition in cytosim_conditions.keys():
        if traj_data[condition] is None:
            continue
        velocity = cytosim_conditions[condition]
        print(f"Saving v={velocity}")
        TrajectoryConverter(traj_data[condition]).save(f"data/actin_compression_velocity={velocity}")


def main():
    args = parse_args()
    cytosim_conditions = {
        "0001" : "0.48",
        "0002" : "1.5",
        "0003" : "4.7",
        "0004" : "15",
        "0005" : "47",
        "0006" : "150",
    }
    readdy_conditions = [
        "4.7",
        "15",
        "47",
        "150",
    ]
    num_repeats = 3
    download_data(args.sub_sampled)
    if not args.sub_sampled:
        cytosim_traj_data, readdy_traj_data = convert_raw_to_simularium(
            cytosim_conditions, readdy_conditions, num_repeats
        )
        traj_data = combine_trajectories(
            cytosim_conditions, cytosim_traj_data, readdy_traj_data, num_repeats
        )
        save_trajectories(cytosim_conditions, traj_data)
    else:
        traj_data = convert_sub_sampled_to_simularium()
        TrajectoryConverter(traj_data).save(f"data/actin_compression_repeat=0")


if __name__ == "__main__":
    main()