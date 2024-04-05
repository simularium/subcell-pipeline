from pathlib import Path
from typing import Dict, Optional

import boto3
import numpy as np
import pandas as pd
from simulariumio import (
    DISPLAY_TYPE,
    DisplayData,
    InputFileData,
    MetaData,
    ModelMetaData,
)
from simulariumio.cytosim import CytosimData, CytosimObjectInfo

save_folder_path = Path("../data/dataframes")


def convert_and_save_dataframe(
    fiber_energy_all: list,
    fiber_forces_all: list,
    file_name: str = "cytosim_actin_compression",
    suffix: str = "",
    rigidity: float = 0.041,
    save_folder: Path = save_folder_path,
    velocity: float = 0.15,
    repeat: int = 0,
    simulator: str = "cytosim",
) -> pd.DataFrame:
    # Convert cytosim output to pandas dataframe and saves to csv.
    bending_energies = []
    for line in fiber_energy_all:
        line = line.strip()
        bending_energy = float(line.split()[2])
        bending_energies.append(bending_energy)

    single_all_lines = fiber_forces_all
    timepoints_forces = []
    outputs = []
    fid = 0
    for line in single_all_lines:
        line = line.strip()
        if line.startswith("%"):
            if line.startswith("% time"):
                time = float(line.split(" ")[-1])
                timepoints_forces.append(time)
                singles: Dict[str, object] = {}
            elif line.startswith("% end"):
                df = pd.DataFrame.from_dict(singles, orient="index")
                outputs.append(df)
                fiber_point = 0
                fid = 0
                # print 'finished parsing ' + rundir + ' timepoint ' + str(time)
        elif len(line.split()) > 0:
            [
                fiber_id,
                xpos,
                ypos,
                zpos,
                xforce,
                yforce,
                zforce,
                segment_curvature,
            ] = line.split()
            # figure out if you're on the first, second fiber point etc
            if int(fid) == int(fiber_id):
                fiber_point += 1
            else:
                fiber_point = 0
                fid += 1

            singles[str(fiber_id) + "_" + str(fiber_point)] = {
                "fiber_id": int(fiber_id),
                "xpos": float(xpos),
                "ypos": float(ypos),
                "zpos": float(zpos),
                "xforce": float(xforce),
                "yforce": float(yforce),
                "zforce": float(zforce),
                "segment_curvature": float(segment_curvature),
                "velocity": velocity,
                "repeat": repeat,
                "simulator": simulator,
            }

    all_outputs = pd.concat(outputs, keys=timepoints_forces, names=["time", "id"])

    all_outputs["force_magnitude"] = np.sqrt(
        np.square(all_outputs["xforce"])
        + np.square(all_outputs["yforce"])
        + np.square(all_outputs["zforce"])
    )

    #  Segment bending energy, in pN nm
    all_outputs["segment_energy"] = all_outputs["segment_curvature"] * rigidity * 1000

    all_outputs.to_csv(save_folder / f"{file_name}{suffix}.csv")

    print(f"Saved Output to {save_folder/f'{file_name}.csv'}")

    return all_outputs


def read_cytosim_s3_file(bucket_name: str, file_name: str) -> list:
    # Read a file from S3 bucket and return a list of lines.

    s3 = boto3.client("s3")
    try:
        response = s3.get_object(Bucket=bucket_name, Key=file_name)
        file_content = response["Body"].read()
        return file_content.decode("utf-8").splitlines()
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return []


def get_s3_file(bucket_name: str, file_name: str) -> object:
    """
    Retrieves a file from an S3 bucket.

    Args:
        bucket_name (str): The name of the S3 bucket.
        file_name (str): The name of the file to retrieve.

    Returns
    -------
        object: The contents of the file as a byte string.
    """
    s3 = boto3.client("s3")
    response = s3.get_object(Bucket=bucket_name, Key=file_name)
    return response["Body"].read()


def create_dataframes_for_repeats(
    bucket_name: str,
    num_repeats: int,
    configs: list,
    save_folder: Path,
    file_name: str = "cytosim_actin_compression",
    velocities: Optional[list] = None,
    overwrite: bool = True,
) -> None:
    """
    Create dataframes for repeated simulations in Cytosim.

    Parameters
    ----------
    bucket_name : str
        The name of the bucket.
    num_repeats : int
        The number of repeats.
    configs : list
        The list of configurations.
    save_folder : Path
        The path to the save folder.
    file_name : str, optional
        The name of the file. Defaults to "cytosim_actin_compression".
    velocities : list, optional
        The list of velocities. Defaults to None.
    overwrite : bool, optional
        Whether to overwrite existing files. Defaults to True.

    Returns
    -------
    None
        This function does not return anything.
    """
    segenergy = np.empty((len(configs), num_repeats), dtype=object)
    fibenergy = np.empty((len(configs), num_repeats), dtype=object)
    fibenergylabels = np.empty((len(configs), num_repeats), dtype=object)
    for index, config in enumerate(configs):
        velocity = velocities[index] if velocities is not None else config
        for repeat in range(num_repeats):
            print(
                f"Processing config {config}, velocity {velocity} and repeat {repeat}"
            )
            suffix = f"_velocity_{velocity}_repeat_{repeat}"
            file_path = save_folder / f"{file_name}{suffix}.csv"
            if file_path.is_file() and not overwrite:
                print(f"File {file_path.name} already exists. Skipping.")
                continue

            segenergy[index, repeat] = read_cytosim_s3_file(
                bucket_name,
                f"{config}/outputs/{repeat}/fiber_segment_curvature.txt",
            )
            fibenergylabels[index, repeat] = read_cytosim_s3_file(
                bucket_name,
                f"{config}/outputs/{repeat}/fiber_energy_labels.txt",
            )
            fibenergy[index, repeat] = read_cytosim_s3_file(
                bucket_name, f"{config}/outputs/{repeat}/fiber_energy.txt"
            )
            convert_and_save_dataframe(
                fiber_energy_all=fibenergy[index][repeat],
                fiber_forces_all=segenergy[index][repeat],
                file_name=file_name,
                suffix=suffix,
                save_folder=save_folder,
                velocity=velocity,
                repeat=repeat,
                simulator="cytosim",
            )


def cytosim_to_simularium(
    path: str,
    box_size: float = 2,
    scale_factor: float = 10,
    color: list = None,
    actin_number: int = 0,
) -> CytosimData:
    example_data = CytosimData(
        meta_data=MetaData(
            box_size=np.array([box_size, box_size, box_size]),
            scale_factor=scale_factor,
            trajectory_title="Some parameter set",
            model_meta_data=ModelMetaData(
                title="Some agent-based model",
                version="8.1",
                authors="A Modeler",
                description=("An agent-based model run with some parameter set"),
                doi="10.1016/j.bpj.2016.02.002",
                source_code_url="https://github.com/simularium/simulariumio",
                source_code_license_url="https://github.com/simularium/simulariumio/blob/main/LICENSE",
                input_data_url="https://allencell.org/path/to/native/engine/input/files",
                raw_output_data_url="https://allencell.org/path/to/native/engine/output/files",
            ),
        ),
        object_info={
            "fibers": CytosimObjectInfo(
                cytosim_file=InputFileData(
                    file_path=path,
                ),
                display_data={
                    1: DisplayData(
                        name=f"actin#{actin_number}",
                        radius=0.02,
                        display_type=DISPLAY_TYPE.FIBER,
                        color=color,
                    )
                },
            ),
        },
    )
    return example_data
