from pathlib import Path
from typing import Dict

import boto3
import numpy as np
import pandas as pd
from IPython.display import Image
from simulariumio.cytosim import CytosimConverter, CytosimData, CytosimObjectInfo
from simulariumio import (
    MetaData,
    DisplayData,
    DISPLAY_TYPE,
    ModelMetaData,
    InputFileData,
)


def convert_and_save_dataframe(
    fiber_energy_all: list,
    fiber_forces_all: list,
    suffix: str = None,
    rigidity: float = 0.041,
) -> pd.DataFrame:
    """
    Convert cytosim output to pandas dataframe and save to csv.
    """
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
            # print(line.split())
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
            #                 figure out if you're on the first, second fiber point etc
            if int(fid) == int(fiber_id):
                fiber_point += 1
            else:
                fiber_point = 0
                fid += 1
            #                     print('id: '+str(fid))
            singles[str(fiber_id) + "_" + str(fiber_point)] = {
                "fiber_id": int(fiber_id),
                "xpos": float(xpos),
                "ypos": float(ypos),
                "zpos": float(zpos),
                "xforce": float(xforce),
                "yforce": float(yforce),
                "zforce": float(zforce),
                "segment_curvature": float(segment_curvature),
            }

    all_outputs = pd.concat(outputs, keys=timepoints_forces, names=["time", "id"])
    # all_outputs = all_outputs.swaplevel('time','id',axis=0).sort_index()
    all_outputs["force_magnitude"] = np.sqrt(
        np.square(all_outputs["xforce"])
        + np.square(all_outputs["yforce"])
        + np.square(all_outputs["zforce"])
    )

    #  Segment bending energy, in pN nm
    all_outputs["segment_energy"] = all_outputs["segment_curvature"] * rigidity * 1000
    # fiber_forces_outputs_allruns.append(all_outputs)

    save_folder = Path("dataframes")
    save_folder.mkdir(exist_ok=True, parents=True)

    file_name = "actin_forces"

    if suffix is not None:
        file_name += suffix

    all_outputs.to_csv(save_folder / f"{file_name}.csv")

    print(f"Saved Output to {save_folder/f'{file_name}.csv'}")
    all_outputs.tail()
    return all_outputs


def read_cytosim_s3_file(bucket_name: str, file_name: str) -> list:
    """
    Read a file from S3 bucket and return a list of lines.
    """
    s3 = boto3.client("s3")
    try:
        response = s3.get_object(Bucket=bucket_name, Key=file_name)
        file_content = response["Body"].read()
        return file_content.decode("utf-8").splitlines()
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return []


def get_s3_file(bucket_name: str, file_name: str) -> object:
    s3 = boto3.client("s3")
    response = s3.get_object(Bucket=bucket_name, Key=file_name)
    return response["Body"].read()


def create_dataframes_for_repeats(
    bucket_name: str, num_repeats: int, configs: list
) -> None:
    """
    Create dataframes for all repeats of given configs.
    """
    segenergy = np.empty((len(configs), num_repeats), dtype=object)
    fibenergy = np.empty((len(configs), num_repeats), dtype=object)
    fibenergylabels = np.empty((len(configs), num_repeats), dtype=object)
    for index, config in enumerate(configs):
        for repeat in range(num_repeats):
            print(f"Processing config {config} and repeat {repeat}")

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
                fibenergy[index][repeat],
                segenergy[index][repeat],
                suffix=f"_{config}_{repeat}",
            )


def cytosim_to_simularium(
    path, box_size=2, scale_factor=10, color=None, actin_number=0
):
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
