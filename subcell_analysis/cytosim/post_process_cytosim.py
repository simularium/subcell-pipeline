from pathlib import Path
from typing import Dict

import boto3
import numpy as np
import pandas as pd


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
                #                     fiber_point=0
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

    file_name = "actin-forces"

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
            print(f"Processing index {index} and repeat {repeat}")

            segenergy[index, repeat] = read_cytosim_s3_file(
                "cytosim-working-bucket",
                f"{config}/outputs/{repeat}/fiber_segment_curvature.txt",
            )
            print(type(segenergy[index][repeat]))

            fibenergylabels[index, repeat] = read_cytosim_s3_file(
                "cytosim-working-bucket",
                f"{config}/outputs/{repeat}/fiber_energy_labels.txt",
            )
            print(type(fibenergylabels[index][repeat]))
            fibenergy[index, repeat] = read_cytosim_s3_file(
                "cytosim-working-bucket", f"{config}/outputs/{repeat}/fiber_energy.txt"
            )
            print(type(fibenergy[index][repeat]))
            convert_and_save_dataframe(
                fibenergy[index][repeat],
                segenergy[index][repeat],
                suffix=f"{index}_{repeat}",
            )
