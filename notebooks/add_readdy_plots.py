import os

import boto3
import numpy as np
from botocore.exceptions import ClientError
from simularium_readdy_models.visualization import ActinVisualization


BUCKET_NAME="readdy-working-bucket"
    
s3_client = boto3.client("s3")

def download_h5_file(file_name):
    """
    Download files (skip files that already exist)
    """
    if os.path.isfile(f"data/aws_downloads/{file_name}.h5"):
        return
    try:
        s3_client.download_file(
            BUCKET_NAME,
            f"outputs/{file_name}.h5",
            f"data/aws_downloads/{file_name}.h5",
        )
        print(f"Downloaded {file_name}")
    except ClientError:
        print(f"!!! Failed to download {file_name}")

def download_data(conditions, num_repeats):
    if not os.path.isdir("data"):
        os.makedirs("data")
    if not os.path.isdir("data/aws_downloads"):
        os.makedirs("data/aws_downloads")
    for repeat in range(num_repeats):
        download_h5_file(
            f"actin_compression_baseline_{repeat}_0"
        )
        for condition in conditions:
            download_h5_file(
                f"actin_compression_velocity={condition}_{repeat}"
            )
            
def add_plots(parameters, total_steps, conditions, num_repeats):
    """
    Re-visualize the trajectories to add plots
    """
    for repeat in range(num_repeats):
        ActinVisualization.analyze_and_visualize_trajectory(
            f"data/aws_downloads/actin_compression_baseline_{repeat}_0",
            total_steps["baseline"],
            parameters,
        )
        for condition in conditions:
            ActinVisualization.analyze_and_visualize_trajectory(
                f"data/aws_downloads/actin_compression_velocity={condition}_{repeat}",
                total_steps[condition],
                parameters,
            )
            
def upload_simularium_file(file_name):
    """
    Upload files (warning for files that fail)
    """
    if not os.path.isfile(f"data/aws_downloads/{file_name}.h5.simularium"):
        print(f"!!! Not found, could not upload {file_name}")
        return
    try:
        s3_client.upload_file(
            f"data/aws_downloads/{file_name}.h5.simularium",
            BUCKET_NAME, 
            f"outputs/{file_name}.h5.simularium",
        )
        print(f"Uploaded {file_name}")
    except ClientError as e:
        print(f"!!! Failed to upload {file_name}")
    
        
def upload_to_s3(conditions, num_repeats):
    for repeat in range(num_repeats):
        upload_simularium_file(
            f"actin_compression_baseline_{repeat}_0"
        )
        for condition in conditions:
            upload_simularium_file(
                f"actin_compression_velocity={condition}_{repeat}"
            )


def main():
    num_repeats = 3
    conditions = [
        "4.7",
        "15",
        "47",
        "150",
    ]
    total_steps = {
        "4.7" : 3.2E8,
        "15" : 1E8,
        "47" : 3.2E7,
        "150" : 1E7,
        "baseline" : 1E7,
    }
    parameters = {
        "box_size" : np.array([600.0, 600.0, 600.0]),
        "internal_timestep" : 0.1,
        "longitudinal_bonds" : True,
        "periodic_boundary" : False,
        "plot_actin_structure" : True,
        "plot_actin_compression" : True,
        "visualize_edges" : True,
        "visualize_normals" : True,
        "visualize_control_pts" : True,
    }
    # download_data(conditions, num_repeats)
    # add_plots(parameters, total_steps, conditions, num_repeats)
    upload_to_s3(conditions, num_repeats)

if __name__ == "__main__":
    main()