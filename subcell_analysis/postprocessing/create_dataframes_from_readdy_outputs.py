# %% [markdown]
# # Process individual readdy outputs to csv files
# %% imports
import boto3
import numpy as np
import pandas as pd
from pathlib import Path

from subcell_analysis.readdy import (
    ReaddyLoader,
    ReaddyPostProcessor,
)
from subcell_analysis.readdy.readdy_post_processor import array_to_dataframe

# %% [markdown]
# ### setup output folders
current_folder = Path(__file__).parent
base_data_path = current_folder.parents[1] / "data"
readdy_df_path = base_data_path / "dataframes/readdy"
readdy_df_path.mkdir(exist_ok=True, parents=True)
print(readdy_df_path)
# %% [markdown]
# ## setup parameters to download h5 files from s3
redownload = False
h5_folder_path = base_data_path / "readdy_h5_files"
h5_folder_path.mkdir(exist_ok=True, parents=True)
print(h5_folder_path)
bucket_name = "readdy-working-bucket"
# %% [markdown]
# ## setup velocities and repeats
readdy_compression_velocities = [4.7, 15, 47, 150]
num_repeats = 5

# %% [markdown]
# ## download h5 files from s3
s3 = boto3.client("s3")
for velocity in readdy_compression_velocities:
    for repeat in range(num_repeats):
        file_name = f"actin_compression_velocity={velocity}_{repeat}.h5"
        h5_file_path = h5_folder_path / file_name
        if h5_file_path.is_file() and not redownload:
            print(f"{file_name} already exists")
            continue
        else:
            print(f"Downloading {file_name}")
            try:
                response = s3.download_file(
                    bucket_name,
                    f"outputs/{file_name}",
                    str(h5_file_path),
                )
            except Exception as e:
                print(e)

# %%
reprocess = True
df_list = []
for v_index, velocity in enumerate(readdy_compression_velocities):
    for repeat in range(num_repeats):
        file_name = f"actin_compression_velocity={velocity}_{repeat}.h5"
        df_save_path = (
            readdy_df_path
            / f"readdy_actin_compression_velocity_{velocity}_repeat_{repeat}.csv"
        )

        if df_save_path.is_file() and not reprocess:
            print(f"{file_name} already processed")
            df_points = pd.read_csv(df_save_path)
            df_list.append(df_points)
            continue

        if v_index == 1 and repeat == 2:
            print(f"Skipping {file_name} due to processing error")
            continue

        h5_file_path = h5_folder_path / file_name

        if not h5_file_path.is_file():
            print(f"{file_name} not found")
            continue

        print(f"Processing {file_name}")
        readdy_loader = ReaddyLoader(str(h5_file_path))
        readdy_post_processor = ReaddyPostProcessor(
            readdy_loader.trajectory(),
            box_size=600.0 * np.ones(3),
        )
        fiber_chain_ids = readdy_post_processor.linear_fiber_chain_ids(
            start_particle_phrases=["pointed"],
            other_particle_types=[
                "actin#",
                "actin#ATP_",
                "actin#mid_",
                "actin#mid_ATP_",
                "actin#fixed_",
                "actin#fixed_ATP_",
                "actin#mid_fixed_",
                "actin#mid_fixed_ATP_",
                "actin#barbed_",
                "actin#barbed_ATP_",
                "actin#fixed_barbed_",
                "actin#fixed_barbed_ATP_",
            ],
            polymer_number_range=5,
        )
        axis_positions, _ = readdy_post_processor.linear_fiber_axis_positions(
            fiber_chain_ids=fiber_chain_ids,
            ideal_positions=np.array(
                [
                    [24.738, 20.881, 26.671],
                    [27.609, 24.061, 27.598],
                    [30.382, 21.190, 25.725],
                ]
            ),
            ideal_vector_to_axis=np.array(
                [-0.01056751, -1.47785105, -0.65833209],
            ),
        )
        fiber_points = readdy_post_processor.linear_fiber_control_points(
            axis_positions=axis_positions,
            segment_length=10.0,
        )
        fiber_points_array = np.array(fiber_points)
        print(fiber_points_array.shape)
        df_points = array_to_dataframe(fiber_points_array)
        df_points.reset_index(inplace=True)
        df_points.rename(columns={0: "xpos", 1: "ypos", 2: "zpos"}, inplace=True)
        df_points["velocity"] = velocity
        df_points["repeat"] = repeat
        df_points["simulator"] = "readdy"
        df_points["normalized_time"] = (df_points["time"] - df_points["time"].min()) / (
            df_points["time"].max() - df_points["time"].min()
        )
        df_points.to_csv(
            df_save_path,
            index=False,
        )
        df_list.append(df_points)
# %%
df_readdy = pd.concat(df_list)
df_readdy.to_csv(
    readdy_df_path / "readdy_actin_compression_all_velocities_and_repeats.csv"
)
df_readdy.to_parquet(
    readdy_df_path / "readdy_actin_compression_all_velocities_and_repeats.parquet"
)

# %%
