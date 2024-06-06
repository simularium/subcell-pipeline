# %%
from pathlib import Path

import pandas as pd
from subcell_analysis.cytosim.post_process_cytosim import create_dataframes_for_repeats

# %% [markdown]
# ### setup output folders
current_folder = Path(__file__).parent
base_df_path = current_folder.parents[1] / "data/dataframes"
cytosim_df_path = base_df_path / "cytosim"
print(cytosim_df_path)
# %% [markdown]
# # Process individual cytosim outputs to csv files
# %% [markdown]
# ## setup parameters for accessing data from s3
bucket_name = "cytosim-working-bucket"
num_repeats = 5
velocity_inds = range(3, 7)
all_cytosim_compression_velocities = [0.15, 0.47434165, 1.5, 4.7, 15, 47, 150]
cytosim_compression_velocities = [
    all_cytosim_compression_velocities[num] for num in velocity_inds
]
configs = [f"vary_compress_rate000{num}" for num in velocity_inds]
# %% [markdown]
# ## convert individual outputs to dataframes
create_dataframes_for_repeats(
    bucket_name=bucket_name,
    num_repeats=num_repeats,
    configs=configs,
    save_folder=cytosim_df_path,
    file_name="cytosim_actin_compression",
    velocities=cytosim_compression_velocities,
    overwrite=True,
)
# %% [markdown]
# # Workflow to combine repeats and velocities
# %% [markdown]
# %% [markdown]
# ## Combine all cytosim outputs
num_repeats = 5
df_list = []
simulator = "cytosim"
for velocity in cytosim_compression_velocities:
    for repeat in range(num_repeats):
        file_path = (
            cytosim_df_path
            / f"cytosim_actin_compression_velocity_{velocity}_repeat_{repeat}.csv"
        )
        if file_path.is_file():
            print(f"Reading velocity {velocity} and repeat {repeat}")
            df_tmp = pd.read_csv(file_path)
        else:
            print(f"Missing csv. Skipped velocity {velocity} and repeat {repeat}")
            continue
        df_tmp["velocity"] = velocity
        df_tmp["repeat"] = repeat
        df_tmp["simulator"] = simulator
        time_vals = df_tmp["time"]
        df_tmp["normalized_time"] = (time_vals - time_vals.min()) / (
            time_vals.max() - time_vals.min()
        )
        df_list.append(df_tmp)
# %%
df_cytosim = pd.concat(df_list)
df_cytosim.to_csv(
    cytosim_df_path / "cytosim_actin_compression_all_velocities_and_repeats.csv"
)
df_cytosim.to_parquet(
    cytosim_df_path / "cytosim_actin_compression_all_velocities_and_repeats.parquet"
)

# %%
