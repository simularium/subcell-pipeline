# %%
import pandas as pd
from pathlib import Path

from subcell_analysis.cytosim.post_process_cytosim import create_dataframes_for_repeats

# %% [markdown]
# ## setup output folders
current_folder = Path(__file__).parent
base_df_path = current_folder.parents[1] / "data/dataframes"
cytosim_df_path = base_df_path / "cytosim"
readdy_df_path = base_df_path / "readdy"
print(base_df_path)
# %% [markdown]
# # Process individual cytosim outputs to csv files
# %% [markdown]
# ## setup parameters for accessing data from s3
bucket_name = "cytosim-working-bucket"
num_repeats = 5
num_velocities = 7
configs = [f"vary_compress_rate000{num}" for num in range(3, num_velocities)]
# %% [markdown]
# ## convert individual outputs to dataframes
create_dataframes_for_repeats(
    bucket_name=bucket_name,
    num_repeats=num_repeats,
    configs=configs,
    save_folder=cytosim_df_path,
    file_name="cytosim_actin_compression",
    overwrite=False,
)
# %% [markdown]
# # Process individual readdy outputs to csv files
# %%
## TODO: add code to process readdy outputs to csv files
# %% [markdown]
# # Start workflow to combine outputs
# %% [markdown] 
# ## set velocities
readdy_compression_velocities = [4.7, 15, 47, 150]
cytosim_compression_velocities = [0.15, 0.47434165, 1.5, 4.7, 15, 47, 150]

# %% [markdown]
# ## Combine all cytosim outputs
num_repeats = 5
velocity_inds = range(3, 7)
df_list = []
for index in velocity_inds:
    for repeat in range(num_repeats):
        print(
            f"Calculating velocity {cytosim_compression_velocities[index]} and repeat {repeat}"
        )
        df_tmp = pd.read_csv(
            cytosim_df_path
            / f"cytosim_actin_compression_velocity_vary_compress_rate000{index}_repeat_{repeat}.csv"
        )

        df_tmp["velocity"] = cytosim_compression_velocities[index]
        df_tmp["repeat"] = repeat
        df_list.append(df_tmp)
df_cytosim = pd.concat(df_list)
df_cytosim.to_csv(
    cytosim_df_path / "cytosim_actin_compression_all_velocities_and_repeats.csv"
)

# %% [markdown]
# ## Combine all readdy outputs
num_repeats = 3
df_list = []
for velocity in readdy_compression_velocities:
    for repeat in range(num_repeats):
        file_path = (
            readdy_df_path
            / f"readdy_actin_compression_velocity_{velocity}_repeat_{repeat}.csv"
        )
        if file_path.is_file():
            df_tmp = pd.read_csv(file_path)
        else:
            continue
        print(f"Calculating velocity {velocity} and repeat {repeat}")
        df_tmp["velocity"] = velocity
        df_tmp["repeat"] = repeat
        df_list.append(df_tmp)

df_readdy = pd.concat(df_list)
df_readdy.to_csv(
    readdy_df_path / "readdy_actin_compression_all_velocities_and_repeats.csv"
)

# %% [markdown]
# ## Combine readdy output with cytosim
df_cytosim["simulator"] = "cytosim"
df_readdy["simulator"] = "readdy"
df_combined = pd.concat([df_cytosim, df_readdy])

# %% [markdown]
# ### save combined df
df_combined.to_csv(base_df_path / "combined_actin_compression_dataset.csv")
df_combined.head()

# %% [markdown]
# # Create combined subsampled dataset
# TODO: add code to create subsampled dataset
