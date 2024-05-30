# %% [markdown]
# # Create combined dataframe
# 1. Run create_dataframes_from_cytosim_outputs.py
# to create dataframes for all repeats of given configs for cytosim.
# 2. Run create_dataframes_from_readdy_outputs.py
# to create dataframes for all repeats of given configs for readdy.
# 3. Run create_combined_dataframe.py
# to combine the dataframes from cytosim and readdy.

# %%
from pathlib import Path

import pandas as pd

# %% [markdown]
# ## setup output folders
current_folder = Path(__file__).parent
base_df_path = current_folder.parents[1] / "data/dataframes"
cytosim_df_path = base_df_path / "cytosim"
readdy_df_path = base_df_path / "readdy"
print(base_df_path)

# %%
cytosim_df = pd.read_parquet(
    cytosim_df_path / "cytosim_actin_compression_all_velocities_and_repeats.parquet"
)
readdy_df = pd.read_parquet(
    readdy_df_path / "readdy_actin_compression_all_velocities_and_repeats.parquet"
)
# %% merge dataframes
combined_df = pd.concat([cytosim_df, readdy_df], ignore_index=True)
unnamed_cols = [col for col in combined_df.columns if "Unnamed" in col]
combined_df = combined_df.drop(columns=unnamed_cols)
combined_df = combined_df.drop(columns=["id"])
# %% save as csv and parquet
combined_df.to_csv(
    base_df_path / "combined_actin_compression_dataset_all_velocities_and_repeats.csv"
)
combined_df.to_parquet(
    base_df_path
    / "combined_actin_compression_dataset_all_velocities_and_repeats.parquet"
)
# %% load subsampled dataframes
subsampled_cytosim_df = pd.read_parquet(
    cytosim_df_path / "cytosim_actin_compression_subsampled.parquet"
)
subsampled_readdy_df = pd.read_parquet(
    readdy_df_path / "readdy_actin_compression_subsampled.parquet"
)
# %% combine subsampled dataframes
subsampled_combined_df = pd.concat(
    [subsampled_cytosim_df, subsampled_readdy_df], ignore_index=True
)
unnamed_cols = [col for col in subsampled_combined_df.columns if "Unnamed" in col]
subsampled_combined_df = subsampled_combined_df.drop(columns=unnamed_cols)
# %% save as csv and parquet
subsampled_combined_df.to_csv(
    base_df_path
    / "combined_actin_compression_dataset_all_velocities_and_repeats_subsampled.csv"
)
subsampled_combined_df.to_parquet(
    base_df_path
    / "combined_actin_compression_dataset_all_velocities_and_repeats_subsampled.parquet"
)
