# %% imports
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# %% load dataframes
current_folder = Path(__file__).parent
base_data_folder = current_folder.parent / "data"
df_folder = base_data_folder / "dataframes"
df_folder.mkdir(parents=True, exist_ok=True)
print(df_folder)
# %%
cytosim_df = pd.read_parquet(
    df_folder / "cytosim/cytosim_actin_compression_all_velocities_and_repeats.parquet"
)
readdy_df = pd.read_parquet(
    df_folder / "readdy/readdy_actin_compression_all_velocities_and_repeats.parquet"
)
print(cytosim_df.shape)
print(readdy_df.shape)

# %%
n_timepoints = 200
n_monomer_points = 200
# %%
cols_to_interp = ["xpos", "ypos", "zpos"]
# %%
# Here we will loop over df for readdy and cytosim, so that you can run the script from start to end
# and it will subsample both the cytosim and readdy data

dfs = [cytosim_df, readdy_df]
df_subsample_list = [[], []]
df_subsampled = [[], []]
# %%
for index, (simulator, df) in enumerate(
    zip(
        ["cytosim", "readdy"],
        dfs,
    )
):
    for velocity, df_velocity in df.groupby("velocity"):
        for repeat, df_repeat in df_velocity.groupby("repeat"):
            num_time_vals = df_repeat["time"].nunique()
            df_time_vals = df_repeat["time"].unique()
            time_inds = np.rint(
                np.interp(
                    np.linspace(0, 1, n_timepoints),
                    np.linspace(0, 1, num_time_vals),
                    np.arange(num_time_vals),
                )
            )
            use_time_vals = df_time_vals[time_inds.astype(int)]
            for time, df_time in df_repeat.groupby("time"):
                if time not in use_time_vals:
                    continue
                normalized_time = df_time["normalized_time"].iloc[0]
                print(f"velocity: {velocity}, repeat: {repeat}, time: {time}")
                df_tmp = pd.DataFrame()
                df_tmp["monomer_ids"] = np.arange(n_monomer_points)
                df_tmp["time"] = time
                df_tmp["normalized_time"] = normalized_time
                df_tmp["velocity"] = velocity
                df_tmp["repeat"] = repeat
                df_tmp["simulator"] = simulator
                for col in cols_to_interp:
                    df_tmp[col] = np.interp(
                        np.linspace(0, 1, n_monomer_points),
                        np.linspace(0, 1, df_time.shape[0]),
                        df_time[col].values,
                    )
                df_subsample_list[index].append(df_tmp)

    df_subsampled[index] = pd.concat(df_subsample_list[index])
# %%
df_subsampled[0].to_csv(
    df_folder / "cytosim/cytosim_actin_compression_subsampled.csv", index=False
)
df_subsampled[1].to_csv(
    df_folder / "readdy/readdy_actin_compression_subsampled.csv", index=False
)
df_subsampled[0].to_parquet(
    df_folder / "cytosim/cytosim_actin_compression_subsampled.parquet", index=False
)
df_subsampled[1].to_parquet(
    df_folder / "readdy/readdy_actin_compression_subsampled.parquet", index=False
)
# %% get df for given velocity and repeat
cytosim_subsampled = df_subsampled[0]
velocity_vals = cytosim_subsampled["velocity"].unique()
repeat_vals = cytosim_subsampled["repeat"].unique()
print(f"velocity_vals: {velocity_vals}, repeat_vals: {repeat_vals}")
# %%
velocity_ind = 0
repeat_ind = 0
df_test = cytosim_subsampled[
    (cytosim_subsampled["velocity"] == velocity_vals[velocity_ind])
    & (cytosim_subsampled["repeat"] == repeat_vals[repeat_ind])
]

# %%
fig, axs = plt.subplots(2, 1, sharex=True)
axs[0].plot(df_tmp["xpos"], df_tmp["ypos"], "ro-", ms=3, label="subsampled")
axs[1].plot(df_time["xpos"], df_time["ypos"], "ko-", ms=3, label="original")
# axs[1].set_xlim([-0.1,-0.05 ])
# ax.legend()
plt.show()
# %%

