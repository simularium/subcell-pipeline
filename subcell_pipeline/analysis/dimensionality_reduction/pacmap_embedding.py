# %%
import numpy as np
import pandas as pd
from pacmap import PaCMAP
from scipy import interpolate as spinterp
from subcell_analysis.compression_analysis import get_pacmap_embedding

# %%
num_repeats = 5
df_list = []
configs = ["vary_compress_rate0006"]
directory = "../data/dataframes"
for config in configs:
    for repeat in range(num_repeats):
        print(config, repeat)
        # cytosim_actin_compression_velocity_vary_compress_rate0006_repeat_0.csv
        df = pd.read_csv(
            f"{directory}/cytosim_actin_compression_velocity_{config}_repeat_{repeat}.csv"
        )
        df["repeat"] = repeat
        df["config"] = config
        df_list.append(df)
df_all = pd.concat(df_list)
df_all.to_csv("dataframes/all_fibers_configs_3_4.csv")

# %%
num_monomers = 100
num_timepoints = 101
all_config_repeats = []
cols_to_interp = ["xpos", "ypos", "zpos"]
for config, df_config in df_all.groupby("config"):
    for repeat, df_repeat in df_config.groupby("repeat"):
        all_times = []
        for time, df_time in df_repeat.groupby("time"):
            # interpolate xpos, ypos, zpos to num_monomers
            X = df_time[cols_to_interp].values
            t = np.linspace(0, 1, X.shape[0])
            F = spinterp.interp1d(t, X.T, bounds_error=False, fill_value="extrapolate")
            u = np.linspace(0, 1, num_monomers)
            all_times.append(F(u).T)
        all_times = np.array(all_times)
        interp_timepoints = np.around(
            len(all_times) / num_timepoints * np.arange(num_timepoints)
        ).astype(int)
        all_config_repeats.append(np.array(all_times)[interp_timepoints, :, :])
all_config_repeats = np.array(all_config_repeats)

# %%
embedding = PaCMAP(n_components=2, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0)
reshaped_metrics = all_config_repeats.reshape(all_config_repeats.shape[0], -1)
embed_pos = embedding.fit_transform(reshaped_metrics)

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
configs = [3, 4]
for ct, config in enumerate(configs):
    inds = ct * num_repeats + np.arange(num_repeats)
    ax.scatter(embed_pos[inds, 0], embed_pos[inds, 1], label=f"config {config}")
ax.set_xlabel("embedding 1")
ax.set_ylabel("embedding 2")
ax.set_title("PaCMAP embedding of all repeats")
ax.legend()
plt.show()

# %%
