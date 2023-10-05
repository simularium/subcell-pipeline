# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from subcell_analysis.compression_analysis import COMPRESSIONMETRIC
from subcell_analysis.compression_workflow_runner import compression_metrics_workflow

from pathlib import Path

# %% set matplotlib defaults
plt.rcdefaults()

# %% Set parameters
readdy_compression_velocities = [4.7, 15, 47, 150]
# cytosim_compression_velocities = [0.15, 0.47434165, 1.5, 4.73413649, 15, 47.4341649, 150]
cytosim_compression_velocities = [0.15, 0.47434165, 1.5, 4.7, 15, 47, 150]

# %%
metrics = [
    COMPRESSIONMETRIC.NON_COPLANARITY,
    COMPRESSIONMETRIC.PEAK_ASYMMETRY,
    COMPRESSIONMETRIC.TOTAL_FIBER_TWIST,
]

# %% Process cytosim data
df_path = Path("../../data/dataframes")
# %% metric options
options = {
    "signed": False,
}
# %%
num_repeats = 5
velocity_inds = range(3, 7)
df_metrics = []
for index in velocity_inds:
    for repeat in range(num_repeats):
        print(
            f"""Calculating metrics for velocity 
            {cytosim_compression_velocities[index]} and repeat {repeat}"""
        )
        df = pd.read_csv(
            f"""{df_path}/cytosim_actin_compression_velocity_vary_
            compress_rate000{index}_repeat_{repeat}.csv"""
        )
        df = compression_metrics_workflow(df, metrics, **options)
        metric_df = (
            df.groupby("time")[[metric.value for metric in metrics]]
            .mean()
            .reset_index()
        )
        metric_df["velocity"] = cytosim_compression_velocities[index]
        metric_df["repeat"] = repeat
        df_metrics.append(metric_df)
        # break
    # break
df_cytosim = pd.concat(df_metrics)
df_cytosim.to_csv(
    f"{df_path}/cytosim_actin_compression_metrics_all_velocities_and_repeats.csv"
)

# %% Load from saved data
# df_cytosim = pd.read_csv(f"{df_path}/cytosim_actin_compression_
# metrics_all_velocities_and_repeats.csv")

# %% Process readdy data
num_repeats = 3
df_metrics = []
for velocity in readdy_compression_velocities:
    for repeat in range(num_repeats):
        file_path = (
            df_path
            / f"readdy_actin_compression_velocity_{velocity}_repeat_{repeat}.csv"
        )
        if file_path.is_file():
            df = pd.read_csv(file_path)
        else:
            continue
        print(f"Calculating metrics for velocity {velocity} and repeat {repeat}")
        df = compression_metrics_workflow(df, metrics, **options)
        metric_df = (
            df.groupby("time")[[metric.value for metric in metrics]]
            .mean()
            .reset_index()
        )
        metric_df["velocity"] = velocity
        metric_df["repeat"] = repeat
        df_metrics.append(metric_df)

df_readdy = pd.concat(df_metrics)
df_readdy.to_csv(
    f"{df_path}/readdy_actin_compression_metrics_all_velocities_and_repeats.csv"
)

# %% Load from saved data
# df_readdy = pd.read_csv(f"{df_path}/readdy_actin
# _compression_metrics_all_velocities_and_repeats.csv")

# %% Plot metrics for readdy and cytosim
figure_path = Path("../../figures")
figure_path.mkdir(exist_ok=True)
# %%
df_cytosim["simulator"] = "cytosim"
df_readdy["simulator"] = "readdy"

df_combined = pd.concat([df_cytosim, df_readdy])

# %%
color_map = {"cytosim": "C0", "readdy": "C1"}

# %% plot metrics vs time
num_velocities = df_combined["velocity"].nunique()
for metric in metrics:
    fig, axs = plt.subplots(
        1, num_velocities, figsize=(num_velocities * 5, 5), sharey=True, dpi=300
    )
    for ct, (velocity, df_velocity) in enumerate(df_combined.groupby("velocity")):
        for simulator, df_simulator in df_velocity.groupby("simulator"):
            for repeat, df_repeat in df_simulator.groupby("repeat"):
                if repeat == 0:
                    label = f"{simulator}"
                else:
                    label = "_nolegend_"
                axs[ct].plot(
                    np.linspace(0, 1, len(df_repeat)),
                    # df_repeat["time"] * time_scale,
                    df_repeat[metric.value],
                    label=label,
                    color=color_map[simulator],
                    alpha=0.7,
                )
        axs[ct].legend()
        axs[ct].set_title(f"Velocity: {velocity}")
        if ct == 0:
            axs[ct].set_ylabel(metric.value)
    fig.supxlabel("Normalized time")
    fig.suptitle(f"{metric.value}")
    plt.tight_layout()
    fig.savefig(figure_path / f"all_simulators_{metric.value}_vs_time.png")
