# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from subcell_analysis.compression_analysis import COMPRESSIONMETRIC
from subcell_analysis.compression_workflow_runner import compression_metrics_workflow

# %% set matplotlib defaults
plt.rcdefaults()

# %%
metrics = [
    COMPRESSIONMETRIC.NON_COPLANARITY,
    COMPRESSIONMETRIC.PEAK_ASYMMETRY,
    COMPRESSIONMETRIC.TOTAL_FIBER_TWIST,
    COMPRESSIONMETRIC.CALC_BENDING_ENERGY,
]

# %% Process cytosim data
df_path = Path("../data/dataframes")
# %% metric options
options = {
    "signed": True,
}
# %% load dataframe
df = pd.read_csv(df_path / "combined_actin_compression_dataset_subsampled.csv")

# %% add metrics
for simulator, df_sim in df.groupby("simulator"):
    for velocity, df_velocity in df_sim.groupby("velocity"):
        for repeat, df_repeat in df_velocity.groupby("repeat"):
            print(f"simulator: {simulator}, velocity: {velocity}, repeat: {repeat}")
            df_repeat = compression_metrics_workflow(
                df_repeat, metrics_to_calculate=metrics, **options
            )
            for metric in metrics:
                df.loc[df_repeat.index, metric.value] = df_repeat[metric.value]
# %% save dataframe
df.to_csv(
    f"{df_path}/combined_actin_compression_metrics_all_velocities_and_repeats_subsampled.csv"
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
        df = compression_metrics_workflow(df, metrics, **options)  # type: ignore
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
df_cytosim = pd.read_csv(f"{df_path}/cytosim_actin_compression_subsampled.csv")
df_readdy = pd.read_csv(f"{df_path}/readdy_actin_compression_subsampled.csv")

# %% Plot metrics for readdy and cytosim
figure_path = Path("../../figures")
figure_path.mkdir(exist_ok=True)

# %%
color_map = {"cytosim": "C0", "readdy": "C1"}

# %% plot metrics vs time
num_velocities = df["velocity"].nunique()
for metric in metrics:
    fig, axs = plt.subplots(
        1, num_velocities, figsize=(num_velocities * 5, 5), sharey=True, dpi=300
    )
    for ct, (velocity, df_velocity) in enumerate(df.groupby("velocity")):
        for simulator, df_simulator in df_velocity.groupby("simulator"):
            for repeat, df_repeat in df_simulator.groupby("repeat"):
                if repeat == 0:
                    label = f"{simulator}"
                else:
                    label = "_nolegend_"
                axs[ct].plot(
                    np.linspace(0, 1, df_repeat["time"].nunique()),
                    # df_repeat["time"] * time_scale,
                    df_repeat.groupby("time")[metric.value].mean(),
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
    fig.savefig(figure_path / f"all_simulators_{metric.value}_vs_time_subsampled.png")

# %%

# %%
