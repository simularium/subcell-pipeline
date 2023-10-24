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
    COMPRESSIONMETRIC.CONTOUR_LENGTH,
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
                xvals = np.linspace(0, 1, df_repeat["time"].nunique())
                yvals = df_repeat.groupby("time")[metric.value].mean()
                if simulator == "cytosim" and metric.value == "CONTOUR_LENGTH":
                    yvals = yvals * 1000
                axs[ct].plot(
                    xvals,
                    # df_repeat["time"] * time_scale,
                    yvals,
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
