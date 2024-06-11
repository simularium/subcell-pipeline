# %% [markdown]
"""
# Compare metrics across simulators

Notebook contains steps to compare metrics of fiber compression across different
simulators. Currently supports comparison of Cytosim and ReaDDy simulations.

- [Load dataframes](#load-dataframes)
- [Calculate metrics and add to dataframe](#calculate-metrics)
- [Plot metrics vs time](#plot-metrics-vs-time)
"""  # noqa: D400, D415
# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from subcell_pipeline.analysis.compression_metrics.compression_analysis import (
    COMPRESSIONMETRIC,
    compression_metrics_workflow,
)

plt.rcdefaults()

datadir = Path(__file__).parents[3] / "data/dataframes"
print(datadir)

# %% [markdown]
"""
## Select metrics to analyze

Available metrics are defined in the `COMPRESSIONMETRIC` enum.
"""
metrics = [
    COMPRESSIONMETRIC.NON_COPLANARITY,
    COMPRESSIONMETRIC.PEAK_ASYMMETRY,
    COMPRESSIONMETRIC.AVERAGE_PERP_DISTANCE,
    COMPRESSIONMETRIC.CALC_BENDING_ENERGY,
    COMPRESSIONMETRIC.CONTOUR_LENGTH,
    COMPRESSIONMETRIC.COMPRESSION_RATIO,
]
options = {
    "signed": True,
}

subsampled = False
suffix = "_subsampled" if subsampled else ""

# %% [markdown]
"""
## Load combined dataframe

"""
df_path = (
    datadir
    / f"combined_actin_compression_dataset_all_velocities_and_repeats{suffix}.parquet"
)
df = pd.read_parquet(df_path)

# %% [markdown]
"""
## Calculate metrics and add to dataframe

"""
for simulator, df_sim in df.groupby("simulator"):
    for velocity, df_velocity in df_sim.groupby("velocity"):
        for repeat, df_repeat in df_velocity.groupby("repeat"):
            print(f"simulator: {simulator}, velocity: {velocity}, repeat: {repeat}")
            df_repeat = compression_metrics_workflow(
                df_repeat, metrics_to_calculate=metrics, **options
            )
            for metric in metrics:
                df.loc[df_repeat.index, metric.value] = df_repeat[metric.value]
# %% [markdown]
"""
## Save dataframe with metrics

"""
df.to_csv(
    f"{datadir}/combined_actin_compression_metrics_all_velocities_and_repeats{suffix}.csv"
)
df.to_parquet(
    f"{datadir}/combined_actin_compression_metrics_all_velocities_and_repeats{suffix}.parquet"
)

# %% Plot metrics for readdy and cytosim
figure_path = Path(__file__).parents[3] / "figures"
print(figure_path)
figure_path.mkdir(exist_ok=True)


# %%
color_map = {"cytosim": "C0", "readdy": "C1"}

# %% plot metrics vs time
num_velocities = df["velocity"].nunique()
compression_distance = 150  # um
CYTOSIM_SCALE = 1000
for metric in metrics:
    fig, axs = plt.subplots(
        1, num_velocities, figsize=(num_velocities * 5, 5), sharey=True, dpi=300
    )
    axs = axs.ravel()
    for ct, (velocity, df_velocity) in enumerate(df.groupby("velocity")):
        for simulator, df_simulator in df_velocity.groupby("simulator"):
            for repeat, df_repeat in df_simulator.groupby("repeat"):
                if repeat == 0:
                    label = f"{simulator}"
                else:
                    label = "_nolegend_"
                total_time = compression_distance / velocity  # s
                xvals = np.linspace(0, 1, df_repeat["time"].nunique()) * total_time
                yvals = df_repeat.groupby("time")[metric.value].mean()
                if simulator == "cytosim" and metric.value in [
                    "CONTOUR_LENGTH",
                    "AVERAGE_PERP_DISTANCE",
                ]:
                    yvals = yvals * CYTOSIM_SCALE
                axs[ct].plot(
                    xvals,
                    # df_repeat["time"] * time_scale,
                    yvals,
                    label=label,
                    color=color_map[simulator],
                    alpha=0.6,
                )
        axs[ct].set_title(f"Velocity: {velocity}")
        if ct == 0:
            axs[ct].legend()
    fig.supxlabel("Normalized Time")
    fig.supylabel(f"{metric.label()}")
    plt.tight_layout()
    fig.savefig(figure_path / f"all_simulators_{metric.value}_vs_time_subsampled.png")
# %% plot metrics vs time with mean +/- SD
compression_distance = 150
num_velocities = df["velocity"].nunique()
for metric in metrics:
    fig, axs = plt.subplots(
        2, num_velocities // 2, figsize=(6, 6), sharey=True, dpi=300
    )
    axs = axs.ravel()
    for ct, (velocity, df_velocity) in enumerate(df.groupby("velocity")):
        for simulator, df_simulator in df_velocity.groupby("simulator"):
            repeat_yvals = []
            for repeat, df_repeat in df_simulator.groupby("repeat"):
                if repeat == 0:
                    label = f"{simulator}"
                else:
                    label = "_nolegend_"
                max_time = compression_distance / df_repeat["velocity"].mean()
                xvals = np.linspace(0, 1, df_repeat["time"].nunique()) * max_time
                yvals = df_repeat.groupby("time")[metric.value].mean()
                if simulator == "cytosim" and metric.value == "CONTOUR_LENGTH":
                    yvals = yvals * 1000
                repeat_yvals.append(yvals)
                axs[ct].plot(
                    xvals,
                    # df_repeat["time"] * time_scale,
                    yvals,
                    # label=label,
                    color=color_map[simulator],
                    alpha=0.1,
                )
            mean_yvals = np.mean(repeat_yvals, axis=0)
            std_yvals = np.std(repeat_yvals, axis=0)
            axs[ct].plot(
                xvals,
                mean_yvals,
                label=f"{simulator}",
                color=color_map[simulator],
                # linestyle="--",
            )
            # axs[ct].fill_between(
            #     xvals,
            #     mean_yvals - std_yvals,
            #     mean_yvals + std_yvals,
            #     color=color_map[simulator],
            #     alpha=0.2,
            #     edgecolor="none",
            # )
        axs[ct].set_title(f"Velocity: {velocity}")
        if ct == 0:
            axs[ct].legend()
    fig.supxlabel("Time (s)")
    fig.supylabel(metric_label_map[metric])
    plt.tight_layout()
    fig.savefig(
        figure_path / f"all_simulators_{metric.value}_vs_time_subsampled_averaged.png"
    )
    fig.savefig(
        figure_path / f"all_simulators_{metric.value}_vs_time_subsampled_averaged.svg"
    )
