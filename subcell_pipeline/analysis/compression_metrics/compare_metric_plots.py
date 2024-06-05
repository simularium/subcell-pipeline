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
    COMPRESSIONMETRIC.AVERAGE_PERP_DISTANCE,
    COMPRESSIONMETRIC.TOTAL_FIBER_TWIST,
    COMPRESSIONMETRIC.CALC_BENDING_ENERGY,
    COMPRESSIONMETRIC.CONTOUR_LENGTH,
    COMPRESSIONMETRIC.COMPRESSION_RATIO,
]
# %%
metric_label_map = {
    COMPRESSIONMETRIC.NON_COPLANARITY: "Non-coplanarity",
    COMPRESSIONMETRIC.PEAK_ASYMMETRY: "Peak asymmetry",
    COMPRESSIONMETRIC.TOTAL_FIBER_TWIST: "Total fiber twist",
    COMPRESSIONMETRIC.CALC_BENDING_ENERGY: "Calculated bending energy",
    COMPRESSIONMETRIC.CONTOUR_LENGTH: "Contour length",
    COMPRESSIONMETRIC.COMPRESSION_RATIO: "Compression ratio",
}

# %% Process cytosim data
df_path = Path("../data/dataframes")
# %% metric options
options = {
    "signed": True,
}
# %% load dataframe
# df = pd.read_csv(df_path / "combined_actin_compression_dataset_all_velocities_and_repeats.csv")

# # %% add metrics
# for simulator, df_sim in df.groupby("simulator"):
#     for velocity, df_velocity in df_sim.groupby("velocity"):
#         for repeat, df_repeat in df_velocity.groupby("repeat"):
#             print(f"simulator: {simulator}, velocity: {velocity}, repeat: {repeat}")
#             df_repeat = compression_metrics_workflow(
#                 df_repeat, metrics_to_calculate=metrics, **options
#             )
#             for metric in metrics:
#                 df.loc[df_repeat.index, metric.value] = df_repeat[metric.value]
# %% save dataframe
df = pd.read_csv(
    f"{df_path}/combined_actin_compression_metrics_all_velocities_and_repeats_subsampled_with_metrics.csv"
)

df

# %% Plot metrics for readdy and cytosim
figure_path = Path("../../figures")
figure_path.mkdir(exist_ok=True)

# %% Load in tomogram data
df_exp = pd.read_csv(df_path / "tomogram_subsampled_with_metrics.csv")
df_exp = df_exp[df_exp["CONTOUR_LENGTH"] > 150]
df_exp
# %%
color_map = {"cytosim": "C0", "readdy": "C1"}

# %% plot metrics vs time
num_velocities = df["velocity"].nunique()
compression_distance = 0.3  # um
for metric in metrics:
    fig, axs = plt.subplots(
        2, num_velocities // 2, figsize=(5, 5), sharey=True, dpi=300
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
                    yvals = yvals * 1000
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
    fig.supxlabel("Time (s)")
    fig.supylabel(f"{metric.value}")
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
# # %%
# import numpy as np
# import matplotlib.pyplot as plt

# # Assuming 'metrics' is defined earlier in your code
# # Assuming 'color_map' is a dictionary mapping simulators to colors, defined earlier in your code
# # Assuming 'figure_path' is defined earlier in your code as the directory to save figures

# num_velocities = df["velocity"].nunique()

# # Calculate the mean and standard deviation for the experimental data

# chosen_metrics = [COMPRESSIONMETRIC.AVERAGE_PERP_DISTANCE]

# for metric in chosen_metrics:
#     exp_mean = np.mean(df_exp.groupby("repeat")["AVERAGE_PERP_DISTANCE"].mean())
#     exp_std = np.std(df_exp.groupby("repeat")["AVERAGE_PERP_DISTANCE"].mean())
#     fig, axs = plt.subplots(
#         1, num_velocities, figsize=(num_velocities * 5, 5), sharey=True, dpi=300
#     )
#     for ct, (velocity, df_velocity) in enumerate(df.groupby("velocity")):
#         for simulator, df_simulator in df_velocity.groupby("simulator"):
#             for repeat, df_repeat in df_simulator.groupby("repeat"):
#                 if repeat == 0:
#                     label = f"{simulator}"
#                 else:
#                     label = "_nolegend_"
#                 xvals = np.linspace(0, 1, df_repeat["time"].nunique())
#                 yvals = df_repeat.groupby("time")[metric.value].mean()
#                 if simulator == "cytosim" and metric.value == "CONTOUR_LENGTH":
#                     yvals = yvals * 1000
#                 axs[ct].plot(
#                     xvals,
#                     yvals,
#                     label=label,
#                     color=color_map[simulator],
#                     alpha=0.7,
#                 )

#         # Overlay the mean and standard deviation for the experimental data
#         # This will add a horizontal line for the mean and shaded area for the SD
#         axs[ct].axhline(exp_mean, color='red', linestyle='-', linewidth=2, label='Exp. Mean')
#         axs[ct].fill_between(xvals, exp_mean - exp_std, exp_mean + exp_std, color='red', alpha=0.2, label='Exp. SD')


#         # axs[ct].set_title(f"Velocity: {velocity}")
#         if ct == 0:
#             axs[ct].legend(loc='upper left')
#     fig.supxlabel("Normalized time")
#     fig.supylabel(f"{metric.value}")
#     # fig.suptitle(f"{metric.value}")
#     plt.tight_layout()
#     fig.savefig(figure_path / f"all_simulators_{metric.value}_vs_time_with_exp.png")

# # %%
# df_exp['PEAK_ASYMMETRY'].mean()
# # %%
# df_exp['NON_COPLANARITY'].mean()
# # %%
# # %%
# def plot_filaments(chosen_filaments):
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     for filament_id, filament in chosen_filaments.groupby('repeat'):
#         ax.plot(filament['xpos'], filament['ypos'], filament['zpos'], marker='o')
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')

#     return fig

# plot_filaments(df_exp[df_exp['COMPRESSION_RATIO']>0.2])
# # %%

# %%
