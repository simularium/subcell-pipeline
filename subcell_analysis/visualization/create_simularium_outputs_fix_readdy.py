import argparse
import math
import os
import sys
from typing import Dict, Tuple

import boto3
import numpy as np
import pandas as pd
from botocore.exceptions import ClientError
from pint import UnitRegistry
from scipy.spatial.transform import Rotation
from simulariumio import (
    DISPLAY_TYPE,
    AgentData,
    CameraData,
    DisplayData,
    FileConverter,
    InputFileData,
    MetaData,
    ScatterPlotData,
    TrajectoryConverter,
    TrajectoryData,
    UnitData,
)
from simulariumio.filters import EveryNthTimestepFilter

from subcell_analysis.compression_analysis import (
    COMPRESSIONMETRIC,
    get_asymmetry_of_peak,
    get_average_distance_from_end_to_end_axis,
    get_bending_energy_from_trace,
    get_contour_length_from_trace,
    get_third_component_variance,
)
from subcell_analysis.compression_workflow_runner import compression_metrics_workflow
from subcell_analysis.cytosim.post_process_cytosim import cytosim_to_simularium

CYTOSIM_CONDITIONS = {
    "0001": 0.48,
    "0002": 1.5,
    "0003": 4.7,
    "0004": 15,
    "0005": 47,
    "0006": 150,
}
READDY_CONDITIONS = [
    4.7,
    15,
    47,
    150,
]
NUM_REPEATS = 5
TOTAL_STEPS = 200
POINTS_PER_FIBER = 200
BENDING_ENERGY_SCALE_FACTOR = 1000.0
CYTOSIM_SCALE_FACTOR = 1000.0
BOX_SIZE = 600.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualizes ReaDDy and Cytosim actin simulations"
    )
    parser.add_argument("--combined", action=argparse.BooleanOptionalAction)
    parser.set_defaults(combined=False)
    parser.add_argument("--cytosim", action=argparse.BooleanOptionalAction)
    parser.set_defaults(cytosim=False)
    parser.add_argument("--upload", action=argparse.BooleanOptionalAction)
    parser.set_defaults(upload=False)
    return parser.parse_args()


s3_client = boto3.client("s3")
for repeat in range(num_repeats):
    s3_client.download_file(
        "cytosim-working-bucket",
        f"vary_compress_rate0003/outputs/{repeat}/fiber_segment_curvature.txt",
        f"../data/fiber_segment_curvature_{repeat}.txt",
    )

# %% [markdown]
# ### Process single repeat

# %%
repeat = 0
input_file_path = f"../data/fiber_segment_curvature_{repeat}.txt"

box_size = 3.0
scale_factor = 100
fiber_data = cytosim_to_simularium(
    input_file_path, box_size=box_size, scale_factor=scale_factor
)

# %% [markdown]
# Create cytosim converter object

# %%
cytosim_converter = CytosimConverter(fiber_data)

# %% [markdown]
# Read metric data

# # %%
# df_path = f"dataframes/actin_forces{config_id}_{repeat}_compression_metrics.csv"
# df = pd.read_csv(df_path)

# %% [markdown]
# Add metric plots

# %%
plot_metrics = [
    COMPRESSIONMETRIC.AVERAGE_PERP_DISTANCE,
    COMPRESSIONMETRIC.TOTAL_FIBER_TWIST,
    COMPRESSIONMETRIC.SUM_BENDING_ENERGY,
    COMPRESSIONMETRIC.PEAK_ASYMMETRY,
    COMPRESSIONMETRIC.NON_COPLANARITY,
]

# # %%
# for metric in plot_metrics:
#     metric_by_time = df.groupby(["time"])[metric.value].mean()
#     cytosim_converter.add_plot(
#         ScatterPlotData(
#             title=f"{metric} over time",
#             xaxis_title="Time",
#             yaxis_title=metric.value,
#             xtrace=np.arange(len(metric_by_time)) * 1e-5,
#             ytraces={
#                 f"repeat {repeat}": metric_by_time,
#             },
#         )
#     )

# %% [markdown]
# Save converted data

# %%
cytosim_converter.save(f"outputs/free_barbed_end_final{repeat}")

# %% [markdown]
# ### Process multiple repeats

# %%
box_size = 3.0
scale_factor = 100
colors = ["#F0F0F0", "#0000FF", "#FF0000", "#00FF00", "#FF00FF"]

# %% [markdown]
# Create initial trajectory data object

# # %%
# input_file_path = f"data/fiber_segment_curvature_0.txt"
# fiber_data = cytosim_to_simularium(
#     input_file_path,
#     box_size=box_size,
#     scale_factor=scale_factor,
#     color=colors[0],
#     actin_number=0,
# )
# cytosim_converter = CytosimConverter(fiber_data)

# trajectory_data = cytosim_converter._data

# %% [markdown]
# Append additional repeats to trajectory data object

# # %%
# for repeat in range(1, num_repeats):
#     input_file_path = f"data/fiber_segment_curvature_{repeat}.txt"
#     fiber_data = cytosim_to_simularium(
#         input_file_path,
#         box_size=box_size,
#         scale_factor=scale_factor,
#         color=colors[repeat],
#         actin_number=repeat,
#     )
#     cytosim_converter = CytosimConverter(fiber_data)
#     new_agent_data = cytosim_converter._data.agent_data

#     trajectory_data.append_agents(new_agent_data)

# # %%
# all_repeats_converter = TrajectoryConverter(trajectory_data)

# %% [markdown]
# ### Add plots for all repeats

# %%
plot_metrics = [
    COMPRESSIONMETRIC.AVERAGE_PERP_DISTANCE,
    COMPRESSIONMETRIC.TOTAL_FIBER_TWIST,
    COMPRESSIONMETRIC.SUM_BENDING_ENERGY,
    COMPRESSIONMETRIC.PEAK_ASYMMETRY,
    COMPRESSIONMETRIC.NON_COPLANARITY,
]

# %% [markdown]
# Get metrics for all repeats

# # %%
# df_list = []
# for repeat in range(num_repeats):
#     df_path = f"dataframes/actin_forces{config_id}_{repeat}_compression_metrics.csv"
#     df = pd.read_csv(df_path)
#     df["repeat"] = repeat
#     df_list.append(df)
# df_all = pd.concat(df_list)

# %% [markdown]
# Add plots to converter object

# # %%
# for metric in plot_metrics:
#     ytraces = {}
#     for repeat, df_repeat in df_all.groupby("repeat"):
#         ytraces[f"repeat {repeat}"] = df_repeat.groupby(["time"])[metric.value].mean()

#     all_repeats_converter.add_plot(
#         ScatterPlotData(
#             title=f"{metric.value} over time",
#             xaxis_title="Time",
#             yaxis_title=metric.value,
#             xtrace=np.arange(metric_by_time.shape[0]) * 1e-5,
#             ytraces=ytraces,
#             render_mode="lines",
#         )
#     )


def upload_cytosim_trajectories():
    for condition in CYTOSIM_CONDITIONS.keys():
        velocity = CYTOSIM_CONDITIONS[condition]
        for repeat in range(NUM_REPEATS):
            upload_file_to_s3(
                bucket_name="cytosim-working-bucket",
                src_path=f"data/cytosim_outputs/actin_compression_velocity={velocity}_{repeat}.simularium",
                s3_path=f"simularium/actin_compression_velocity={velocity}_{repeat}.simularium",
            )
    for repeat in range(NUM_REPEATS):
        upload_file_to_s3(
            bucket_name="cytosim-working-bucket",
            src_path=f"data/cytosim_outputs/actin_compression_baseline_{repeat}.simularium",
            s3_path=f"simularium/actin_compression_baseline_{repeat}.simularium",
        )


# %%
cytosim_converter.save(f"outputs/vary_compress_rate_0003_all_repeats")

# %%
