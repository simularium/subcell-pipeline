# %% [markdown]
# ### 1. Cloud Simulation Analysis

import os
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
from simulariumio import ScatterPlotData
from simulariumio.cytosim import CytosimConverter

from subcell_analysis.compression_analysis import COMPRESSIONMETRIC
from subcell_analysis.compression_workflow_runner import (
    compression_metrics_workflow,
    plot_metric,
    plot_metric_list,
)

# %%
from subcell_analysis.cytosim.post_process_cytosim import create_dataframes_for_repeats

# %%
bucket_name = "cytosim-working-bucket"
num_repeats = 5
configs = ["vary_compress_rate0006"]


# %%
directory = "../data/dataframes"
if not os.path.exists(directory):
    os.makedirs(directory)

create_dataframes_for_repeats(
    bucket_name, num_repeats, configs, save_folder=Path(directory)
)
num_repeats = 5
outputs = [None] * num_repeats
for config in configs:
    for repeat in range(num_repeats):
        all_output = pd.read_csv(
            f"{directory}/cytosim_actin_compression_velocity_{config}_repeat_{repeat}.csv"
        )
        outputs[repeat] = compression_metrics_workflow(
            all_output,
            [
                COMPRESSIONMETRIC.PEAK_ASYMMETRY,
                COMPRESSIONMETRIC.AVERAGE_PERP_DISTANCE,
                COMPRESSIONMETRIC.NON_COPLANARITY,
                COMPRESSIONMETRIC.TOTAL_FIBER_TWIST,
                COMPRESSIONMETRIC.SUM_BENDING_ENERGY,
                COMPRESSIONMETRIC.COMPRESSION_RATIO,
            ],
        )


# %%
outputs[0]

# %%
import matplotlib.pyplot as plt

metrics = [
    COMPRESSIONMETRIC.AVERAGE_PERP_DISTANCE,
    COMPRESSIONMETRIC.TOTAL_FIBER_TWIST,
    COMPRESSIONMETRIC.SUM_BENDING_ENERGY,
    COMPRESSIONMETRIC.PEAK_ASYMMETRY,
    COMPRESSIONMETRIC.NON_COPLANARITY,
    COMPRESSIONMETRIC.COMPRESSION_RATIO,
]
for metric in metrics:
    fig, ax = plt.subplots()
    for repeat in range(num_repeats):
        metric_by_time = outputs[repeat].groupby(["time"])[metric.value].mean()
        ax.plot(metric_by_time, label=f"repeat {repeat}")
    ax.legend()
    ax.set_xlabel("time")
    ax.set_ylabel(metric.value)
    ax.set_title(f"{metric.value} by time")

# %% [markdown]
# ### 2. Generate Simularium Outputs

# %%
from subcell_analysis.cytosim.post_process_cytosim import cytosim_to_simularium

s3_client = boto3.client("s3")
s3_client.download_file(
    "cytosim-working-bucket",
    "vary_compress_rate0006/outputs/2/fiber_segment_curvature.txt",
    "fiber_segment_curvature.txt",
)
input_data = cytosim_to_simularium("fiber_segment_curvature.txt")

# %%
cytosim_converter = CytosimConverter(input_data)
repeat = 0
for metric in metrics:
    metric_by_time = outputs[repeat].groupby(["time"])[metric.value].mean()
    cytosim_converter.add_plot(
        ScatterPlotData(
            title=f"{metric.value} over time",
            xaxis_title="Time",
            yaxis_title=metric.value,
            xtrace=np.arange(len(metric_by_time)) * 1e-5,
            ytraces={
                f"repeat {repeat}": metric_by_time,
            },
        )
    )

cytosim_converter.save("vary_compress_rate_0006_replicate0")

# %%
