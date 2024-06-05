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
configs = ["no_linkers_no_compression"]


# %%
directory = "../data/dataframes"
if not os.path.exists(directory):
    os.makedirs(directory)

create_dataframes_for_repeats(
    bucket_name, num_repeats, configs, save_folder=Path(directory)
)

# %%
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
                COMPRESSIONMETRIC.CALC_BENDING_ENERGY,
                COMPRESSIONMETRIC.CONTOUR_LENGTH,
            ],
        )

# %%
outputs

# %%
# %%
outputs[0]

# %%
import matplotlib.pyplot as plt

metrics = [
    COMPRESSIONMETRIC.AVERAGE_PERP_DISTANCE,
    COMPRESSIONMETRIC.TOTAL_FIBER_TWIST,
    COMPRESSIONMETRIC.CALC_BENDING_ENERGY,
    COMPRESSIONMETRIC.PEAK_ASYMMETRY,
    COMPRESSIONMETRIC.NON_COPLANARITY,
    COMPRESSIONMETRIC.CONTOUR_LENGTH,
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


# %%
