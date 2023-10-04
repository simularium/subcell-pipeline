# %% [markdown]
# ## Generate Simularium Outputs

# %%
import boto3
import pandas as pd
import numpy as np
from subcell_analysis.cytosim.post_process_cytosim import cytosim_to_simularium
from subcell_analysis.compression_analysis import COMPRESSIONMETRIC

# %%
from simulariumio.cytosim import CytosimConverter
from simulariumio import ScatterPlotData, TrajectoryConverter

# %%
num_repeats = 5
config_id = 4

# %% [markdown]
# Download files (only needs to be done once)

# %%
s3_client = boto3.client("s3")
for repeat in range(num_repeats):
    s3_client.download_file(
        "cytosim-working-bucket",
        f"vary_compress_rate0006/outputs/{repeat}/fiber_segment_curvature.txt",
        f"data/fiber_segment_curvature_{repeat}.txt",
    )

# %% [markdown]
# ### Process single repeat

# %%
repeat = 0
input_file_path = f"data/fiber_segment_curvature_{repeat}.txt"

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

# %%
df_path = f"dataframes/actin_forces{config_id}_{repeat}_compression_metrics.csv"
df = pd.read_csv(df_path)

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

# %%
for metric in plot_metrics:
    metric_by_time = df.groupby(["time"])[metric.value].mean()
    cytosim_converter.add_plot(
        ScatterPlotData(
            title=f"{metric} over time",
            xaxis_title="Time",
            yaxis_title=metric.value,
            xtrace=np.arange(len(metric_by_time)) * 1e-5,
            ytraces={
                f"repeat {repeat}": metric_by_time,
            },
        )
    )

# %% [markdown]
# Save converted data

# %%
cytosim_converter.save(f"outputs/vary_compress_rate_0006_repeat_{repeat}")

# %% [markdown]
# ### Process multiple repeats

# %%
box_size = 3.0
scale_factor = 100
colors = ["#F0F0F0", "#0000FF", "#FF0000", "#00FF00", "#FF00FF"]

# %% [markdown]
# Create initial trajectory data object

# %%
input_file_path = f"data/fiber_segment_curvature_0.txt"
fiber_data = cytosim_to_simularium(
    input_file_path,
    box_size=box_size,
    scale_factor=scale_factor,
    color=colors[0],
    actin_number=0,
)
cytosim_converter = CytosimConverter(fiber_data)

trajectory_data = cytosim_converter._data

# %% [markdown]
# Append additional repeats to trajectory data object

# %%
for repeat in range(1, num_repeats):
    input_file_path = f"data/fiber_segment_curvature_{repeat}.txt"
    fiber_data = cytosim_to_simularium(
        input_file_path,
        box_size=box_size,
        scale_factor=scale_factor,
        color=colors[repeat],
        actin_number=repeat,
    )
    cytosim_converter = CytosimConverter(fiber_data)
    new_agent_data = cytosim_converter._data.agent_data

    trajectory_data.append_agents(new_agent_data)

# %%
all_repeats_converter = TrajectoryConverter(trajectory_data)

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

# %%
df_list = []
for repeat in range(num_repeats):
    df_path = f"dataframes/actin_forces{config_id}_{repeat}_compression_metrics.csv"
    df = pd.read_csv(df_path)
    df["repeat"] = repeat
    df_list.append(df)
df_all = pd.concat(df_list)

# %% [markdown]
# Add plots to converter object

# %%
for metric in plot_metrics:
    ytraces = {}
    for repeat, df_repeat in df_all.groupby("repeat"):
        ytraces[f"repeat {repeat}"] = df_repeat.groupby(["time"])[metric.value].mean()

    all_repeats_converter.add_plot(
        ScatterPlotData(
            title=f"{metric.value} over time",
            xaxis_title="Time",
            yaxis_title=metric.value,
            xtrace=np.arange(metric_by_time.shape[0]) * 1e-5,
            ytraces=ytraces,
            render_mode="lines",
        )
    )

# %% [markdown]
# Save converted data

# %%
all_repeats_converter.save(f"outputs/vary_compress_rate_0006_all_repeats")
