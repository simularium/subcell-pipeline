"""
Methods for obtaining compression metric data and
plotting compression metrics vs time.
"""

from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
from io_collection.keys.check_key import check_key
from io_collection.load.load_dataframe import load_dataframe
from io_collection.save.save_dataframe import save_dataframe
from matplotlib import pyplot as plt

from subcell_pipeline.analysis.compression_metrics.compression_metric import (
    CompressionMetric,
)
from subcell_pipeline.analysis.compression_metrics.constants import (
    DEFAULT_COMPRESSION_DISTANCE,
    SIMULATOR_COLOR_MAP,
)


def get_compression_metric_data(
    bucket: str,
    series_name: str,
    condition_keys: list[str],
    random_seeds: list[int],
    metrics: list[CompressionMetric],
    recalculate: bool = False,
) -> pd.DataFrame:
    """
    Load or create merged data with metrics for given conditions and random seeds.

    If merged data already exists, load the data.
    Otherwise, iterate through the conditions and seeds to merge the data.

    Parameters
    ----------
    bucket
        Name of S3 bucket for input and output files.

    series_name
        Name of simulation series.

    condition_keys
        List of condition keys.

    random_seeds
        Random seeds for simulations.

    metrics
        List of metrics to calculate.

    recalculate
        True if data should be recalculated, False otherwise.

    Returns
    -------
    :
        Merged dataframe with one row per fiber with calculated metrics.
    """

    data_key = f"{series_name}/analysis/{series_name}_compression_metrics.csv"

    # Return data, if merged data already exists.
    if check_key(bucket, data_key) and not recalculate:
        print(
            f"Dataframe [ { data_key } ] already exists. Loading existing merged data."
        )
        return load_dataframe(bucket, data_key, dtype={"key": "str"})

    all_metrics: list[pd.DataFrame] = []

    for condition_key in condition_keys:
        series_key = f"{series_name}_{condition_key}" if condition_key else series_name

        for seed in random_seeds:
            print(
                f"Loading samples and calculating metrics for "
                f"[ {condition_key} ] seed [ {seed} ]"
            )

            sample_key = f"{series_name}/samples/{series_key}_{seed:06d}.csv"
            samples = load_dataframe(bucket, sample_key)

            metric_data = calculate_compression_metrics(samples, metrics)
            metric_data["seed"] = seed
            metric_data["key"] = condition_key

            all_metrics.append(metric_data)

    metrics_dataframe = pd.concat(all_metrics)
    save_dataframe(bucket, data_key, metrics_dataframe, index=False)

    return metrics_dataframe


def calculate_compression_metrics(
    df: pd.DataFrame, metrics: List[Any], **options: Dict[str, Any]
) -> pd.DataFrame:
    """
    Calculate compression metrics for a single simulation condition and seed.

    Parameters
    ----------
    df
        The input DataFrame for a single simulator.

    metrics
        The list of metrics to calculate.

    **options
        Additional options for the calculation.

    Returns
    -------
    :
        The DataFrame with the calculated metrics.
    """
    time_values = df["time"].unique()
    df_metrics = pd.DataFrame(
        index=time_values, columns=[metric.value for metric in metrics]
    )

    for time, fiber_at_time in df.groupby("time"):
        polymer_trace = fiber_at_time[["xpos", "ypos", "zpos"]].values
        for metric in metrics:
            df_metrics.loc[time, metric.value] = metric.calculate_metric(
                polymer_trace=polymer_trace, **options
            )

    df_metrics = df_metrics.reset_index().rename(columns={"index": "time"})
    df_metrics["normalized_time"] = df_metrics["time"] / df_metrics["time"].max()

    return df_metrics


def save_compression_metrics(
    data: pd.DataFrame, save_location: str, save_key: str
) -> None:
    """
    Save combined compression metrics data.

    Parameters
    ----------
    data
        Compression metrics data.
    save_location
        Location for output file (local path or S3 bucket).
    save_key
        Name key for output file.
    """

    save_dataframe(save_location, save_key, data, index=False)


def plot_metrics_vs_time(
    df: pd.DataFrame,
    metrics: List[CompressionMetric],
    figure_path: Union[Path, None] = None,
    suffix: str = "",
    compression_distance: float = DEFAULT_COMPRESSION_DISTANCE,
    use_real_time: bool = False,
) -> None:
    """
    Plot metrics vs time.

    Parameters
    ----------
    df
        The input DataFrame.

    metrics
        The list of metrics to plot.

    figure_path
        The path to save the figure.

    suffix
        The suffix to append to the figure filename.
        Defaults to "".

    compression_distance
        The compression distance in nm.
        Defaults to 150.0.

    use_real_time
        Whether to use real time for the x-axis.
        Defaults to False.
    """
    num_velocities = df["velocity"].nunique()
    total_time = 1.0
    time_label = "Normalized Time"
    plt.rcParams.update({"font.size": 16})

    for metric in metrics:
        fig, axs = plt.subplots(
            1, num_velocities, figsize=(num_velocities * 5, 5), sharey=True, dpi=300
        )
        axs = axs.ravel()
        for ct, (velocity, df_velocity) in enumerate(df.groupby("velocity")):
            if use_real_time:
                # type checker is unable to infer the datatype of velocity
                total_time = compression_distance / velocity  # type: ignore
                time_label = "Time (s)"
            for simulator, df_simulator in df_velocity.groupby("simulator"):
                for repeat, df_repeat in df_simulator.groupby("repeat"):
                    if repeat == 0:
                        label = f"{simulator}"
                    else:
                        label = "_nolegend_"
                    xvals = np.linspace(0, 1, df_repeat["time"].nunique()) * total_time
                    yvals = df_repeat.groupby("time")[metric.value].mean()

                    # type checker is unable to infer the datatype of velocity
                    axs[ct].plot(
                        xvals,
                        yvals,
                        label=label,
                        color=SIMULATOR_COLOR_MAP[simulator],  # type: ignore
                        alpha=0.6,
                    )
            axs[ct].set_title(f"Velocity: {velocity}")
            if ct == 0:
                axs[ct].legend()
        fig.supxlabel(time_label)
        fig.supylabel(metric.label())
        fig.tight_layout()
        if figure_path is not None:
            fig.savefig(figure_path / f"{metric.value}_vs_time{suffix}.png")


def plot_metric_distribution(
    df: pd.DataFrame,
    metrics: List[CompressionMetric],
    figure_path: Union[Path, None] = None,
    suffix: str = "",
) -> None:
    """
    Plot metrics vs time.

    Parameters
    ----------
    df
        The input DataFrame.

    metrics
        The list of metrics to plot.

    figure_path
        The path to save the figure.

    suffix
        The suffix to append to the figure filename.
        Defaults to "".

    """
    num_velocities = df["velocity"].nunique()
    plt.rcParams.update({"font.size": 16})

    for metric in metrics:
        fig, axs = plt.subplots(
            1,
            num_velocities,
            figsize=(num_velocities * 5, 5),
            sharey=True,
            sharex=True,
            dpi=300,
        )
        axs = axs.ravel()
        for ct, (velocity, df_velocity) in enumerate(df.groupby("velocity")):
            metric_values = df_velocity[metric.value]
            bins = np.linspace(np.nanmin(metric_values), np.nanmax(metric_values), 20)
            for simulator, df_simulator in df_velocity.groupby("simulator"):
                axs[ct].hist(
                    df_simulator[metric.value],
                    label=f"{simulator}",
                    color=SIMULATOR_COLOR_MAP[simulator],  # type: ignore
                    alpha=0.7,
                    bins=bins,
                )
            axs[ct].set_title(f"Velocity: {velocity}")
            if ct == 0:
                axs[ct].legend()
        fig.supxlabel(metric.label())
        fig.supylabel("Count")
        fig.tight_layout()
        if figure_path is not None:
            fig.savefig(figure_path / f"{metric.value}_histogram{suffix}.png")
