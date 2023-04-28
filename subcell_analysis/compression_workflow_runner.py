import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .compression_analysis import (
    COMPRESSIONMETRIC,
    get_asymmetry_of_peak,
    get_average_distance_from_end_to_end_axis,
    get_energy_asymmetry,
    get_third_component_variance,
    get_total_fiber_twist,
)


def run_metric_calculation(
    all_points: pd.core.frame.DataFrame, metric: COMPRESSIONMETRIC
) -> pd.core.frame.DataFrame:
    """
    Given cytosim output, run_metric_calculation calculates a chosen metric over
    all points in a fiber.

    Parameters
    ----------
    all_points: [(num_timepoints * num_points) x n columns] pandas dataframe
        all_points is a dataframe of cytosim outputs that is generated after
        some post-processing.
        includes [fiber_id, x_pos, y_pos, z_pos, xforce, yforce, zforce,
        segment_curvature,
        force_magnitude, segment_energy] columns and any metric columns
    metric: COMPRESSIONMETRIC enum
        metric that includes chosen compression metric

    Returns
    -------
    all_points dataframe with calculated metric appended
    """
    all_points[metric] = np.nan
    for _ct, (_time, fiber_at_time) in enumerate(all_points.groupby("time")):
        if metric == COMPRESSIONMETRIC.PEAK_ASYMMETRY:
            fiber_values = fiber_at_time[["xpos", "ypos", "zpos"]].values
            all_points.loc[fiber_at_time.index, metric] = get_asymmetry_of_peak(
                fiber_values
            )

        if metric == COMPRESSIONMETRIC.NON_COPLANARITY:
            fiber_values = fiber_at_time[["xpos", "ypos", "zpos"]].values
            all_points.loc[fiber_at_time.index, metric] = get_third_component_variance(
                fiber_values
            )

        if metric == COMPRESSIONMETRIC.AVERAGE_PERP_DISTANCE:
            fiber_values = fiber_at_time[["xpos", "ypos", "zpos"]].values
            all_points.loc[
                fiber_at_time.index, metric
            ] = get_average_distance_from_end_to_end_axis(fiber_values)

        if metric == COMPRESSIONMETRIC.TOTAL_FIBER_TWIST:
            fiber_values = fiber_at_time[["xpos", "ypos", "zpos"]].values
            all_points.loc[fiber_at_time.index, metric] = get_total_fiber_twist(
                fiber_values
            )

        if metric == COMPRESSIONMETRIC.ENERGY_ASYMMETRY:
            fiber_values = fiber_at_time[
                ["xpos", "ypos", "zpos", "segment_energy"]
            ].values
            all_points.loc[fiber_at_time.index, metric] = get_energy_asymmetry(
                fiber_values
            )
    return all_points


def run_workflow(
    all_points: pd.core.frame.DataFrame, metrics_to_calculate: list
) -> pd.core.frame.DataFrame:
    """
    Calculates chosen metrics from cytosim output of fiber positions and
    properties across timesteps.

    Parameters
    ----------
    all_points: [(num_timepoints * num_points) x n columns] pandas dataframe
        all_points is a dataframe of cytosim outputs that is generated
        after some post-processing.
        includes [fiber_id, x_pos, y_pos, z_pos, xforce, yforce, zforce,
        segment_curvature,
        force_magnitude, segment_energy] columns and any metric columns
    metrics_to_calculate: [n] list of CM to calculate
        list of COMPRESSIONMETRICS


    Returns
    -------
    all_points dataframe with chosen metrics appended as columns

    """
    for metric in metrics_to_calculate:
        all_points = run_metric_calculation(all_points, metric)
    return all_points


def plot_metric(all_points: pd.core.frame.DataFrame, metric: COMPRESSIONMETRIC) -> None:
    """
    Plots and saves metric values over time.

    Parameters
    ----------
    all_points: [(num_timepoints * num_points) x n columns] pandas dataframe
        includes [fiber_id, x_pos, y_pos, z_pos, xforce, yforce, zforce,
        segment_curvature,
        force_magnitude, segment_energy] columns and any metric columns
    metric: metric name to be plotted
        chosen COMPRESSIONMETRIC

    """
    metric_by_time = all_points.groupby(level=["time"])[metric].mean()
    plt.plot(metric_by_time)
    plt.xlabel("Time")
    plt.ylabel(metric)
    plt.savefig(str(metric) + "-time.pdf")
    plt.savefig(str(metric) + "-time.png")


def plot_metric_list(all_points: pd.core.frame.DataFrame, metrics: list) -> None:
    # docs
    for metric in metrics:
        plot_metric(all_points, metric)
