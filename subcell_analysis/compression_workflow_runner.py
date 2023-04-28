import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from .compression_analysis import get_asymmetry_of_peak, get_sum_bending_energy, get_third_component_variance, get_energy_asymmetry, get_total_fiber_twist, get_average_distance_from_end_to_end_axis, COMPRESSION_METRIC

def run_metric_calculation(
    all_points: pd.core.frame.DataFrame, metric: COMPRESSION_METRIC
) -> pd.core.frame.DataFrame:
    """
    Given cytosim output, run_metric_calculation calculates a chosen metric over all points in a fiber.

    Parameters
    ----------
    all_points: [(num_timepoints * num_points) x n columns] pandas dataframe
        all_points is a dataframe of cytosim outputs that is generated after some post-processing.
        includes [fiber_id, x_pos, y_pos, z_pos, xforce, yforce, zforce, segment_curvature, force_magnitude, segment_energy] columns and any metric columns
    metric: metric name as a string

    Returns
    -------
    all_points dataframe with calculated metric appended
    """
    all_points[metric] = np.nan
    for ct, (time, fiber_at_time) in enumerate(all_points.groupby("time")):
        if metric == COMPRESSION_METRIC.PEAK_ASYMMETRY:
            fiber_values = fiber_at_time[["xpos", "ypos", "zpos"]].values
            all_points.loc[fiber_at_time.index, metric] = get_asymmetry_of_peak(
                fiber_values
            )

        if metric == COMPRESSION_METRIC.SUM_BENDING_ENERGY:
            fiber_values = fiber_at_time[
                ["xpos", "ypos", "zpos", "segment_energy"]
            ].values
            all_points.loc[fiber_at_time.index, metric] = get_sum_bending_energy(
                fiber_values
            )

        if metric == COMPRESSION_METRIC.NON_COPLANARITY:
            fiber_values = fiber_at_time[["xpos", "ypos", "zpos"]].values
            all_points.loc[fiber_at_time.index, metric] = get_third_component_variance(
                fiber_values
            )

        if metric == COMPRESSION_METRIC.AVERAGE_PERP_DISTANCE:
            fiber_values = fiber_at_time[["xpos", "ypos", "zpos"]].values
            all_points.loc[
                fiber_at_time.index, metric
            ] = get_average_distance_from_end_to_end_axis(fiber_values)

        if metric == COMPRESSION_METRIC.TOTAL_FIBER_TWIST:
            fiber_values = fiber_at_time[["xpos", "ypos", "zpos"]].values
            all_points.loc[fiber_at_time.index, metric] = get_total_fiber_twist(
                fiber_values
            )

        if metric == COMPRESSION_METRIC.ENERGY_ASYMMETRY:
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
    Calculates chosen metrics from cytosim output of fiber positions and properties across timesteps

    Parameters
    ----------
    all_points: [(num_timepoints * num_points) x n columns] pandas dataframe
        all_points is a dataframe of cytosim outputs that is generated after some post-processing.
        includes [fiber_id, x_pos, y_pos, z_pos, xforce, yforce, zforce, segment_curvature, force_magnitude, segment_energy] columns and any metric columns
    metrics: [n] list of CM to calculate with metrics passed in as strings

    Returns
    -------
    all_points dataframe with chosen metrics appended as columns

    """
    for metric in metrics_to_calculate:
        all_points = run_metric_calculation(all_points, metric)
    return all_points

def plot_metric(all_points, metric):
    """
    Plots and saves metric values over time

    Parameters
    ----------
    all_points: [(num_timepoints * num_points) x n columns] pandas dataframe
        includes [fiber_id, x_pos, y_pos, z_pos, xforce, yforce, zforce, segment_curvature, force_magnitude, segment_energy] columns and any metric columns
    metric: metric name as a string
    """
    metric_by_time = all_points.groupby(level=["time"])[metric].mean()
    plt.plot(metric_by_time)
    plt.xlabel("Time")
    plt.ylabel(metric)
    plt.savefig(metric + "-time.pdf")
    plt.savefig(metric + "-time.png")

def plot_metric_list(all_points, metrics):
    # docs 
    for metric in metrics:
        plot_metric(all_points, metric)

