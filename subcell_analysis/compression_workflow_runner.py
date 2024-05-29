import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .compression_analysis import (
    COMPRESSIONMETRIC,
    get_asymmetry_of_peak,
    get_average_distance_from_end_to_end_axis,
    get_energy_asymmetry,
    get_sum_bending_energy,
    get_third_component_variance,
    get_total_fiber_twist,
)

ABS_TOL = 1e-6


def run_metric_calculation(
    all_points: pd.core.frame.DataFrame, metric: COMPRESSIONMETRIC, **options: dict
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
    **options: dict
        Additional options as key-value pairs.

    Returns
    -------
    all_points dataframe with calculated metric appended
    """
    all_points[metric.value] = np.nan
    for _ct, (_time, fiber_at_time) in enumerate(all_points.groupby("time")):
        if metric == COMPRESSIONMETRIC.PEAK_ASYMMETRY:
            polymer_trace = fiber_at_time[["xpos", "ypos", "zpos"]].values
            all_points.loc[fiber_at_time.index, metric.value] = get_asymmetry_of_peak(
                polymer_trace,
                **options,
            )

        if metric == COMPRESSIONMETRIC.NON_COPLANARITY:
            polymer_trace = fiber_at_time[["xpos", "ypos", "zpos"]].values
            all_points.loc[
                fiber_at_time.index, metric.value
            ] = get_third_component_variance(
                polymer_trace,
                **options,
            )

        if metric == COMPRESSIONMETRIC.AVERAGE_PERP_DISTANCE:
            polymer_trace = fiber_at_time[["xpos", "ypos", "zpos"]].values
            all_points.loc[
                fiber_at_time.index, metric.value
            ] = get_average_distance_from_end_to_end_axis(polymer_trace, **options)

        if metric == COMPRESSIONMETRIC.TOTAL_FIBER_TWIST:
            polymer_trace = fiber_at_time[["xpos", "ypos", "zpos"]].values
            all_points.loc[fiber_at_time.index, metric.value] = get_total_fiber_twist(
                polymer_trace,
                compression_axis=0,
                signed=True,
                tolerance=ABS_TOL,
                **options,
            )

        if metric == COMPRESSIONMETRIC.ENERGY_ASYMMETRY:
            polymer_trace = fiber_at_time[
                ["xpos", "ypos", "zpos", "segment_energy"]
            ].values
            all_points.loc[fiber_at_time.index, metric.value] = get_energy_asymmetry(
                polymer_trace,
                **options,
            )

        if metric == COMPRESSIONMETRIC.SUM_BENDING_ENERGY:
            polymer_trace = fiber_at_time[
                ["xpos", "ypos", "zpos", "segment_energy"]
            ].values
            all_points.loc[fiber_at_time.index, metric.value] = get_sum_bending_energy(
                polymer_trace, **options
            )
    return all_points


def compression_metrics_workflow(
    all_points: pd.core.frame.DataFrame, metrics_to_calculate: list, **options: dict
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
    **options: dict
        Additional options as key-value pairs.

    Returns
    -------
    all_points dataframe with chosen metrics appended as columns

    """
    for metric in metrics_to_calculate:
        all_points = run_metric_calculation(all_points, metric, **options)
    return all_points


def plot_metric(all_points: pd.core.frame.DataFrame, metric: COMPRESSIONMETRIC) -> None:
    """
    Plots and saves metric values over time.
    gi
    Parameters
    ----------
    all_points: [(num_timepoints * num_points) x n columns] pandas dataframe
        includes [fiber_id, x_pos, y_pos, z_pos, xforce, yforce, zforce,
        segment_curvature,
        force_magnitude, segment_energy] columns and any metric columns
    metric: metric name to be plotted
        chosen COMPRESSIONMETRIC.

    """
    metric_by_time = all_points.groupby(["time"])[metric].mean()
    plt.plot(metric_by_time)
    plt.xlabel("Time")
    plt.ylabel(metric)
    # Save files if needed.
    # plt.savefig(str(metric) + "-time.pdf")
    # plt.savefig(str(metric) + "-time.png")


def plot_metric_list(all_points: pd.core.frame.DataFrame, metrics: list) -> None:
    # docs
    for metric in metrics:
        plot_metric(all_points, metric)
