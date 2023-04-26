#!/usr/bin/env python
from typing import Any, Tuple
from sklearn.decomposition import PCA
import matplotlib as plt
import numpy as np
import pandas as pd


# TODO: consider creating a fiber class?


def get_end_to_end_axis_distances_and_projections(
    polymer_trace: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns the distances of the polymer trace points from the end-to-end axis.
    Here, the end-to-end axis is defined as the line joining the first and last
    fiber point.

    Parameters
    ----------
    polymer_trace: [n x 3] numpy array
        array containing the x,y,z positions of the polymer trace points
        at a given time

    Returns
    -------
    perp_distances: [n x 1] numpy array
        perpendicular distances of the polymer trace from the end-to-end axis

    scaled_projections: [n x 1] numpy array
        length of fiber point projections along the end-to-end axis, scaled
        by axis length.
        Can be negative.

    projection_positions: [n x 3] numpy array
        positions of points on the end-to-end axis that are
        closest from the respective points in the polymer trace.
        The distance from projection_positions
        to the trace points is the shortest distance from the end-to-end axis
    """
    end_to_end_axis = polymer_trace[-1] - polymer_trace[0]
    end_to_end_axis_length = np.linalg.norm(end_to_end_axis)

    position_vectors = polymer_trace - polymer_trace[0]
    dot_products = np.dot(position_vectors, end_to_end_axis)

    projections = dot_products / end_to_end_axis_length
    projection_positions = (
        polymer_trace[0]
        + projections[:, None] * end_to_end_axis / end_to_end_axis_length
    )

    perp_distances = np.linalg.norm(polymer_trace - projection_positions, axis=1)
    scaled_projections = projections / end_to_end_axis_length

    return perp_distances, scaled_projections, projection_positions


def get_average_distance_from_end_to_end_axis(
    polymer_trace: np.ndarray,
) -> float:
    """
    Returns the average perpendicular distance of polymer trace points from
    the end-to-end axis.

    Parameters
    ----------
    polymer_trace: [n x 3] numpy array
        array containing the x,y,z positions of the polymer trace
        at a given time

    Returns
    -------
    avg_perp_distance: float
        average perpendicular distance of polymer trace points from the
        end-to-end axis
    """
    perp_distances, _, _ = get_end_to_end_axis_distances_and_projections(
        polymer_trace=polymer_trace
    )
    avg_perp_distance = np.nanmean(perp_distances)

    return avg_perp_distance


def get_asymmetry_of_peak(
    polymer_trace: np.ndarray,
) -> float:
    """
    returns the scaled distance of the projection of the peak from the
    end-to-end axis midpoint.

    Parameters
    ----------
    polymer_trace: [n x 3] numpy array
        array containing the x,y,z positions of the polymer trace
        at a given time

    Returns
    -------
    peak_asym: float
        scaled distance of the projection of the peak from the axis midpoint
    """
    (
        perp_distances,
        scaled_projections,
        _,
    ) = get_end_to_end_axis_distances_and_projections(polymer_trace=polymer_trace)
    projection_of_peak = scaled_projections[perp_distances == np.max(perp_distances)]
    peak_asym = np.max(projection_of_peak - 0.5)  # max kinda handles multiple peaks

    return peak_asym


def get_total_fiber_twist(
    polymer_trace: np.ndarray,
) -> float:
    """
    Returns the sum of angles between consecutive vectors from the
    polymer trace points to the end-to-end axis.

    Parameters
    ----------
    polymer_trace: [n x 3] numpy array
        array containing the x,y,z positions of the polymer trace
        at a given time

    Returns
    -------
    total_twist: float
        sum of angles between vectors from trace points to axis
    """
    _, _, projection_positions = get_end_to_end_axis_distances_and_projections(
        polymer_trace=polymer_trace
    )
    perp_vectors = polymer_trace - projection_positions
    perp_vectors = perp_vectors / np.linalg.norm(perp_vectors, axis=1)[:, None]
    consecutive_angles = np.arccos(
        np.einsum("ij,ij->i", perp_vectors[1:], perp_vectors[:-1])
    )
    total_twist = np.nansum(consecutive_angles)

    return total_twist


def get_third_component_variance(
    polymer_trace: np.ndarray,
) -> float:
    """
    Returns the third PCA component given the x,y,z positions of a fiber at
    a given time. This component reflects non-coplanarity/out of planeness

    Parameters
    ----------
    polymer_trace: [n x 3] numpy array
        array containing the x,y,z positions of the polymer trace
        at a given time

    Returns
    -------
    third_component_variance: float
        noncoplanarity of fiber
    """
    pca = PCA(n_components=3)
    pca.fit(polymer_trace)
    return pca.explained_variance_ratio_[2]


def run_metric_calculation(
    all_points: pd.core.frame.DataFrame, metric: str
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
        if metric == "peak_asymmetry":
            fiber_values = fiber_at_time[["xpos", "ypos", "zpos"]].values
            all_points.loc[fiber_at_time.index, metric] = get_asymmetry_of_peak(
                fiber_values
            )

        if metric == "sum_bending_energy":
            all_points.loc[fiber_at_time.index, metric] = fiber_at_time["segment_energy"].sum()

        if metric == "non_coplanarity":
            fiber_values = fiber_at_time[["xpos", "ypos", "zpos"]].values
            all_points.loc[fiber_at_time.index, metric] = get_third_component_variance(
                fiber_values
            )

        if metric == "average_perp_distance":
            fiber_values = fiber_at_time[["xpos", "ypos", "zpos"]].values
            all_points.loc[
                fiber_at_time.index, metric
            ] = get_average_distance_from_end_to_end_axis(fiber_values)

        if metric == "total_fiber_twist":
            fiber_values = fiber_at_time[["xpos", "ypos", "zpos"]].values
            all_points.loc[fiber_at_time.index, metric] = get_total_fiber_twist(
                fiber_values
            )

        if metric == "energy_asymmetry":
            all_points.loc[fiber_at_time.index, metric] = get_energy_asymmetry(
                fiber_at_time["segment_energy"].values
            )
    return all_points


def run_compression_workflow(
    all_points: pd.core.frame.DataFrame, metrics_to_calculate: list
) -> pd.core.frame.DataFrame:
    """
    Calculates chosen metrics from cytosim output of fiber positions and properties across timesteps

    Parameters
    ----------
    all_points: [(num_timepoints * num_points) x n columns] pandas dataframe
        all_points is a dataframe of cytosim outputs that is generated after some post-processing.
        includes [fiber_id, x_pos, y_pos, z_pos, xforce, yforce, zforce, segment_curvature, force_magnitude, segment_energy] columns and any metric columns
    metrics: [n] list of metrics to calculate with metrics passed in as strings

    Returns
    -------
    all_points dataframe with chosen metrics appended as columns

    """
    for metric in metrics_to_calculate:
        all_points = run_metric_calculation(all_points, metric)
    return all_points




def get_energy_asymmetry(
    fiber_energy: np.ndarray,
) -> float:
    """
    Returns the sum bending energy given a single fiber x,y,z positions and segment energy values

    Parameters
    ----------
    fiber_energy: [n x 4] numpy array
        array containing the x,y,z positions of the polymer trace and segment energy
        at a given time

    Returns
    -------
    total_energy: float
        energy of a vector at a given time
    """

    middle_index = np.round(len(fiber_energy) / 2).astype(int)
    diff = np.zeros(len(fiber_energy))
    for index, point in enumerate(fiber_energy):
        diff[index] = np.abs(fiber_energy[index] - fiber_energy[-1 - index])
        if index == middle_index:
            break
    return np.sum(diff)


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
