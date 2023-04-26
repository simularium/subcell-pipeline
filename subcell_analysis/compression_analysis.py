#!/usr/bin/env python
from typing import Any, Tuple
from pacmap import PaCMAP

import matplotlib as plt
import numpy as np

# TODO: consider creating a fiber class?


# TODO add actual types
def asymmetry(fibers_df: Any, timepoints: Any, fiber_at_time: Any) -> Any:
    last_timepoint = fibers_df.loc[timepoints[-1]]
    last_timepoint_tension = last_timepoint["segment_energy"]
    diff = np.zeros(len(last_timepoint_tension))
    for index, _timepoint in enumerate(last_timepoint_tension):
        np.round(fiber_at_time.shape[0] / 2).astype(int)
        diff[index] = np.abs(
            last_timepoint_tension[index] - last_timepoint_tension[-1 - index]
        )
    xs = np.linspace(0, 1, len(last_timepoint_tension))
    plt.scatter(xs, diff)
    plt.xlabel("Position along filament")
    plt.ylabel("Level of Asymmetry")
    # print(diff[index])
    print(len(diff))
    # write


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


def get_pacmap_embedding(polymer_trace_time_series: np.ndarray) -> np.ndarray:
    """
    Returns the pacmap embedding of the polymer trace time series.

    Parameters
    ----------
    polymer_trace: [k x t x n x 3] numpy array
        array containing the x,y,z positions of the polymer trace
        at each time point. k = number of traces, t = number of time points,
        n = number of points in each trace
        If k = 1, then the embedding is of a single trace

    Returns
    -------
    pacmap_embedding: [k x 2] numpy array
        pacmap embedding of each polymer trace
        If k = 1, then the embedding is of a single trace with size [t x 2]
    """
    embedding = PaCMAP(n_components=2, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0)

    reshaped_time_series = polymer_trace_time_series.reshape(
        polymer_trace_time_series.shape[0], -1
    )

    return embedding.fit_transform(reshaped_time_series, init="pca")
