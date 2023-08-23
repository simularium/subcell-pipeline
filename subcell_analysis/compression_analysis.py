#!/usr/bin/env python
from enum import Enum
from typing import Tuple

import numpy as np
from pacmap import PaCMAP
from sklearn.decomposition import PCA

# TODO: consider creating a fiber class?

ABS_TOL = 1e-16


class COMPRESSIONMETRIC(Enum):
    NON_COPLANARITY = "NON_COPLANARITY"
    PEAK_ASYMMETRY = "PEAK_ASYMMETRY"
    SUM_BENDING_ENERGY = "SUM_BENDING_ENERGY"
    AVERAGE_PERP_DISTANCE = "AVERAGE_PERP_DISTANCE"
    TOTAL_FIBER_TWIST = "TOTAL_FIBER_TWIST"
    ENERGY_ASYMMETRY = "ENERGY_ASYMMETRY"


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

    # if all perpendicular distances are zero, return 0
    if np.all(perp_distances < ABS_TOL):
        return 0

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
        in number of rotations
    """
    (
        perp_distances,
        _,
        projection_positions,
    ) = get_end_to_end_axis_distances_and_projections(polymer_trace=polymer_trace)

    # if all perpendicular distances are zero, return 0
    if np.all(perp_distances < ABS_TOL):
        return 0

    perp_vectors = polymer_trace - projection_positions
    perp_vec_lengths = np.linalg.norm(perp_vectors, axis=1)
    perp_vec_lengths[perp_vec_lengths < ABS_TOL] = 1
    perp_vectors = perp_vectors / perp_vec_lengths[:, None]
    perp_vectors[perp_vec_lengths < ABS_TOL] = [np.nan, np.nan, np.nan]

    consecutive_angles = np.arccos(
        np.einsum("ij,ij->i", perp_vectors[1:], perp_vectors[:-1])
    )
    total_twist = np.nansum(consecutive_angles) / 2 / np.pi

    return total_twist


def get_pacmap_embedding(polymer_trace_time_series: np.ndarray) -> np.ndarray:
    """
    Returns the pacmap embedding of the polymer trace time series.

    Parameters
    ----------
    polymer_trace_time_series: [k x t x n x 3] numpy array
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

    return embedding.fit_transform(reshaped_time_series)


def get_third_component_variance(
    polymer_trace: np.ndarray,
) -> float:
    """
    Returns the third PCA component given the x,y,z positions of a fiber at
    a given time. This component reflects non-coplanarity/out of planeness.

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


def get_energy_asymmetry(
    fiber_energy: np.ndarray,
) -> float:
    """
    Returns the sum bending energy given a single fiber x,y,z positions
    and segment energy values.

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
    for index, _point in enumerate(fiber_energy):
        diff[index] = np.abs(fiber_energy[index] - fiber_energy[-1 - index])
        if index == middle_index:
            break
    return np.sum(diff)


def get_sum_bending_energy(
    fiber_energy: np.ndarray,
) -> float:
    return fiber_energy[3].sum()
