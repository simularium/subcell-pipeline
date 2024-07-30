"""Methods to calculate metrics from polymer trace data."""

from typing import Any, Dict, Tuple

import numpy as np
from sklearn.decomposition import PCA

from subcell_pipeline.analysis.compression_metrics.constants import (
    ABSOLUTE_TOLERANCE,
    DEFAULT_BENDING_CONSTANT,
)
from subcell_pipeline.analysis.compression_metrics.vectors import (
    get_end_to_end_unit_vector,
)


def get_end_to_end_axis_distances_and_projections(
    polymer_trace: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate distances of the polymer trace points from the end-to-end axis.

    Here, the end-to-end axis is defined as the line joining the first and last
    fiber point.

    Parameters
    ----------
    polymer_trace
        Array containing the x,y,z positions of the polymer trace.

    Returns
    -------
    perp_distances
        Perpendicular distances of the polymer trace from the end-to-end axis.
    scaled_projections
        Length of fiber point projections along the end-to-end axis, scaled by
        axis length. Can be negative.
    projection_positions
        Positions of points on the end-to-end axis that are closest from the
        respective points in the polymer trace. The distance from
        projection_positions to the trace points is the shortest distance from
        the end-to-end axis.
    """
    end_to_end_axis = get_end_to_end_unit_vector(polymer_trace=polymer_trace)
    end_to_end_axis_length = np.linalg.norm(polymer_trace[-1] - polymer_trace[0])

    position_vectors = polymer_trace - polymer_trace[0]
    projections = np.dot(position_vectors, end_to_end_axis)
    scaled_projections = projections / end_to_end_axis_length

    projection_positions = polymer_trace[0] + projections[:, None] * end_to_end_axis

    perp_distances = np.linalg.norm(polymer_trace - projection_positions, axis=1)

    return perp_distances, scaled_projections, projection_positions


def get_average_distance_from_end_to_end_axis(
    polymer_trace: np.ndarray, **options: Dict[str, Any]
) -> float:
    """
    Calculate the average perpendicular distance of polymer trace points from
    the end-to-end axis.

    Parameters
    ----------
    polymer_trace
        Array containing the x,y,z positions of the polymer trace.
    **options
        Additional options as key-value pairs. Unused.

    Returns
    -------
    :
        Average perpendicular distance of polymer trace points from the
        end-to-end axis.
    """
    perp_distances, _, _ = get_end_to_end_axis_distances_and_projections(
        polymer_trace=polymer_trace
    )
    avg_perp_distance = np.nanmean(perp_distances)

    return float(avg_perp_distance)


def get_asymmetry_of_peak(
    polymer_trace: np.ndarray, **options: Dict[str, Any]
) -> float:
    """
    Calculate the scaled distance of the projection of the peak from the
    end-to-end axis midpoint.

    Parameters
    ----------
    polymer_trace
        Array containing the x,y,z positions of the polymer trace.
    **options
        Additional options as key-value pairs. Unused.

    Returns
    -------
    :
        Scaled distance of the projection of the peak from the axis midpoint.
    """
    (
        perp_distances,
        scaled_projections,
        _,
    ) = get_end_to_end_axis_distances_and_projections(polymer_trace=polymer_trace)

    # if all perpendicular distances are zero, return 0
    if np.all(perp_distances < ABSOLUTE_TOLERANCE):
        return 0

    projection_of_peak = scaled_projections[perp_distances == np.max(perp_distances)]
    peak_asym = np.max(
        np.abs(projection_of_peak - 0.5)
    )  # max kinda handles multiple peaks

    return peak_asym


def get_pca_polymer_trace_projection(polymer_trace: np.ndarray) -> np.ndarray:
    """
    Calculate the PCA projection of the polymer trace.

    Parameters
    ----------
    polymer_trace
        Array containing the x,y,z positions of the polymer trace.

    Returns
    -------
    :
        PCA projection of the polymer trace.
    """
    pca = fit_pca_to_polymer_trace(polymer_trace=polymer_trace)
    return pca.transform(polymer_trace)


def get_contour_length_from_trace(
    polymer_trace: np.ndarray, **options: Dict[str, Any]
) -> float:
    """
    Calculate the sum of inter-monomer distances in the trace.

    Parameters
    ----------
    polymer_trace
        Array containing the x,y,z positions of the polymer trace.
    **options
        Additional options as key-value pairs. Unused.

    Returns
    -------
    :
        Sum of inter-monomer distances in the trace.
    """
    return np.sum(np.linalg.norm(np.diff(polymer_trace, axis=0), axis=1))


def get_bending_energy_from_trace(
    polymer_trace: np.ndarray, **options: Dict[str, Any]
) -> float:
    """
    Calculate the bending energy per monomer of a polymer trace.

    Parameters
    ----------
    polymer_trace
        Array containing the x,y,z positions of the polymer trace.
    **options
        Additional options as key-value pairs:

        bending_constant: float
            Bending constant of the fiber in pN nm.

    Returns
    -------
    :
        Bending energy per monomer of the polymer trace.
    """
    bending_constant = options.get("bending_constant", DEFAULT_BENDING_CONSTANT)

    assert isinstance(bending_constant, (float, np.floating))

    cos_angle = np.zeros(len(polymer_trace) - 2)
    for ind in range(len(polymer_trace) - 2):
        vec1 = polymer_trace[ind + 1] - polymer_trace[ind]
        vec2 = polymer_trace[ind + 2] - polymer_trace[ind + 1]

        if np.isclose(np.linalg.norm(vec1), 0.0) or np.isclose(
            np.linalg.norm(vec2), 0.0
        ):
            # TODO handle this differently?
            cos_angle[ind] = 0.0
            print("Warning: zero vector in bending energy calculation.")
            continue

        cos_angle[ind] = (
            np.dot(vec1, vec2) / np.linalg.norm(vec1) / np.linalg.norm(vec2)
        )

    # since the bending constant is obtained from a kwargs dictionary
    # the type checker is unable to infer its type
    energy = bending_constant * (1 - np.nanmean(cos_angle))

    return energy.item()


def get_2d_polymer_trace(
    polymer_trace: np.ndarray, compression_axis: int = 0
) -> np.ndarray:
    """
    Get the 2D projection of the polymer trace.

    Parameters
    ----------
    polymer_trace
        Array containing the x,y,z positions of the polymer trace.
    compression_axis
        The axis along which the polymer trace is compressed.

    Returns
    -------
    :
        The 2D projection of the polymer trace.
    """
    if polymer_trace.shape[1] == 2:
        return polymer_trace

    return polymer_trace[
        :, [ax for ax in range(polymer_trace.shape[1]) if ax != compression_axis]
    ]


def get_normalized_tangent_vectors(polymer_trace: np.ndarray) -> np.ndarray:
    """
    Calculate the normalized tangent vectors for a polymer trace.

    Parameters
    ----------
    polymer_trace
        Array containing the x,y,z positions of the polymer trace.

    Returns
    -------
    :
        The normalized tangent vectors for the polymer.
    """
    tangents = np.diff(polymer_trace, axis=0)

    tangent_lengths = np.linalg.norm(tangents, axis=1)

    if np.all(tangent_lengths < ABSOLUTE_TOLERANCE):
        return np.zeros_like(tangents)

    tangents /= tangent_lengths[:, np.newaxis]

    return tangents


def get_twist_angle(polymer_trace: np.ndarray, **options: Dict[str, Any]) -> float:
    """
    Calculate the twist angle of the polymer trace.

    Parameters
    ----------
    polymer_trace
        Array containing the x,y,z positions of the polymer trace.
    **options
        Additional options as key-value pairs. Unused.

    Returns
    -------
    :
        Twist angle of the polymer trace in degrees.
    """
    compression_axis = options.get("compression_axis", 0)
    assert isinstance(compression_axis, int)

    trace_2d = get_2d_polymer_trace(
        polymer_trace=polymer_trace, compression_axis=compression_axis
    )

    tangents = get_normalized_tangent_vectors(
        polymer_trace=trace_2d,
    )

    angle = get_angle_between_vectors(tangents[0], -tangents[-1], signed=False)
    chirality = get_chirality(polymer_trace=polymer_trace)

    return chirality * angle * 180 / np.pi


def get_chirality(polymer_trace: np.ndarray, **options: Dict[str, Any]) -> float:
    """
    Calculate the chirality of a polymer trace.

    Parameters
    ----------
    polymer_trace
        Array containing the x,y,z positions of the polymer trace.
    **options
        Additional options as key-value pairs. Unused.

    Returns
    -------
    :
        Chirality of the polymer trace.
    """
    trace_2d = get_2d_polymer_trace(polymer_trace=polymer_trace)
    tangents = get_normalized_tangent_vectors(polymer_trace=trace_2d)

    chirality = 0
    for i in range(len(tangents) - 1):
        cross_product = np.cross(tangents[i], tangents[i + 1])
        if cross_product > 0:
            chirality += 1
        elif cross_product < 0:
            chirality -= 1

    return np.sign(chirality)


def get_total_fiber_twist(
    polymer_trace: np.ndarray, **options: Dict[str, Any]
) -> float:
    """
    Calculate the total twist of a polymer trace using the normal vectors.

    Parameters
    ----------
    polymer_trace
        Array containing the x,y,z positions of the polymer trace.
    **options
        Additional options as key-value pairs:

        signed: bool
            Whether to return the signed or unsigned total twist.

    Returns
    -------
    :
        Total twist of the polymer trace.
    """
    signed = options.get("signed", False)
    assert isinstance(signed, bool)

    tangents = np.diff(polymer_trace, axis=0)
    tangents /= np.linalg.norm(tangents, axis=1)[:, np.newaxis]

    normals = np.cross(tangents[:-1], tangents[1:])
    normal_lengths = np.linalg.norm(normals, axis=1)
    if np.all(normal_lengths < ABSOLUTE_TOLERANCE):
        return 0
    normals = normals / normal_lengths[:, np.newaxis]

    twists = []
    for i in range(1, len(normals)):
        angle = get_angle_between_vectors(normals[i - 1], normals[i], signed=signed)

        twists.append(angle)

    # Sum the twist angles to get the total twist
    total_twist = np.sum(twists)

    # Normalize by the contour length
    total_twist /= get_contour_length_from_trace(polymer_trace)

    return total_twist


def get_total_fiber_twist_project(
    polymer_trace: np.ndarray, **options: Dict[str, Any]
) -> float:
    """
    Calculate the total twist using projections of the polymer trace in the 2nd
    and 3rd dimension.

    Parameters
    ----------
    polymer_trace
        Array containing the x,y,z positions of the polymer trace.
    **options
        Additional options as key-value pairs:

        compression_axis: int
            Axis along which the polymer trace is compressed.
        signed: bool
            Whether to return the signed or unsigned total twist.
        tolerance: float
            Tolerance for vector length.

    Returns
    -------
    :
        Sum of angles between PCA projection vectors.
    """
    compression_axis = options.get("compression_axis", 0)
    signed = options.get("signed", True)
    tolerance = options.get("tolerance", ABSOLUTE_TOLERANCE)

    assert isinstance(signed, bool)
    assert isinstance(tolerance, (float, np.floating))
    assert isinstance(compression_axis, int)

    trace_2d = get_2d_polymer_trace(
        polymer_trace=polymer_trace, compression_axis=compression_axis
    )
    trace_2d = trace_2d - np.mean(trace_2d, axis=0)

    return get_total_fiber_twist_2d(trace_2d, signed=signed, tolerance=tolerance)


def get_total_fiber_twist_pca(
    polymer_trace: np.ndarray, tolerance: float = ABSOLUTE_TOLERANCE
) -> float:
    """
    Calculate the total twist using PCA projections of the polymer trace in the
    2nd and 3rd dimension.

    Parameters
    ----------
    polymer_trace
        Array containing the x,y,z positions of the polymer trace.
    tolerance
        Tolerance for vector length.

    Returns
    -------
    :
        Sum of angles between PCA projection vectors.
    """
    pca_trace = get_pca_polymer_trace_projection(polymer_trace=polymer_trace)
    pca_trace_2d = pca_trace[:, 1:]

    return get_total_fiber_twist_2d(pca_trace_2d, tolerance=tolerance)


def get_angle_between_vectors(
    vec1: np.ndarray, vec2: np.ndarray, signed: bool = False
) -> float:
    """
    Calculate the signed angle between two vectors.

    Parameters
    ----------
    vec1
        The first vector.
    vec2
        The second vector.
    signed
        True to get signed angle between vectors, False otherwise.

    Returns
    -------
    :
        Angle between vec1 and vec2.
    """
    vec1_length = np.linalg.norm(vec1)
    vec2_length = np.linalg.norm(vec2)

    if vec1_length < ABSOLUTE_TOLERANCE or vec2_length < ABSOLUTE_TOLERANCE:
        return 0

    vec1 = vec1 / vec1_length
    vec2 = vec2 / vec2_length

    angle = np.arccos(np.dot(vec1, vec2))

    if signed:
        if np.cross(vec1, vec2) < 0:
            angle = -angle

    return angle


def get_total_fiber_twist_2d(
    trace_2d: np.ndarray, signed: bool = False, tolerance: float = ABSOLUTE_TOLERANCE
) -> float:
    """
    Calculate the total twist for 2D traces.

    The 2D twist is defined as the sum of (signed) angles between consecutive
    vectors in the 2D projection along the compression axis.

    Parameters
    ----------
    trace_2d
        Array containing the x,y positions of the polymer trace.
    signed
        True to calculate signed total twist, False otherwise.
    tolerance
        Tolerance for vector length.

    Returns
    -------
    :
        Sum of angles between trace vectors.
    """
    prev_vec = None
    angles = np.zeros(len(trace_2d))
    for i in range(len(trace_2d)):
        if prev_vec is None:
            prev_vec_length = np.linalg.norm(trace_2d[i])
            if prev_vec_length < tolerance:
                prev_vec = None
                continue
            prev_vec = trace_2d[i] / prev_vec_length

        curr_vec_length = np.linalg.norm(trace_2d[i])
        if curr_vec_length < tolerance:
            continue
        curr_vec = trace_2d[i] / curr_vec_length

        angles[i] = get_angle_between_vectors(prev_vec, curr_vec, signed=signed)

        prev_vec = curr_vec

    return np.abs(np.nansum(angles) / 2 / np.pi)


def fit_pca_to_polymer_trace(polymer_trace: np.ndarray) -> PCA:
    """
    Fits PCA to the polymer trace and returns the fitted object.

    Parameters
    ----------
    polymer_trace
        Array containing the x,y,z positions of the polymer trace.

    Returns
    -------
    :
        PCA object fit to the polymer trace.
    """
    pca = PCA(n_components=3)
    pca.fit(polymer_trace)
    return pca


def get_third_component_variance(
    polymer_trace: np.ndarray, **options: Dict[str, Any]
) -> float:
    """
    Calculate the 3rd PCA component given the x,y,z positions of a fiber.

    This component reflects non-coplanarity/out-of-planeness of the fiber.

    Parameters
    ----------
    polymer_trace
        Array containing the x,y,z positions of the polymer trace.
    **options
        Additional options as key-value pairs. Unused.

    Returns
    -------
    :
        Variance explained by the third principal component.
    """
    pca = fit_pca_to_polymer_trace(polymer_trace=polymer_trace)
    return pca.explained_variance_ratio_[2]


def get_sum_bending_energy(
    fiber_energy: np.ndarray, **options: Dict[str, Any]
) -> float:
    """
    Calculate the sum of bending energy from the given fiber energy array.

    Parameters
    ----------
    fiber_energy
        Array containing fiber energy values.
    **options
        Additional options as key-value pairs. Unused.

    Returns
    -------
    :
        Sum of bending energy.
    """
    return fiber_energy[3].sum()


def get_compression_ratio(
    polymer_trace: np.ndarray, **options: Dict[str, Any]
) -> float:
    """
    Calculate the compression ratio of a polymer trace.

    The compression ratio is defined as 1 minus the ratio of the length of the
    end-to-end vector to the contour length of the polymer trace.

    Parameters
    ----------
    polymer_trace
        Array containing the x,y,z positions of the polymer trace.
    **options
        Additional options as key-value pairs. Unused.

    Returns
    -------
    :
        The compression ratio of the polymer trace.
    """
    end_to_end_axis_length = np.linalg.norm(polymer_trace[-1] - polymer_trace[0]).item()
    return 1 - end_to_end_axis_length / get_contour_length_from_trace(polymer_trace)
