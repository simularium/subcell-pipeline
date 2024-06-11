from enum import Enum
from typing import Any, Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pacmap import PaCMAP
from sklearn.decomposition import PCA

ABSOLUTE_TOLERANCE = 1e-6


class COMPRESSIONMETRIC(Enum):
    # Enum class for compression metrics

    NON_COPLANARITY = "non_coplanarity"
    PEAK_ASYMMETRY = "peak_asymmetry"
    SUM_BENDING_ENERGY = "sum_bending_energy"
    AVERAGE_PERP_DISTANCE = "average_perp_distance"
    TOTAL_FIBER_TWIST = "total_fiber_twist"
    ENERGY_ASYMMETRY = "energy_asymmetry"
    CALC_BENDING_ENERGY = "calc_bending_energy"
    CONTOUR_LENGTH = "contour_length"
    COMPRESSION_RATIO = "compression_ratio"

    def label(self):
        # Returns the label for the compression metric
        labels = {
            COMPRESSIONMETRIC.NON_COPLANARITY: "Non-coplanarity",
            COMPRESSIONMETRIC.PEAK_ASYMMETRY: "Peak asymmetry",
            COMPRESSIONMETRIC.SUM_BENDING_ENERGY: "Bending energy",
            COMPRESSIONMETRIC.AVERAGE_PERP_DISTANCE: "Average perpendicular distance",
            COMPRESSIONMETRIC.TOTAL_FIBER_TWIST: "Total fiber twist",
            COMPRESSIONMETRIC.ENERGY_ASYMMETRY: "Energy asymmetry",
            COMPRESSIONMETRIC.CALC_BENDING_ENERGY: "Calculated bending energy",
            COMPRESSIONMETRIC.CONTOUR_LENGTH: "Contour length",
            COMPRESSIONMETRIC.COMPRESSION_RATIO: "Compression ratio",
        }
        return labels[self]

    def calculate_metric(self, polymer_trace: np.ndarray, **options: dict):
        # Returns the calculated metric value
        functions = {
            COMPRESSIONMETRIC.NON_COPLANARITY: get_third_component_variance,
            COMPRESSIONMETRIC.PEAK_ASYMMETRY: get_asymmetry_of_peak,
            COMPRESSIONMETRIC.SUM_BENDING_ENERGY: get_sum_bending_energy,
            COMPRESSIONMETRIC.AVERAGE_PERP_DISTANCE: get_average_distance_from_end_to_end_axis,
            COMPRESSIONMETRIC.TOTAL_FIBER_TWIST: get_total_fiber_twist,
            COMPRESSIONMETRIC.ENERGY_ASYMMETRY: get_energy_asymmetry,
            COMPRESSIONMETRIC.CALC_BENDING_ENERGY: get_bending_energy_from_trace,
            COMPRESSIONMETRIC.CONTOUR_LENGTH: get_contour_length_from_trace,
            COMPRESSIONMETRIC.COMPRESSION_RATIO: get_compression_ratio,
        }
        return functions[self](polymer_trace, **options)


def get_unit_vector(
    vector: np.array,
) -> Tuple[np.array, Union[float, np.floating[Any]]]:
    """
    Calculates the unit vector and length of a given vector.

    Parameters:
        vector (np.array): The input vector.

    Returns:
        Tuple[np.array, Union[float, np.floating[Any]]]: A tuple containing the unit vector
        and length of the input vector.
    """
    if np.linalg.norm(vector) < ABSOLUTE_TOLERANCE or np.isnan(vector).any():
        return np.array([0, 0, 0]), 0.0
    else:
        vec_length = np.linalg.norm(vector)
        return vector / vec_length, vec_length


def get_end_to_end_unit_vector(
    polymer_trace: np.ndarray,
) -> Tuple[np.array, Union[float, np.floating[Any]]]:
    """
    Returns the unit vector of the end-to-end axis of a polymer trace.

    Parameters
    ----------
    polymer_trace: [n x 3] numpy array
        array containing the x,y,z positions of the polymer trace points

    Returns
    -------
    end_to_end_unit_vector: [3 x 1] numpy array
        unit vector of the end-to-end axis of the polymer trace
    end_to_end_axis_length: float
        length of the end-to-end axis of the polymer trace
    """
    assert len(polymer_trace) > 1, "Polymer trace must have at least 2 points"
    assert polymer_trace.shape[1] == 3, "Polymer trace must have 3 columns"

    end_to_end_axis = polymer_trace[-1] - polymer_trace[0]

    return get_unit_vector(end_to_end_axis)


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
    end_to_end_axis, end_to_end_axis_length = get_end_to_end_unit_vector(
        polymer_trace=polymer_trace
    )

    position_vectors = polymer_trace - polymer_trace[0]
    projections = np.dot(position_vectors, end_to_end_axis)
    scaled_projections = projections / end_to_end_axis_length

    projection_positions = polymer_trace[0] + projections[:, None] * end_to_end_axis

    perp_distances = np.linalg.norm(polymer_trace - projection_positions, axis=1)

    return perp_distances, scaled_projections, projection_positions


def get_average_distance_from_end_to_end_axis(
    polymer_trace: np.ndarray,
    **options: dict,
) -> Union[float, np.floating[Any]]:
    """
    Returns the average perpendicular distance of polymer trace points from
    the end-to-end axis.

    Parameters
    ----------
    polymer_trace: [n x 3] numpy array
        array containing the x,y,z positions of the polymer trace
        at a given time

    **options: dict
        Additional options as key-value pairs.

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
    **options: dict,
) -> float:
    """
    returns the scaled distance of the projection of the peak from the
    end-to-end axis midpoint.

    Parameters
    ----------
    polymer_trace: [n x 3] numpy array
        array containing the x,y,z positions of the polymer trace
        at a given time
    **options: dict
        Additional options as key-value pairs.

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
    if np.all(perp_distances < ABSOLUTE_TOLERANCE):
        return 0

    projection_of_peak = scaled_projections[perp_distances == np.max(perp_distances)]
    peak_asym = np.max(
        np.abs(projection_of_peak - 0.5)
    )  # max kinda handles multiple peaks

    return peak_asym


def get_pca_polymer_trace_projection(
    polymer_trace: np.ndarray,
) -> np.ndarray:
    """
    Returns the PCA projection of the polymer trace.

    Parameters
    ----------
    polymer_trace: [n x 3] numpy array
        array containing the x,y,z positions of the polymer trace

    Returns
    -------
    pca_projection: [n x 3] numpy array
        PCA projection of the polymer trace
    """
    pca = fit_pca_to_polymer_trace(polymer_trace=polymer_trace)
    return pca.transform(polymer_trace)


def get_contour_length_from_trace(
    polymer_trace: np.ndarray,
    **options: dict,
) -> Union[float, np.floating[Any]]:
    """
    Returns the sum of inter-monomer distances in the trace.

    Parameters
    ----------
    polymer_trace: [n x 3] numpy array
        array containing the x,y,z positions of the polymer trace
    **options: dict
        Additional options as key-value pairs.

    Returns
    -------
    total_distance: float
        sum of inter-monomer distances in the trace
    """
    total_distance = 0
    for i in range(len(polymer_trace) - 1):
        total_distance += np.linalg.norm(polymer_trace[i] - polymer_trace[i + 1])
    return total_distance


def get_bending_energy_from_trace(
    polymer_trace: np.ndarray,
    **options: dict[str, Any],
) -> Union[float, np.floating[Any]]:
    """
    Returns the bending energy per monomer of a polymer trace.

    Parameters
    ----------
    polymer_trace: [n x 3] numpy array
        array containing the x,y,z positions of the polymer trace
    **options: dict
        Additional options as key-value pairs.
        bending_constant: float
            bending constant of the fiber
    """
    bending_constant = float(options.get("bending_constant", 1))

    cos_angle = np.zeros(len(polymer_trace) - 2)
    for ind in range(len(polymer_trace) - 2):
        vec1 = polymer_trace[ind + 1] - polymer_trace[ind]
        vec2 = polymer_trace[ind + 2] - polymer_trace[ind + 1]

        cos_angle[ind] = (
            np.dot(vec1, vec2) / np.linalg.norm(vec1) / np.linalg.norm(vec2)
        )

    energy = bending_constant * np.nanmean(1 - cos_angle)

    return energy


def get_total_fiber_twist(
    polymer_trace: np.ndarray,
    **options: dict,
) -> float:
    """
    Calculates the total twist using projections of the polymer trace
    in the 2nd and 3rd dimension.

    Parameters
    ----------
    polymer_trace: [n x 3] numpy array
        array containing the x,y,z positions of the polymer trace
        at a given time
    **options: dict
        Additional options as key-value pairs:

        compression_axis: int
            axis along which the polymer trace is compressed
        signed: bool
            whether to return the signed or unsigned total twist
        tolerance: float
            ABSOLUTE_TOLERANCE
    Returns
    ----------
    total_twist: float
        sum of angles between PCA projection vectors
    """
    compression_axis = options.get("compression_axis", 0)
    signed = options.get("signed", True)
    tolerance = options.get("tolerance", ABSOLUTE_TOLERANCE)

    trace_2d = polymer_trace[
        :, [ax for ax in range(polymer_trace.shape[1]) if ax != compression_axis]
    ]
    trace_2d = trace_2d - np.mean(trace_2d, axis=0)

    return get_total_fiber_twist_2d(
        trace_2d, signed=signed, tolerance=tolerance  # type: ignore
    )


def get_total_fiber_twist_pca(
    polymer_trace: np.ndarray,
    tolerance: float = ABSOLUTE_TOLERANCE,
) -> float:
    """
    Calculates the total twist using PCA projections of the polymer trace
    in the 2nd and 3rd dimension.

    Parameters
    ----------
    polymer_trace: [n x 3] numpy array
        array containing the x,y,z positions of the polymer trace
        at a given time
    tolerance: float
        ABSOLUTE_TOLERANCE
    Returns
    ----------
    total_twist: float
        sum of angles between PCA projection vectors
    """
    pca_trace = get_pca_polymer_trace_projection(polymer_trace=polymer_trace)
    pca_trace_2d = pca_trace[:, 1:]

    return get_total_fiber_twist_2d(pca_trace_2d, tolerance=tolerance)


def get_angle_between_vectors(
    vec1: np.ndarray,
    vec2: np.ndarray,
    signed: bool = False,
) -> float:
    """
    Returns the signed angle between two vectors.

    Parameters
    ----------
    vec1: [2 x 1] numpy array
        vector 1
    vec2: [2 x 1] numpy array
        vector 2
    signed: bool
        if True, returns the signed angle between vec1 and vec2

    Returns
    -------
    signed_angle: float
        signed angle between vec1 and vec2
    """
    signed_angle = np.arctan2(vec1[1], vec1[0]) - np.arctan2(vec2[1], vec2[0])

    # normalize to [-pi, pi]
    if signed_angle > np.pi:
        angle = signed_angle - 2 * np.pi
    elif signed_angle < -np.pi:
        angle = signed_angle + 2 * np.pi
    else:
        angle = signed_angle

    if not signed:
        angle = np.abs(signed_angle)

    return angle


def get_total_fiber_twist_2d(
    trace_2d: np.ndarray,
    signed: bool = False,
    tolerance: float = ABSOLUTE_TOLERANCE,
) -> float:
    """
    Calculates the total twist for 2d traces.

    Parameters
    ----------
    trace_2d: [n x 2] numpy array
        array containing the x,y positions of the polymer trace
    signed: bool
        if True, returns the signed total twist
    tolerance: float
        ABSOLUTE_TOLERANCE
    Returns
    ----------
    total_twist: float
        sum of angles between trace vectors
    """
    prev_vec = None
    angles = np.zeros(len(trace_2d))
    for i in range(len(trace_2d)):
        if prev_vec is None:
            prev_vec, prev_vec_length = get_unit_vector(trace_2d[i])
            if prev_vec_length < tolerance:
                prev_vec = None
            continue

        curr_vec, curr_vec_length = get_unit_vector(trace_2d[i])
        if curr_vec_length < tolerance:
            continue

        angles[i] = get_angle_between_vectors(prev_vec, curr_vec, signed=signed)

        prev_vec = curr_vec

    return np.abs(np.nansum(angles) / 2 / np.pi)


def get_total_fiber_twist_bak(
    polymer_trace: np.ndarray,
    tolerance: float = ABSOLUTE_TOLERANCE,
) -> float:
    """
    Returns the sum of angles between consecutive vectors from the
    polymer trace points to the end-to-end axis.

    Parameters
    ----------
    polymer_trace: [n x 3] numpy array
        array containing the x,y,z positions of the polymer trace
        at a given time
    tolerance: float
        ABSOLUTE_TOLERANCE

    Returns
    -------
    total_twist: float
        sum of angles between vectors from trace points to axis
        in degrees
    """
    (
        perp_distances,
        _,
        projection_positions,
    ) = get_end_to_end_axis_distances_and_projections(polymer_trace=polymer_trace)

    # if all perpendicular distances are zero, return 0
    if np.all(perp_distances < tolerance):
        return 0

    perp_vectors = polymer_trace - projection_positions

    twist_angle = 0

    prev_vec, prev_vec_length = get_unit_vector(perp_vectors[1])

    for i in range(2, len(perp_vectors) - 1):
        curr_vec, curr_vec_length = get_unit_vector(perp_vectors[i])

        if prev_vec_length < tolerance:
            prev_vec = curr_vec
            prev_vec_length = curr_vec_length
            continue

        if curr_vec_length < tolerance:
            continue

        dot_product = np.dot(prev_vec, curr_vec)

        if np.isnan(dot_product) or np.abs(dot_product) > 1:
            continue

        # print(prev_vec_length, curr_vec_length, dot_product, twist_angle)

        curr_angle = np.arccos(dot_product)

        if ~np.isnan(curr_angle):
            twist_angle += curr_angle
            prev_vec = curr_vec
            prev_vec_length = curr_vec_length

    total_twist = twist_angle / np.pi

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


def fit_pca_to_polymer_trace(
    polymer_trace: np.ndarray,
) -> PCA:
    """
    Returns the pca object fit to the polymer trace.

    Parameters
    ----------
    polymer_trace: [n x 3] numpy array
        array containing the x,y,z positions of the polymer trace

    Returns
    -------
    pca: PCA object
    """
    pca = PCA(n_components=3)
    pca.fit(polymer_trace)
    return pca


def get_third_component_variance(
    polymer_trace: np.ndarray,
    **options: dict,
) -> float:
    """
    Returns the third PCA component given the x,y,z positions of a fiber at
    a given time. This component reflects non-coplanarity/out of planeness.

    Parameters
    ----------
    polymer_trace: [n x 3] numpy array
        array containing the x,y,z positions of the polymer trace
        at a given time
    **options: dict
        Additional options as key-value pairs.

    Returns
    -------
    third_component_variance: float
        noncoplanarity of fiber
    """
    pca = fit_pca_to_polymer_trace(polymer_trace=polymer_trace)
    return pca.explained_variance_ratio_[2]


def get_energy_asymmetry(
    fiber_energy: np.ndarray,
    **options: dict,
) -> float:
    """
    Returns the sum bending energy given a single fiber x,y,z positions
    and segment energy values.

    Parameters
    ----------
    fiber_energy: [n x 4] numpy array
        array containing the x,y,z positions of the polymer trace and segment energy
        at a given time
    **options: dict
        Additional options as key-value pairs.

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
    **options: dict,
) -> float:
    return fiber_energy[3].sum()


def get_compression_ratio(
    polymer_trace: np.ndarray,
    **options: dict,
) -> float:
    return 1 - get_end_to_end_unit_vector(polymer_trace)[
        1
    ] / get_contour_length_from_trace(polymer_trace)


def run_single_metric_calculation(
    df_repeat: pd.DataFrame, metric: COMPRESSIONMETRIC, **options: dict
) -> pd.DataFrame:
    """
    Given cytosim output, run_metric_calculation calculates a chosen metric over
    all points in a fiber.

    Parameters
    ----------
    df_repeat: [(num_timepoints * num_points) x n columns] pandas dataframe
        df_repeat is a dataframe of cytosim outputs that is generated after
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
    df_repeat dataframe with calculated metric appended
    """
    df_repeat[metric.value] = np.nan
    for _ct, (_time, fiber_at_time) in enumerate(df_repeat.groupby("time")):
        polymer_trace = fiber_at_time[["xpos", "ypos", "zpos"]].values
        df_repeat.loc[fiber_at_time.index, metric.value] = metric.calculate_metric(
            polymer_trace=polymer_trace, **options
        )

    return df_repeat


def compression_metrics_workflow(
    df_repeat: pd.DataFrame, metrics_to_calculate: list, **options: dict
) -> pd.DataFrame:
    """
    Calculates chosen metrics from cytosim output of fiber positions and
    properties across timesteps.

    Parameters
    ----------
    df_repeat: [(num_timepoints * num_points) x n columns] pandas dataframe
        df_repeat is a dataframe of cytosim outputs that is generated
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
    df_repeat dataframe with chosen metrics appended as columns

    """
    for metric in metrics_to_calculate:
        df_repeat = run_single_metric_calculation(df_repeat, metric, **options)
    return df_repeat


def plot_metric(df_repeat: pd.DataFrame, metric: COMPRESSIONMETRIC) -> None:
    """
    Plots and saves metric values over time.
    gi
    Parameters
    ----------
    df_repeat: [(num_timepoints * num_points) x n columns] pandas dataframe
        includes [fiber_id, x_pos, y_pos, z_pos, xforce, yforce, zforce,
        segment_curvature,
        force_magnitude, segment_energy] columns and any metric columns
    metric: metric name to be plotted
        chosen COMPRESSIONMETRIC.

    """
    metric_by_time = df_repeat.groupby(["time"])[metric].mean()
    plt.plot(metric_by_time)
    plt.xlabel("Time")
    plt.ylabel(metric.label())
    # Save files if needed.
    # plt.savefig(str(metric) + "-time.pdf")
    # plt.savefig(str(metric) + "-time.png")


def plot_metric_list(df_repeat: pd.DataFrame, metrics: list) -> None:
    # docs
    for metric in metrics:
        plot_metric(df_repeat, metric)


def calculate_compression_metrics(
    df: pd.DataFrame, metrics: List[Any], **options: Dict[str, Any]
) -> pd.DataFrame:
    """
    Calculate compression metrics for each group in the given DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        metrics (List[Any]): The list of metrics to calculate.
        **options (Dict[str, Any]): Additional options for the calculation.

    Returns:
        pd.DataFrame: The DataFrame with the calculated metrics.
    """
    for simulator, df_sim in df.groupby("simulator"):
        for velocity, df_velocity in df_sim.groupby("velocity"):
            for repeat, df_repeat in df_velocity.groupby("repeat"):
                print(f"simulator: {simulator}, velocity: {velocity}, repeat: {repeat}")
                df_repeat = compression_metrics_workflow(
                    df_repeat, metrics_to_calculate=metrics, **options
                )
                for metric in metrics:
                    df.loc[df_repeat.index, metric.value] = df_repeat[metric.value]
    return df
