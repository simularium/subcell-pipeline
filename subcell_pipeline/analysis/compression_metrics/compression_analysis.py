from enum import Enum
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from io_collection.keys.check_key import check_key
from io_collection.load.load_dataframe import load_dataframe
from io_collection.save.save_dataframe import save_dataframe
from sklearn.decomposition import PCA

from subcell_pipeline.analysis.compression_metrics.constants import ABSOLUTE_TOLERANCE
from subcell_pipeline.analysis.compression_metrics.vectors import (
    get_end_to_end_unit_vector,
    get_unit_vector,
)


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
        """
        Return the label for the compression metric.

        Parameters
        ----------
        self
            the COMPRESSIONMETRIC object

        Returns
        -------
        :
            The label for the compression metric.
        """
        labels = {
            COMPRESSIONMETRIC.NON_COPLANARITY.value: "Non-coplanarity",
            COMPRESSIONMETRIC.PEAK_ASYMMETRY.value: "Peak Asymmetry",
            COMPRESSIONMETRIC.SUM_BENDING_ENERGY.value: "Sum Bending Energy",
            COMPRESSIONMETRIC.AVERAGE_PERP_DISTANCE.value: (
                "Average Perpendicular Distance"
            ),
            COMPRESSIONMETRIC.TOTAL_FIBER_TWIST.value: "Total Fiber Twist",
            COMPRESSIONMETRIC.CALC_BENDING_ENERGY.value: "Calculated Bending Energy",
            COMPRESSIONMETRIC.CONTOUR_LENGTH.value: "Contour Length",
            COMPRESSIONMETRIC.COMPRESSION_RATIO.value: "Compression Ratio",
        }
        return labels.get(self.value, "")

    def calculate_metric(self, polymer_trace: np.ndarray, **options: dict):
        """
        Calculate the compression metric for the given polymer trace.

        Parameters
        ----------
        self
            the COMPRESSIONMETRIC object

        polymer_trace
            array containing the x,y,z positions of the polymer trace

        **options
            Additional options as key-value pairs.

        Returns
        -------
        :
            The calculated compression metric for the polymer
        """
        functions = {
            COMPRESSIONMETRIC.NON_COPLANARITY: get_third_component_variance,
            COMPRESSIONMETRIC.PEAK_ASYMMETRY: get_asymmetry_of_peak,
            COMPRESSIONMETRIC.SUM_BENDING_ENERGY: get_sum_bending_energy,
            COMPRESSIONMETRIC.AVERAGE_PERP_DISTANCE: (
                get_average_distance_from_end_to_end_axis
            ),
            COMPRESSIONMETRIC.TOTAL_FIBER_TWIST: get_total_fiber_twist,
            COMPRESSIONMETRIC.CALC_BENDING_ENERGY: get_bending_energy_from_trace,
            COMPRESSIONMETRIC.CONTOUR_LENGTH: get_contour_length_from_trace,
            COMPRESSIONMETRIC.COMPRESSION_RATIO: get_compression_ratio,
        }
        return functions[self](polymer_trace, **options)


def get_end_to_end_axis_distances_and_projections(
    polymer_trace: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the distances of the polymer trace points from the end-to-end axis.
    Here, the end-to-end axis is defined as the line joining the first and last
    fiber point.

    Parameters
    ----------
    polymer_trace
        array containing the x,y,z positions of the polymer trace points
        at a given time

    Returns
    -------
    perp_distances
        perpendicular distances of the polymer trace from the end-to-end axis

    scaled_projections
        length of fiber point projections along the end-to-end axis, scaled
        by axis length.
        Can be negative.

    projection_positions
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
    Calculate the average perpendicular distance of polymer trace points from
    the end-to-end axis.

    Parameters
    ----------
    polymer_trace
        array containing the x,y,z positions of the polymer trace
        at a given time

    **options
        Additional options as key-value pairs.

    Returns
    -------
    :
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
    Calculate the scaled distance of the projection of the peak from the
    end-to-end axis midpoint.

    Parameters
    ----------
    polymer_trace
        array containing the x,y,z positions of the polymer trace
        at a given time

    **options
        Additional options as key-value pairs.

    Returns
    -------
    :
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
    Calculate the PCA projection of the polymer trace.

    Parameters
    ----------
    polymer_trace
        array containing the x,y,z positions of the polymer trace

    Returns
    -------
    pca_projection
        PCA projection of the polymer trace
    """
    pca = fit_pca_to_polymer_trace(polymer_trace=polymer_trace)
    return pca.transform(polymer_trace)


def get_contour_length_from_trace(
    polymer_trace: np.ndarray,
    **options: dict,
) -> Union[float, np.floating[Any]]:
    """
    Calculate the sum of inter-monomer distances in the trace.

    Parameters
    ----------
    polymer_trace
        n x 3 array containing the x,y,z positions of the polymer trace

    **options
        Additional options as key-value pairs.

    Returns
    -------
    :
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
    Calculate the bending energy per monomer of a polymer trace.

    Parameters
    ----------
    polymer_trace
        array containing the x,y,z positions of the polymer trace

    **options
        Additional options as key-value pairs.
        bending_constant: float
            bending constant of the fiber

    Returns
    -------
    :
        bending energy per monomer of the polymer trace
    """
    bending_constant = options.get("bending_constant", 1.0)

    cos_angle = np.zeros(len(polymer_trace) - 2)
    for ind in range(len(polymer_trace) - 2):
        vec1 = polymer_trace[ind + 1] - polymer_trace[ind]
        vec2 = polymer_trace[ind + 2] - polymer_trace[ind + 1]

        cos_angle[ind] = (
            np.dot(vec1, vec2) / np.linalg.norm(vec1) / np.linalg.norm(vec2)
        )

    energy = bending_constant * (1 - np.nanmean(cos_angle))

    return energy


def get_total_fiber_twist(
    polymer_trace: np.ndarray,
    **options: dict,
) -> float:
    """
    Calculate the total twist using projections of the polymer trace
    in the 2nd and 3rd dimension.

    Parameters
    ----------
    polymer_trace
        array containing the x,y,z positions of the polymer trace

    **options: dict
        Additional options as key-value pairs:

        compression_axis: int
            axis along which the polymer trace is compressed
        signed: bool
            whether to return the signed or unsigned total twist
        tolerance: float
            ABSOLUTE_TOLERANCE

    Returns
    -------
    :
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
    Calculate the total twist using PCA projections of the polymer trace
    in the 2nd and 3rd dimension.

    Parameters
    ----------
    polymer_trace
        array containing the x,y,z positions of the polymer trace

    tolerance
        ABSOLUTE_TOLERANCE

    Returns
    -------
    :
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
    Calculate the signed angle between two vectors.

    Parameters
    ----------
    vec1
        The first vector

    vec2
        The second vector

    signed
        if True, returns the signed angle between vec1 and vec2
        Default is False

    Returns
    -------
    :
        signed angle between vec1 and vec2
    """
    vec1, vec1_length = get_unit_vector(vec1)
    vec2, vec2_length = get_unit_vector(vec2)

    if vec1_length < ABSOLUTE_TOLERANCE or vec2_length < ABSOLUTE_TOLERANCE:
        return 0

    angle = np.arccos(np.dot(vec1, vec2))

    if signed:
        if np.cross(vec1, vec2) < 0:
            angle = -angle

    return angle


def get_total_fiber_twist_2d(
    trace_2d: np.ndarray,
    signed: bool = False,
    tolerance: float = ABSOLUTE_TOLERANCE,
) -> float:
    """
    Calculate the total twist for 2d traces. The 2D twist is defined as the sum of
    (signed) angles between consecutive vectors in the 2D projection along the
    compression axis.

    Parameters
    ----------
    trace_2d
        array containing the x,y positions of the polymer trace

    signed
        if True, returns the signed total twist
        Default is False

    tolerance
        Tolerance for vector length

    Returns
    -------
    :
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


def fit_pca_to_polymer_trace(
    polymer_trace: np.ndarray,
) -> PCA:
    """
    Fits PCA to the polymer trace and returns the fitted object.

    Parameters
    ----------
    polymer_trace
        array containing the x,y,z positions of the polymer trace

    Returns
    -------
    :
        PCA object fitted to the polymer trace
    """
    pca = PCA(n_components=3)
    pca.fit(polymer_trace)
    return pca


def get_third_component_variance(
    polymer_trace: np.ndarray,
    **options: dict,
) -> float:
    """
    Calculate the third PCA component given the x,y,z positions of a fiber at
    a given time. This component reflects non-coplanarity/out of planeness.

    Parameters
    ----------
    polymer_trace
        array containing the x,y,z positions of the polymer trace

    **options: dict
        Additional options as key-value pairs.

    Returns
    -------
    :
        noncoplanarity of fiber
        defined as the explained variance ratio of the third PCA component
    """
    pca = fit_pca_to_polymer_trace(polymer_trace=polymer_trace)
    return pca.explained_variance_ratio_[2]


def get_sum_bending_energy(
    fiber_energy: np.ndarray,
    **options: dict,
) -> float:
    """
    Calculate the sum of bending energy from the given fiber energy array.

    Parameters
    ----------
    fiber_energy
        Array containing fiber energy values.

    options
        Additional options for calculation.

    Returns
    -------
    :
        Sum of bending energy.

    """
    return fiber_energy[3].sum()


def get_compression_ratio(
    polymer_trace: np.ndarray,
    **options: dict,
) -> Union[float, np.floating[Any]]:
    """
    Calculate the compression ratio of a polymer trace.

    The compression ratio is defined as 1 minus the ratio of the length of
    the end-to-end vector to the contour length of the polymer trace.

    Parameters
    ----------
    polymer_trace: np.ndarray
        The polymer trace as a numpy array.

    **options: dict
        Additional options for the calculation.

    Returns
    -------
    :
        The compression ratio of the polymer trace.
    """
    return 1 - get_end_to_end_unit_vector(polymer_trace)[
        1
    ] / get_contour_length_from_trace(polymer_trace)


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

    return df_metrics.reset_index().rename(columns={"index": "time"})


def get_compression_metric_data(
    bucket: str,
    series_name: str,
    condition_keys: list[str],
    random_seeds: list[int],
    metrics: list[COMPRESSIONMETRIC],
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
