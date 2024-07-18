"""
Methods to obtain labels and calculate metrics from the
CompressionMetric Enum class.
"""

from enum import Enum
from typing import Any, Callable, Dict, Union

import numpy as np

from subcell_pipeline.analysis.compression_metrics.polymer_trace import (
    get_asymmetry_of_peak,
    get_average_distance_from_end_to_end_axis,
    get_bending_energy_from_trace,
    get_compression_ratio,
    get_contour_length_from_trace,
    get_sum_bending_energy,
    get_third_component_variance,
    get_total_fiber_twist,
    get_twist_angle,
)


class CompressionMetric(Enum):
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
    TWIST_ANGLE = "twist_angle"

    def label(self: Enum) -> str:
        """
        Return the label for the compression metric.

        Parameters
        ----------
        self
            the CompressionMetric object

        Returns
        -------
        :
            The label for the compression metric.
        """
        labels = {
            CompressionMetric.NON_COPLANARITY.value: "Non-coplanarity",
            CompressionMetric.PEAK_ASYMMETRY.value: "Peak Asymmetry",
            CompressionMetric.SUM_BENDING_ENERGY.value: "Sum Bending Energy",
            CompressionMetric.AVERAGE_PERP_DISTANCE.value: (
                "Average Perpendicular Distance"
            ),
            CompressionMetric.TOTAL_FIBER_TWIST.value: "Fiber Twist",
            CompressionMetric.CALC_BENDING_ENERGY.value: "Calculated Bending Energy",
            CompressionMetric.CONTOUR_LENGTH.value: "Contour Length",
            CompressionMetric.COMPRESSION_RATIO.value: "Compression Ratio",
            CompressionMetric.TWIST_ANGLE.value: "Twist Angle",
        }
        return labels.get(self.value, "")

    def description(self: Enum) -> str:
        """
        Return the description for the compression metric.

        Parameters
        ----------
        self
            the CompressionMetric object

        Returns
        -------
        :
            The description (and units) for the compression metric.
        """
        units = {
            CompressionMetric.NON_COPLANARITY.value: "3rd component variance from PCA",
            CompressionMetric.PEAK_ASYMMETRY.value: "normalized peak distance",
            CompressionMetric.SUM_BENDING_ENERGY.value: "sum of bending energy",
            CompressionMetric.AVERAGE_PERP_DISTANCE.value: "distance (nm)",
            CompressionMetric.TOTAL_FIBER_TWIST.value: "total fiber twist",
            CompressionMetric.CALC_BENDING_ENERGY.value: "energy",
            CompressionMetric.CONTOUR_LENGTH.value: "filament contour length (nm)",
            CompressionMetric.COMPRESSION_RATIO.value: "compression ratio",
            CompressionMetric.TWIST_ANGLE.value: (
                "difference between initial and final tangent (degrees)"
            ),
        }
        return units.get(self.value, "")

    def bounds(self: Enum) -> tuple[float, float]:
        """
        Return the default bounds for the compression metric.

        Parameters
        ----------
        self
            the CompressionMetric object

        Returns
        -------
        :
            The default bounds for the compression metric.
        """
        bounds = {
            CompressionMetric.NON_COPLANARITY.value: (0, 0.03),
            CompressionMetric.PEAK_ASYMMETRY.value: (0, 0.5),
            CompressionMetric.SUM_BENDING_ENERGY.value: (0, 0),  # TODO
            CompressionMetric.AVERAGE_PERP_DISTANCE.value: (0, 85.0),
            CompressionMetric.TOTAL_FIBER_TWIST.value: (0, 0),  # TODO
            CompressionMetric.CALC_BENDING_ENERGY.value: (0, 10),
            CompressionMetric.CONTOUR_LENGTH.value: (480, 505),
            CompressionMetric.COMPRESSION_RATIO.value: (0, 1),  # TODO
            CompressionMetric.TWIST_ANGLE.value: (-180, 180),
        }
        return bounds.get(self.value, (0, 0))

    def calculate_metric(
        self, polymer_trace: np.ndarray, **options: dict[str, Any]
    ) -> Union[float, np.floating[Any]]:
        """
        Calculate the compression metric for the given polymer trace.

        Parameters
        ----------
        self
            the CompressionMetric object

        polymer_trace
            array containing the x,y,z positions of the polymer trace

        **options
            Additional options as key-value pairs.

        Returns
        -------
        :
            The calculated compression metric for the polymer
        """
        functions: Dict[CompressionMetric, Callable] = {
            CompressionMetric.NON_COPLANARITY: get_third_component_variance,
            CompressionMetric.PEAK_ASYMMETRY: get_asymmetry_of_peak,
            CompressionMetric.SUM_BENDING_ENERGY: get_sum_bending_energy,
            CompressionMetric.AVERAGE_PERP_DISTANCE: (
                get_average_distance_from_end_to_end_axis
            ),
            CompressionMetric.TOTAL_FIBER_TWIST: get_total_fiber_twist,
            CompressionMetric.CALC_BENDING_ENERGY: get_bending_energy_from_trace,
            CompressionMetric.CONTOUR_LENGTH: get_contour_length_from_trace,
            CompressionMetric.COMPRESSION_RATIO: get_compression_ratio,
            CompressionMetric.TWIST_ANGLE: get_twist_angle,
        }
        return functions[self](polymer_trace, **options)
