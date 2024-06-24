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
            CompressionMetric.TOTAL_FIBER_TWIST.value: "Total Fiber Twist",
            CompressionMetric.CALC_BENDING_ENERGY.value: "Calculated Bending Energy",
            CompressionMetric.CONTOUR_LENGTH.value: "Contour Length",
            CompressionMetric.COMPRESSION_RATIO.value: "Compression Ratio",
        }
        return labels.get(self.value, "")

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
        }
        return functions[self](polymer_trace, **options)
