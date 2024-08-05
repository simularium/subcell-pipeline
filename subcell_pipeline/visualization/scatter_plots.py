"""Methods for scatter plot visualization."""

from typing import Optional

import numpy as np
from simulariumio import ScatterPlotData

from subcell_pipeline.analysis.compression_metrics.compression_metric import (
    CompressionMetric,
)


def make_empty_scatter_plots(
    metrics: list[CompressionMetric],
    total_steps: int = -1,
    times: Optional[np.ndarray] = None,
    time_units: Optional[str] = None,
) -> dict[CompressionMetric, ScatterPlotData]:
    """
    Create empty scatter plot placeholders for list of metrics.

    Parameters
    ----------
    metrics
        List of metrics.
    total_steps
        Total number of timesteps. Required if times is not given.
    times
        List of timepoints. Required if total_steps is not given.
    time_units
        Time units. Used only with times.

    Returns
    -------
    :
        Map of metric to empty scatter plot placeholder.
    """

    if total_steps < 0 and times is None:
        raise Exception("Either total_steps or times array is required for plots")
    elif times is None:
        # use normalized time
        xlabel = "T (normalized)"
        xtrace = (1 / float(total_steps)) * np.arange(total_steps)
    else:
        # use actual time
        xlabel = f"T ({time_units})"
        xtrace = times
        total_steps = times.shape[0]

    plots = {}

    for metric in metrics:
        lower_bound, upper_bound = metric.bounds()
        plots[metric] = ScatterPlotData(
            title=metric.label(),
            xaxis_title=xlabel,
            yaxis_title=metric.description(),
            xtrace=xtrace,
            ytraces={
                "<<<": lower_bound * np.ones(total_steps),
                ">>>": upper_bound * np.ones(total_steps),
            },
            render_mode="lines",
        )

    return plots
