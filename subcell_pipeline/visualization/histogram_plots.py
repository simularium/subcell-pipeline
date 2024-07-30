"""Methods for histogram plot visualization."""

from simulariumio import HistogramPlotData, ScatterPlotData

from subcell_pipeline.analysis.compression_metrics.compression_metric import (
    CompressionMetric,
)


def make_empty_histogram_plots(
    metrics: list[CompressionMetric],
) -> dict[CompressionMetric, ScatterPlotData]:
    """
    Create empty histogram plot placeholders for list of metrics.

    Parameters
    ----------
    metrics
        List of metrics.

    Returns
    -------
    :
        Map of metric to empty histogram plot placeholder.
    """

    plots = {}

    for metric in metrics:
        plots[metric] = HistogramPlotData(
            title=metric.label(),
            xaxis_title=metric.description(),
            traces={},
        )

    return plots
