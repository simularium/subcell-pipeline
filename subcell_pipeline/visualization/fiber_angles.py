from typing import Optional, Tuple, Union

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from subcell_pipeline.analysis.compression_metrics.constants import SIMULATOR_COLOR_MAP
from subcell_pipeline.analysis.compression_metrics.polymer_trace import (
    get_2d_polymer_trace,
)

plt.rcParams.update({"font.size": 14})


def plot_initial_and_final_tangents(
    polymer_trace: np.ndarray,
    compression_axis: int,
    ax: Optional[matplotlib.axes.Axes] = None,
    color: str = "r",
    scale: int = 1,
) -> Tuple[Union[matplotlib.figure.Figure, None], Union[matplotlib.axes.Axes, None]]:
    """
    Plot the normalized tangent vectors along the fiber trace.

    Parameters
    ----------
    polymer_trace
        N x 3 array of fiber coordinates.

    compression_axis
        The axis along which to compress the fibers.

    ax
        The matplotlib axes object to plot on.

    color
        The color of the tangent vectors

    scale
        The scaling factor for the tangent

    Returns
    -------
    :
        None
    """
    if ax is None:
        fig, ax = plt.subplots(dpi=300)
    else:
        fig = ax.get_figure()

    arrowprops = {"arrowstyle": "->", "color": color, "lw": 1}

    trace_2d = get_2d_polymer_trace(polymer_trace, compression_axis)
    trace_2d_norm = trace_2d / np.linalg.norm(trace_2d, axis=1)[:, np.newaxis]

    ax.annotate(
        "",
        xy=trace_2d_norm[1] * scale,
        xytext=trace_2d_norm[0],
        arrowprops=arrowprops,
    )

    ax.annotate(
        "",
        xy=trace_2d_norm[-1] * scale,
        xytext=trace_2d_norm[-2],
        arrowprops=arrowprops,
    )

    ax.plot(trace_2d[:, 0], trace_2d[:, 1], color=color)

    ax.set_ylabel("Z")
    ax.set_xlabel("Y")
    plt.tight_layout()

    return fig, ax


def visualize_tangent_angles(
    merged_df: pd.DataFrame,
    compression_axis: int = 0,
) -> None:
    """
    Visualize tangent angles for each fiber in the merged dataframe
    at the last timepoint.

    Parameters
    ----------
    merged_df
        The merged dataframe containing the fiber data.

    compression_axis
        The axis along which to compress the fibers.

    Returns
    -------
    :
        None
    """
    _, ax = plt.subplots(dpi=300)
    for simulator, df_simulator in merged_df.groupby("simulator"):
        color = SIMULATOR_COLOR_MAP[str(simulator)]
        for _, df_condition in df_simulator.groupby("key"):
            for _, df_seed in df_condition.groupby("seed"):
                df_fiber = df_seed[df_seed["time"] == df_seed["time"].max()]
                polymer_trace = df_fiber[["xpos", "ypos", "zpos"]].values
                _, ax = plot_initial_and_final_tangents(
                    polymer_trace=polymer_trace,
                    compression_axis=compression_axis,
                    ax=ax,
                    color=color,
                )

    ax.set_aspect("equal")
    plt.show()
