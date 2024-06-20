from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from subcell_pipeline.analysis.compression_metrics.compression_analysis import (
    COMPRESSIONMETRIC,
)
from subcell_pipeline.analysis.compression_metrics.constants import SIMULATOR_COLOR_MAP


def plot_metrics_vs_time(
    df: pd.DataFrame,
    metrics: List[COMPRESSIONMETRIC],
    figure_path: Union[Path, None] = None,
    suffix: str = "",
    compression_distance: float = 150.0,
    use_real_time: bool = False,
) -> None:
    """
    Plot metrics vs time.

    Parameters
    ----------
    df
        The input DataFrame.

    metrics
        The list of metrics to plot.

    figure_path
        The path to save the figure.

    suffix
        The suffix to append to the figure filename.
        Defaults to "".

    compression_distance
        The compression distance in nm.
        Defaults to 150.0.

    use_real_time
        Whether to use real time for the x-axis.
        Defaults to False.
    """
    if figure_path is None:
        figure_path = Path(__file__).parents[3] / "analysis_outputs/figures"
        figure_path.mkdir(parents=True, exist_ok=True)

    num_velocities = df["velocity"].nunique()
    total_time = 1.0
    time_label = "Normalized Time"
    plt.rcParams.update({"font.size": 16})

    for metric in metrics:
        fig, axs = plt.subplots(
            1, num_velocities, figsize=(num_velocities * 5, 5), sharey=True, dpi=300
        )
        axs = axs.ravel()
        for ct, (velocity, df_velocity) in enumerate(df.groupby("velocity")):
            if use_real_time:
                total_time = compression_distance / velocity  # s
                time_label = "Time (s)"
            for simulator, df_simulator in df_velocity.groupby("simulator"):
                for repeat, df_repeat in df_simulator.groupby("repeat"):
                    if repeat == 0:
                        label = f"{simulator}"
                    else:
                        label = "_nolegend_"
                    xvals = np.linspace(0, 1, df_repeat["time"].nunique()) * total_time
                    yvals = df_repeat.groupby("time")[metric.value].mean()

                    axs[ct].plot(
                        xvals,
                        yvals,
                        label=label,
                        color=SIMULATOR_COLOR_MAP[simulator],
                        alpha=0.6,
                    )
            axs[ct].set_title(f"Velocity: {velocity}")
            if ct == 0:
                axs[ct].legend()
        fig.supxlabel(time_label)
        fig.supylabel(metric.label())
        fig.tight_layout()
        fig.savefig(figure_path / f"all_simulators_{metric.value}_vs_time{suffix}.png")
