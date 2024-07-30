"""Methods for dimensionality reduction using PaCMAP."""

import matplotlib.pyplot as plt
import pandas as pd
from pacmap import PaCMAP

from subcell_pipeline.analysis.dimensionality_reduction.fiber_data import reshape_fibers


def run_pacmap(data: pd.DataFrame) -> tuple[pd.DataFrame, PaCMAP]:
    """
    Run Pairwise Controlled Manifold Approximation (PaCMAP) on simulation data.

    Parameters
    ----------
    data
        Simulated fiber data.

    Returns
    -------
    :
        Dataframe with PaCMAP emebdding appended and the PaCMAP object.
    """

    all_fibers, all_features = reshape_fibers(data)

    pacmap = PaCMAP(n_components=2, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0)
    transform = pacmap.fit_transform(all_fibers)

    pacmap_results = pd.concat(
        [
            pd.DataFrame(transform, columns=["PACMAP1", "PACMAP2"]),
            pd.DataFrame(all_features),
        ],
        axis=1,
    )
    pacmap_results = pacmap_results.sample(frac=1, random_state=1)

    return pacmap_results, pacmap


def plot_pacmap_feature_scatter(data: pd.DataFrame, features: dict) -> None:
    """
    Plot scatter of PaCMAP embedding colored by the given features.

    Parameters
    ----------
    data
        PaCMAP results data.
    features
        Map of feature name to coloring.
    """

    _, ax = plt.subplots(1, len(features), figsize=(10, 3), sharey=True, sharex=True)

    for index, (feature, colors) in enumerate(features.items()):
        if isinstance(colors, dict):
            ax[index].scatter(
                data["PACMAP1"],
                data["PACMAP2"],
                s=2,
                c=data[feature].map(colors),
            )
        elif isinstance(colors, tuple):
            ax[index].scatter(
                data["PACMAP1"],
                data["PACMAP2"],
                s=2,
                c=data[feature].map(colors[0]),
                cmap=colors[1],
            )
        else:
            ax[index].scatter(
                data["PACMAP1"],
                data["PACMAP2"],
                s=2,
                c=data[feature],
                cmap=colors,
            )

        ax[index].set_title(feature)
        ax[index].set_xlabel("PACMAP1")
        ax[index].set_ylabel("PACMAP2")

    plt.tight_layout()
    plt.show()
