"""Methods for dimensionality reduction using PCA."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from subcell_pipeline.analysis.dimensionality_reduction.fiber_data import reshape_fibers


def run_pca(data: pd.DataFrame) -> tuple[pd.DataFrame, PCA]:
    """
    Run Principal Component Analysis (PCA) on simulation data.

    Parameters
    ----------
    data
        Simulated fiber data.

    Returns
    -------
    :
        Dataframe with PCA components appended and the PCA object.
    """

    all_fibers, all_features = reshape_fibers(data)

    pca = PCA(n_components=2)
    pca = pca.fit(all_fibers)
    transform = pca.transform(all_fibers)

    pca_results = pd.concat(
        [pd.DataFrame(transform, columns=["PCA1", "PCA2"]), all_features],
        axis=1,
    )
    pca_results = pca_results.sample(frac=1, random_state=1)

    return pca_results, pca


def plot_pca_feature_scatter(data: pd.DataFrame, features: dict, pca: PCA) -> None:
    """
    Plot scatter of PCA components colored by the given features.

    Parameters
    ----------
    data : pd.DataFrame
        PCA results data.
    features : dict
        Map of feature name to coloring.
    pca : PCA
        PCA object.
    """

    _, ax = plt.subplots(1, len(features), figsize=(10, 3), sharey=True, sharex=True)

    for index, (feature, colors) in enumerate(features.items()):
        if isinstance(colors, dict):
            ax[index].scatter(
                data["PCA1"],
                data["PCA2"],
                s=2,
                c=data[feature].map(colors),
            )
        elif isinstance(colors, tuple):
            ax[index].scatter(
                data["PCA1"],
                data["PCA2"],
                s=2,
                c=data[feature].map(colors[0]),
                cmap=colors[1],
            )
        else:
            ax[index].scatter(
                data["PCA1"],
                data["PCA2"],
                s=2,
                c=data[feature],
                cmap=colors,
            )

        ax.set_title(feature)
        ax.set_xlabel(f"PCA1 ({(pca.explained_variance_ratio_[0] * 100):.1f} %)")
        ax.set_ylabel(f"PCA2 ({(pca.explained_variance_ratio_[1] * 100):.1f} %)")

    plt.tight_layout()
    plt.show()


def plot_pca_inverse_transform(pca: PCA, pca_results: pd.DataFrame) -> None:
    """
    Plot inverse transform of PCA.

    Parameters
    ----------
    pca : PCA
        PCA object.
    pca_results : pd.DataFrame
        PCA results data.
    """

    _, ax = plt.subplots(2, 3, figsize=(10, 6))

    points = np.arange(-2, 2, 0.5)
    stdev_pc1 = pca_results["PCA1"].std(ddof=0)
    stdev_pc2 = pca_results["PCA2"].std(ddof=0)
    cmap = plt.colormaps.get_cmap("RdBu_r")

    for point in points:
        # Traverse PC 1
        fiber = pca.inverse_transform([point * stdev_pc1, 0]).reshape(-1, 3)
        ax[0, 0].plot(fiber[:, 0], fiber[:, 1], color=cmap((point + 2) / 4))
        ax[0, 1].plot(fiber[:, 1], fiber[:, 2], color=cmap((point + 2) / 4))
        ax[0, 2].plot(fiber[:, 0], fiber[:, 2], color=cmap((point + 2) / 4))

        # Traverse PC 2
        fiber = pca.inverse_transform([0, point * stdev_pc2]).reshape(-1, 3)
        ax[1, 0].plot(fiber[:, 0], fiber[:, 1], color=cmap((point + 2) / 4))
        ax[1, 1].plot(fiber[:, 1], fiber[:, 2], color=cmap((point + 2) / 4))
        ax[1, 2].plot(fiber[:, 0], fiber[:, 2], color=cmap((point + 2) / 4))

    ax[0, 0].set_title("PC1 X/Y")
    ax[0, 1].set_title("PC1 Y/Z")
    ax[0, 2].set_title("PC1 X/Z")

    ax[1, 0].set_title("PC2 X/Y")
    ax[1, 1].set_title("PC2 Y/Z")
    ax[1, 2].set_title("PC2 X/Z")

    plt.tight_layout()
    plt.show()
