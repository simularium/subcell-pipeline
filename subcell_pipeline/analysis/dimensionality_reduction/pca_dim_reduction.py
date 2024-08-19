"""Methods for dimensionality reduction using PCA."""

import random
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from io_collection.save.save_dataframe import save_dataframe
from io_collection.save.save_figure import save_figure
from io_collection.save.save_json import save_json
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

    return pca_results, pca


def save_pca_results(
    pca_results: pd.DataFrame, save_location: str, save_key: str, resample: bool = True
) -> None:
    """
    Save PCA results data.

    Parameters
    ----------
    pca_results
        PCA trajectory data.
    save_location
        Location for output file (local path or S3 bucket).
    save_key
        Name key for output file.
    resample
        True if data should be resampled before saving, False otherwise.
    """

    if resample:
        pca_results = pca_results.copy().sample(frac=1.0, random_state=1)

    save_dataframe(save_location, save_key, pca_results, index=False)


def save_pca_trajectories(
    pca_results: pd.DataFrame, save_location: str, save_key: str
) -> None:
    """
    Save PCA trajectories data.

    Parameters
    ----------
    pca_results
        PCA trajectory data.
    save_location
        Location for output file (local path or S3 bucket).
    save_key
        Name key for output file.
    """

    output = []

    for (simulator, repeat, velocity), group in pca_results.groupby(
        ["SIMULATOR", "REPEAT", "VELOCITY"]
    ):
        output.append(
            {
                "simulator": simulator.upper(),
                "replicate": int(repeat),
                "velocity": velocity,
                "x": group["PCA1"].tolist(),
                "y": group["PCA2"].tolist(),
            }
        )

    random.Random(1).shuffle(output)
    save_json(save_location, save_key, output)


def save_pca_transforms(
    pca: PCA, points: list[list[float]], save_location: str, save_key: str
) -> None:
    """
    Save PCA transform data.

    Parameters
    ----------
    pca
        PCA object.
    points
        List of inverse transform points.
    save_location
        Location for output file (local path or S3 bucket).
    save_key
        Name key for output file.
    """

    output = []

    pc1_points, pc2_points = points

    for point in pc1_points:
        fiber = pca.inverse_transform([point, 0]).reshape(-1, 3)
        output.append(
            {
                "component": 1,
                "point": point,
                "x": fiber[:, 0].tolist(),
                "y": fiber[:, 1].tolist(),
                "z": fiber[:, 2].tolist(),
            }
        )

    for point in pc2_points:
        fiber = pca.inverse_transform([0, point]).reshape(-1, 3)
        output.append(
            {
                "component": 2,
                "point": point,
                "x": fiber[:, 0].tolist(),
                "y": fiber[:, 1].tolist(),
                "z": fiber[:, 2].tolist(),
            }
        )

    save_json(save_location, save_key, output)


def plot_pca_feature_scatter(
    data: pd.DataFrame,
    features: dict,
    pca: PCA,
    save_location: Optional[str] = None,
    save_key: str = "pca_feature_scatter.png",
) -> None:
    """
    Plot scatter of PCA components colored by the given features.

    Parameters
    ----------
    data
        PCA results data.
    features
        Map of feature name to coloring.
    pca
        PCA object.
    save_location
        Location for output file (local path or S3 bucket).
    save_key
        Name key for output file.
    """

    figure, ax = plt.subplots(
        1, len(features), figsize=(10, 3), sharey=True, sharex=True
    )

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

        ax[index].set_title(feature)
        ax[index].set_xlabel(f"PCA1 ({(pca.explained_variance_ratio_[0] * 100):.1f} %)")
        ax[index].set_ylabel(f"PCA2 ({(pca.explained_variance_ratio_[1] * 100):.1f} %)")

    plt.tight_layout()
    plt.show()

    if save_location is not None:
        save_figure(save_location, save_key, figure)


def plot_pca_inverse_transform(
    pca: PCA,
    pca_results: pd.DataFrame,
    save_location: Optional[str] = None,
    save_key: str = "pca_inverse_transform.png",
) -> None:
    """
    Plot inverse transform of PCA.

    Parameters
    ----------
    pca
        PCA object.
    pca_results
        PCA results data.
    save_location
        Location for output file (local path or S3 bucket).
    save_key
        Name key for output file.
    """

    figure, ax = plt.subplots(2, 3, figsize=(10, 6))

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

    for index in [0, 1]:
        ax[index, 0].set_xlabel("X")
        ax[index, 0].set_ylabel("Y", rotation=0)
        ax[index, 1].set_xlabel("Y")
        ax[index, 1].set_ylabel("Z", rotation=0)
        ax[index, 2].set_xlabel("X")
        ax[index, 2].set_ylabel("Z", rotation=0)

    for index in [0, 1, 2]:
        ax[0, index].set_title("PC1")
        ax[1, index].set_title("PC2")

    plt.tight_layout()
    plt.show()

    if save_location is not None:
        save_figure(save_location, save_key, figure)
