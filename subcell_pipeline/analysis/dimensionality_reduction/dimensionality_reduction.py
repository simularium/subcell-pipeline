from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from io_collection.keys.check_key import check_key
from io_collection.load.load_dataframe import load_dataframe
from io_collection.save.save_dataframe import save_dataframe
from matplotlib.axes import Axes
from pacmap import PaCMAP
from sklearn.decomposition import PCA


def get_merged_data(
    bucket: str,
    series_name: str,
    condition_keys: list[str],
    random_seeds: list[int],
    align: bool = True,
) -> pd.DataFrame:
    """
    Load or create merged data for given conditions and random seeds.

    If merged data (aligned or unaligned) already exists, load the data.
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
    align
        True if data should be aligned, False otherwise.

    Returns
    -------
    :
        Merged data.
    """

    align_key = "all_samples_aligned" if align else "all_samples_unaligned"
    data_key = f"{series_name}/analysis/{series_name}_{align_key}.csv"

    # Return data, if merged data already exists.
    if check_key(bucket, data_key):
        print(
            f"Dataframe [ { data_key } ] already exists. Loading existing merged data."
        )
        return load_dataframe(bucket, data_key)

    all_samples: list[pd.DataFrame] = []

    for condition_key in condition_keys:
        series_key = f"{series_name}_{condition_key}" if condition_key else series_name

        for seed in random_seeds:
            print(f"Loading samples for [ {condition_key} ] seed [ {seed} ]")

            sample_key = f"{series_name}/samples/{series_key}_{seed:06d}.csv"
            samples = load_dataframe(bucket, sample_key)
            samples["seed"] = seed
            samples["key"] = condition_key

            if align:
                align_fibers(samples)

            all_samples.append(samples)

    samples_dataframe = pd.concat(all_samples)
    save_dataframe(bucket, data_key, samples_dataframe, index=False)

    return samples_dataframe


def align_fibers(data: pd.DataFrame) -> None:
    """
    Align fibers for each time point in the data.

    Parameters
    ----------
    data
        Simulated fiber data.
    """

    aligned_fibers = []

    for time, group in data.groupby("time", sort=False):
        coords = group[["xpos", "ypos", "zpos"]].values

        if time == 0:
            fiber = coords
        else:
            fiber = align_fiber(coords)

        aligned_fibers.append(fiber)

    all_aligned_fibers = np.vstack(aligned_fibers)

    data["xpos"] = all_aligned_fibers[:, 0]
    data["ypos"] = all_aligned_fibers[:, 1]
    data["zpos"] = all_aligned_fibers[:, 2]


def align_fiber(coords: np.ndarray) -> np.ndarray:
    """
    Aligns an array of x, y, z positions along the positive x axis.

    Parameters
    ----------
    coords
        Array of x, y, and z positions.
    """

    # Identify rotation angle based on distance to point furthest from (0,0)
    distances = np.sqrt(np.sum(coords[:, 1:] ** 2, axis=1))
    max_index = np.argmax(distances)
    angle = np.arctan2(coords[max_index, 2], coords[max_index, 1])

    # Create rotation matrix
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array(((c, -s), (s, c)))

    # Rotate y and z
    rotated = np.dot(coords[:, 1:], rot)

    return np.concatenate((coords[:, 0:1], rotated), axis=1)


def reshape_fibers(data: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Reshapes data from tidy data format to array of fibers and fiber features.

    Parameters
    ----------
    data
        Simulated fiber data.

    Returns
    -------
    :
        Array of fibers and dataframe of fiber features.
    """

    all_features = []
    all_fibers = []

    for (time, velocity, repeat, simulator), group in data.groupby(
        ["time", "velocity", "repeat", "simulator"]
    ):
        fiber = group[["xpos", "ypos", "zpos"]].values.reshape(-1, 1)
        all_fibers.append(fiber)
        all_features.append(
            {
                "TIME": time,
                "VELOCITY": velocity,
                "REPEAT": repeat,
                "SIMULATOR": simulator.upper(),
            }
        )

    return np.array(all_fibers).squeeze(), pd.DataFrame(all_features)


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

    print(f"Principal component 1: {(pca.explained_variance_ratio_[0] * 100):.1f} %")
    print(f"Principal component 2: {(pca.explained_variance_ratio_[1] * 100):.1f} %")

    return pca_results, pca


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

    return pacmap_results


def plot_fibers_by_key_and_seed(data: pd.DataFrame) -> None:
    """
    Plot simulated fiber data for each condition key and random seed.

    Parameters
    ----------
    data
        Simulated fiber data.
    """

    rows = data["key"].unique()
    cols = data["seed"].unique()

    _, ax = plt.subplots(
        len(rows), len(cols), figsize=(10, 6), sharey=True, sharex=True
    )

    for row_index, row in enumerate(rows):
        for col_index, col in enumerate(cols):
            if row_index == 0:
                ax[row_index, col_index].set_title(f"REPEAT = {col}")
            if col_index == 0:
                ax[row_index, col_index].set_ylabel(f"KEY = {row}")

            subset = data[(data["key"] == row) & (data["seed"] == col)]

            for (_, simulator), group in subset.groupby(["time", "simulator"]):
                color = "red" if simulator == "readdy" else "blue"
                coords = group[["xpos", "ypos", "zpos"]].values
                ax[row_index, col_index].plot(
                    coords[:, 1], coords[:, 2], lw=0.5, color=color, alpha=0.5
                )

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


def plot_pca_feature_scatter(pca_results: pd.DataFrame, features: dict) -> None:
    """
    Plot scatter of PCA components colored by the given features.

    Parameters
    ----------
    pca_results : pd.DataFrame
        PCA results data.
    features : dict
        Map of feature name to coloring.
    """

    _, ax = plt.subplots(1, len(features), figsize=(10, 3), sharey=True, sharex=True)

    for index, (feature, colors) in enumerate(features.items()):
        plot_feature_scatter(ax[index], pca_results, "PC", feature, colors)

    plt.tight_layout()
    plt.show()


def plot_pacmap_feature_scatter(pacmap_results: pd.DataFrame, features: dict) -> None:
    """
    Plot scatter of PaCMAP embedding colored by the given features.

    Parameters
    ----------
    pacmap_results : pd.DataFrame
        PaCMAP results data.
    features : dict
        Map of feature name to coloring.
    """

    _, ax = plt.subplots(1, len(features), figsize=(10, 3), sharey=True, sharex=True)

    for index, (feature, colors) in enumerate(features.items()):
        plot_feature_scatter(ax[index], pacmap_results, "PACMAP", feature, colors)

    plt.tight_layout()
    plt.show()


def plot_feature_scatter(
    ax: Axes,
    data: pd.DataFrame,
    column: str,
    feature: str,
    colors: Union[dict, tuple[dict, str], str],
) -> None:
    """
    Plot scatter of specified dimensionality reduction colored by the given feature.

    Parameters
    ----------
    ax
        Axes object for plotting.
    data
        Dimensionality reduction results data.
    column
        Name of the embedding column.
    feature
        Feature column name.
    colors
        Coloring specification.
    """

    if isinstance(colors, dict):
        ax.scatter(
            data[f"{column}1"],
            data[f"{column}2"],
            s=2,
            c=data[feature].map(colors),
        )
    elif isinstance(colors, tuple):
        ax.scatter(
            data[f"{column}1"],
            data[f"{column}2"],
            s=2,
            c=data[feature].map(colors[0]),
            cmap=colors[1],
        )
    else:
        ax.scatter(
            data[f"{column}1"],
            data[f"{column}2"],
            s=2,
            c=data[feature],
            cmap=colors,
        )

    ax.set_title(feature)
    ax.set_xlabel(f"{column}1")
    ax.set_ylabel(f"{column}2")
