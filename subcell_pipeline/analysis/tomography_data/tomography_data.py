import pandas as pd
from io_collection.keys.check_key import check_key
from io_collection.load.load_dataframe import load_dataframe
from io_collection.save.save_dataframe import save_dataframe


def read_tomography_data(file: str, label: str = "fil") -> pd.DataFrame:
    """
    Read tomography data from file as dataframe.

    Parameters
    ----------
    file
        Path to tomography data.
    label
        Label for the filament id column.

    Returns
    -------
    :
        Dataframe of tomography data.
    """

    coordinates = pd.read_table(file, delim_whitespace=True)

    if len(coordinates.columns) == 4:
        coordinates.columns = [label, "X", "Y", "Z"]
    elif len(coordinates.columns) == 5:
        coordinates.columns = ["object", label, "X", "Y", "Z"]
    else:
        print("Data file [ {file} ] has an unexpected number of columns")

    return coordinates


def rescale_tomography_data(data: pd.DataFrame, scale_factor: float = 1.0) -> None:
    """
    Rescale tomography data from pixels to um.

    Parameters
    ----------
    data
        Unscaled tomography data.
    scale_factor
        Data scaling factor (pixels to um).
    """

    data["X"] = data["X"] * scale_factor
    data["Y"] = data["Y"] * scale_factor
    data["Z"] = data["Z"] * scale_factor


def get_branched_tomography_data(
    bucket: str,
    repository: str,
    datasets: "list[tuple[str, str]]",
    scale_factor: float = 1.0,
) -> pd.DataFrame:
    """
    Load or create merged branched actin tomography data for given datasets.

    Parameters
    ----------
    bucket
        Name of S3 bucket for input and output files.
    repository : str
        Data repository for downloading tomography data
    datasets : list[tuple[str, str]]
        Folders and names of branched actin datasets
    scale_factor : float, optional
        Data scaling factor (pixels to um).

    Returns
    -------
    pd.DataFrame
        Merged data.
    """

    return get_tomography_data(bucket, repository, datasets, "branched", scale_factor)


def get_unbranched_tomography_data(
    bucket: str,
    repository: str,
    datasets: "list[tuple[str, str]]",
    scale_factor: float = 1.0,
) -> pd.DataFrame:
    """
    Load or create merged unbranched actin tomography data for given datasets.

    Parameters
    ----------
    bucket
        Name of S3 bucket for input and output files.
    repository : str
        Data repository for downloading tomography data
    datasets : list[tuple[str, str]]
        Folders and names of branched actin datasets
    scale_factor : float, optional
        Data scaling factor (pixels to um).

    Returns
    -------
    pd.DataFrame
        Merged data.
    """

    return get_tomography_data(bucket, repository, datasets, "unbranched", scale_factor)


def get_tomography_data(
    bucket: str,
    repository: str,
    datasets: "list[tuple[str, str]]",
    group: str,
    scale_factor: float = 1.0,
) -> pd.DataFrame:
    """
    Load or create merged tomography data for given datasets.

    Parameters
    ----------
    bucket
        Name of S3 bucket for input and output files.
    repository : str
        Data repository for downloading tomography data
    datasets : list[tuple[str, str]]
        Folders and names of branched actin datasets
    group : str
        Actin filament group ("branched" or "unbranched")
    scale_factor : float, optional
        Data scaling factor (pixels to um).

    Returns
    -------
    pd.DataFrame
        Merged data.
    """

    data_key = f"tomogram_cme_all_{group}_coordinates.csv"

    if check_key(bucket, data_key):
        print(f"Loading existing combined tomogram data from [ { data_key } ]")
        return load_dataframe(bucket, data_key)
    else:
        all_tomogram_dfs = []

        for folder, name in datasets:
            print(f"Loading tomogram data for [ { name } ]")
            tomogram_file = f"{repository}/{folder}/{group.title()}Actin_{name}.txt"
            tomogram_df = read_tomography_data(tomogram_file)
            tomogram_df["dataset"] = name
            rescale_tomography_data(tomogram_df, scale_factor)
            all_tomogram_dfs.append(tomogram_df)

        all_tomogram_df = pd.concat(all_tomogram_dfs)

        print(f"Saving combined tomogram data to [ { data_key } ]")
        save_dataframe(bucket, data_key, all_tomogram_df, index=False)

        return all_tomogram_df
