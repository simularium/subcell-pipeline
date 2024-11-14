"""Methods compression metric analysis and plotting."""

from random import uniform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from io_collection.load.load_dataframe import load_dataframe
from io_collection.save.save_figure import save_figure
import MDAnalysis as mda
from MDAnalysis.analysis.polymer import PersistenceLength

def get_persistence_length_data(
    bucket: str,
    series_name: str,
    condition_keys: list[str],
    random_seeds: list[int],
) -> dict[str, list[float]]:
    """
    Log mean and standard deviation of persistence lengths for a given dataset.

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

    Returns
    -------
    :
        Dict with list of persistence length over time mapped to each run name.
    """
    result = {}
    for condition_key in condition_keys:
        series_key = f"{series_name}_{condition_key}" if condition_key else series_name

        for seed in random_seeds:
            print(
                f"Loading samples and calculating persistence length for "
                f"seed [ {seed} ]"
            )

            sample_key = f"{series_name}/samples/{series_key}_{seed:06d}.csv"
            samples = load_dataframe(bucket, sample_key)
            
            sim_name = bucket.split("s3://")[1].split("-")[0]
            result[f"{sim_name}_{seed:01d}"] = calculate_persistence_lengths(samples)
            
    return result


def calculate_persistence_lengths(df: pd.DataFrame) -> list[float]:
    """
    Calculate persistence length (µm) over time 
    for a single simulation run.

    Parameters
    ----------
    df
        Input data for a single simulator.

    Returns
    -------
    :
        list of persistence length per time.
    """
    result = []
    for _, fiber_at_time in df.groupby("time"):
        
        polymer_trace = fiber_at_time[["xpos", "ypos", "zpos"]].values
        
        n_points = polymer_trace.shape[0]
        u = mda.Universe.empty(
            n_atoms=n_points,
            n_residues=n_points,
            atom_resindex=np.arange(n_points).tolist(),
            residue_segindex=[0] * n_points,
            trajectory=True,
        )
        u.add_TopologyAttr("name", ["p"] * n_points)
        u.add_TopologyAttr("bonds", [(n, n + 1) for n in np.arange(n_points - 1)])
        u.atoms.positions = polymer_trace
        chains = u.atoms.fragments
        plen = PersistenceLength(chains)
        plen.run()
        result.append(plen.results.lp * 1e-3) # nm -> µm

    return result


def plot_persistence_length(
    data: dict[str, list[float]],
    save_location: str,
) -> None:
    """
    Plot persistence length for each trajectory.

    Parameters
    ----------
    data
        Persistence length results for each trajectory at each time.
    save_location
        Location for output file (local path or S3 bucket).
    """
    figure, ax = plt.subplots()
    times = None
    
    greens = matplotlib.colormaps.get_cmap('GnBu')
    oranges = matplotlib.colormaps.get_cmap('YlOrBr')
    
    for name, results in data.items():
        
        if times is None:
            times = (np.arange(len(results)) / len(results)).tolist()
            
        cmap = greens if name.startswith("cytosim") else oranges
        color = matplotlib.colors.to_hex(cmap(uniform(0.4, 0.7)))
        
        ax.plot(times, results, color=color, label=name)
        
    EXP_ACTIN_PERSISTENCE_LENGTH = 10.7 
    ax.plot(times, [EXP_ACTIN_PERSISTENCE_LENGTH] * len(results), color='black', label="experimental")
    
    ax.set_title("Persistence Length of Uncompressed Actin")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Persistence Length (µm)")
    ax.set_ylim(0, 30 * EXP_ACTIN_PERSISTENCE_LENGTH)
    ax.legend(loc="best")
    figure.set_size_inches(12.8, 9.6)
    figure.set_dpi(100)
    
    if save_location is not None:
        save_figure(save_location, "persistence_length/persistence_length.png", figure)

