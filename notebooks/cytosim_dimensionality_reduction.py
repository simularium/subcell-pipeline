# %%
import math
from typing import List
import numpy as np
import pacmap
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.spatial.transform import Rotation
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


# %%
# ### Init Fns/Helpers
# --- data refs
data_directory = "../data"

# --- math funcs
# Get RMSD between 2 curves
def rmsd(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
    return np.sqrt(((((vec1 - vec2) ** 2)) * 3).mean())

# --- color funcs
def color_fader(c1, c2, mix=0) -> str:
    """Get a colour gradient to represent time"""
    c1 = np.array(mpl.colors.to_rgba(c1))
    c2 = np.array(mpl.colors.to_rgba(c2))
    return mpl.colors.to_rgba((1 - mix) * c1 + mix * c2)

def color_list_generator(num_of_vals_to_make: int, idx: int = 0) -> list[str]:
    """For each fiber, we'll want a different color for intensity to differentiate when aligning"""
     # TODO: adjust the alpha on the choices above based on idx
    c1 = "white"
    # select new colors via incrementing idx so we can color fibers differently
    c2 = [
        "blue",
        "red",
        "green",
        "purple",
        "orange",
    ][idx]
    # return range of values
    return [color_fader(c1, c2, i / num_of_vals_to_make) for i in range(num_of_vals_to_make)]

def marker_list_generator(num_of_vals_to_make: int, idx: int = 0) -> list[str]:
    marker_list = [
        "o",
        "s",
        "D",
        "*",
        "P",
    ]
    return [marker_list[idx]] * num_of_vals_to_make

# --- BLAIR'S NOTEBOOK FUNCS. TODO: REMOVE THIS
def align_fibers(fibers: np.ndarray) -> np.ndarray:
    """
    Rotationally align the given fibers around the x-axis.

    Parameters
    ----------
    fiber_points: np.ndarray (shape = time x fiber x (3 * points_per_fiber))
        Array containing the flattened x,y,z positions of control points
        for each fiber at each time.

    Returns
    ----------
    aligned_data: np.ndarray
        The given data aligned.
    """
    # get angle to align each fiber at the last time point
    align_by = []
    points_per_fiber = int(fibers.shape[2] / 3)
    # reference fiber selected. choosing last fiber position for reference?
    ref = fibers[-1][0].copy().reshape((points_per_fiber, 3))
    for fiber_ix in range(len(fibers[-1])):
        best_rmsd = math.inf
        for angle in np.linspace(0, 2 * np.pi, 1000):
            rot = Rotation.from_rotvec(angle * np.array([1, 0, 0]))
            new_vec = Rotation.apply(
                rot, fibers[-1][fiber_ix].copy().reshape((points_per_fiber, 3))
            )
            test_rmsd = rmsd(new_vec, ref)
            if test_rmsd < best_rmsd:
                best_angle = angle
                best_rmsd = test_rmsd
        align_by.append(best_angle)
    # align all the fibers to ref across all time points
    aligned = np.zeros_like(fibers)
    for fiber_ix in range(fibers.shape[1]):
        rot = Rotation.from_rotvec(align_by[fiber_ix] * np.array([1, 0, 0]))
        for time_ix in range(fibers.shape[0]):
            fiber = fibers[time_ix][fiber_ix].copy().reshape((points_per_fiber, 3))
            new_fiber = Rotation.apply(rot, fiber)
            aligned[time_ix][fiber_ix] = new_fiber.flatten()
    return aligned


# %%
# ### Data Preppers

# --- loaded csvs/txt files into study dataframes (DEPRECATED: we're now expecting a pre-processed single CSV of subsampled data)
def study_loader(data_df, source, num_fiber_monomers, num_timepoints):
    study_dfs = []
    for param_velocity, velocity_df in data_df.groupby('velocity'):
        fibers = [] # 1 fiber per simulation, so treating 'repeats' as fibers
        for repeat, repeat_df in velocity_df.groupby('repeat'):
            fiber_timepoints = []
            for time, time_df in repeat_df.groupby('time'):
                fiber_timepoints.append(time_df[['xpos', 'ypos', 'zpos']].values.flatten()) # flattening to match data patterns
            fibers.append(fiber_timepoints) # 1001 timepoints x flattened(50 monomers x 3 dimensions)
        # --- append
        df = pd.DataFrame({
            'fibers': [np.array(fibers)], # need as np array for shape property
            'source': source,
            'param_fibers': len(fibers),
            'param_timepoints': num_timepoints,
            'param_fiber_monomers': num_fiber_monomers,
            'param_dimensions': 3,
            'param_velocity': param_velocity
        })
        study_dfs.append(df)
    return study_dfs

# --- load in prepped CSV of subsamples (pre-processed for timepoints, monomers)
def study_subsamples_loader(subsamples_df: pd.DataFrame):
    study_dfs = []
    num_timepoints = 200
    num_monomers = 200
    for sim_name, sim_df in subsamples_df.groupby("simulator"):
        for param_velocity, velocity_df in sim_df.groupby('velocity'):
            fibers = [] # 1 fiber per simulation, so treating 'repeats' as fibers
            for repeat, repeat_df in velocity_df.groupby('repeat'):
                fiber_timepoints = []
                for time, time_df in repeat_df.groupby('time'):
                    fiber_timepoints.append(time_df[['xpos', 'ypos', 'zpos']].values.flatten()) # flattening to match data patterns
                fibers.append(fiber_timepoints) # 1001 timepoints x flattened(50 monomers x 3 dimensions)
            # --- append
            df = pd.DataFrame({
                'fibers': [np.array(fibers)], # need as np array for shape property
                'source': sim_name,
                'param_fibers': len(fibers),
                'param_timepoints': num_timepoints,
                'param_fiber_monomers': num_monomers,
                'param_dimensions': 3,
                'param_velocity': param_velocity
            })
            study_dfs.append(df)
    return study_dfs


# --- reshaping datasets for analysis
def get_study_df_analysis_dataset(study_df: pd.DataFrame, align: bool = False):
    """
    Function for getting a consistently (re)shaped dataset for PCA/PACMAP analysis
    """
    # print('get_study_df_analysis_dataset: ', np.array(study_df.fibers[0]).shape)
    num_pca_samples = int(study_df.param_fibers) * int(study_df.param_timepoints)
    num_pca_features = int(study_df.param_fiber_monomers) * int(study_df.param_dimensions)
    fibers = study_df.fibers[0] if align == False else align_fibers(study_df.fibers[0])
    fibers_flattened = np.ravel(fibers)
    pca_dataset = fibers_flattened.reshape((num_pca_samples, num_pca_features))
    return pca_dataset


# %%
# ### Incrementing Specturm/Legend

def plot_timepoint_sepectrum(colors_by_step: list[str], step_label: str):
    num_steps = len(colors_by_step)
    # Plot
    plt.figure(figsize=(8, 1))
    for i, color in enumerate(colors_by_step):
        plt.barh(1, width=1, left=i, height=1, color=color)
    # Add step labels
    plt.text(0, 1.5, '0', ha='center', va='bottom')
    plt.text(num_steps, 1.5, num_steps, ha='center', va='bottom')
    for i in range(math.trunc(num_steps / 6), num_steps, math.trunc(num_steps / 6)):
        plt.text(i, 1.5, str(i), ha='center', va='bottom')
    # Add legend/title/etc.
    plt.xlim(0, num_steps)
    plt.axis('off')  # Turning off the axis for a clean look
    plt.title(f"Legend: {step_label}", size=10, pad=14)
    plt.show()


# %%
# ### PCA: Setup

def get_study_pca_dfs(study_dfs: List[pd.DataFrame], align: bool) -> List[pd.DataFrame]:
    """
    Get a PCA analysis dataframe from a fiber study dataframe.
    """
    pca_space = PCA(n_components=2)
    # --- create our embedding space by fitting the combined data (dataset func flattens/orders our monomer datapoints)
    study_datasets = [get_study_df_analysis_dataset(study_df, align) for study_df in study_dfs]
    combined_prepared_data = np.vstack(study_datasets) # needing to create a 1 dim arr, but errs if I just pass in [study_datasets], TODO: better understand this
    pca_space.fit(combined_prepared_data)
    # --- transform the data
    pca_transformed_study_data = [pca_space.transform(data) for data in study_datasets]
    # --- scale/normalize the data
    pca_results = []
    for study_idx, pca_transformed_study_data in enumerate(pca_transformed_study_data):
        pca_df = pd.DataFrame(
            data=MinMaxScaler().fit_transform(pca_transformed_study_data),
            columns=["principal component 1", "principal component 2"],
        )
        pca_df["label"] = study_dfs[study_idx].param_velocity
        pca_df["time"] = int(study_dfs[study_idx].param_timepoints)
        pca_results.append(pca_df)
    return pca_results


# %%
# ### PACMAP: Setup

def get_study_pacmap_dfs(study_dfs: List[pd.DataFrame], align: bool) -> List[pd.DataFrame]:
    """
    Get a PACMAP analysis dataframe from a fiber study dataframe.     
    """
    n_neighbors = sum(map(lambda df: df.param_fibers.values[0], study_dfs)) # per fiber? meaning, all study fiber counts?
    pacmap_space = pacmap.PaCMAP(n_components=2, n_neighbors=n_neighbors)
    # --- create the embedding space
    study_datasets = [get_study_df_analysis_dataset(study_df, align) for study_df in study_dfs]
    combined_prepared_data = np.vstack(study_datasets)
    # --- fix+transform the data (we're using fit_trasnform, because pacmap treats "transform" as additional/new datasets and it says positions can be in different places)
    pacmap_transformed_study_data = pacmap_space.fit_transform(combined_prepared_data, init="pca")
    # --- scale/normalize the data (we have to regroup data since previous step collapsed it)
    pacmap_results = []
    start = 0
    for study_df in study_dfs:
        num_samples = study_df.param_fibers.values[0] * study_df.param_timepoints.values[0]
        end = start + num_samples
        pacmap_df = pd.DataFrame(
            data=MinMaxScaler().fit_transform(pacmap_transformed_study_data[start:end]),
            columns=["principal component 1", "principal component 2"]
        )
        pacmap_df["label"] = study_df.param_velocity
        pacmap_df["time"] = int(study_df.param_timepoints)
        pacmap_results.append(pacmap_df)
        start = end
    return pacmap_results


# %%
# ### Plotter Setup

# --- for single simulation
def plot_study_df(analysis_df: pd.DataFrame, title: str, figsize=6):
    """
    Plot a PCA analysis dataframe
    """
    fig, ax = plt.subplots(figsize=(figsize, figsize))
    # --- List of parameters or conditions under which the PCA was run.
    ax.set_xlabel("PC1", loc="left")
    ax.set_ylabel("PC2", loc="bottom")
    ax.set_title(title, fontsize=10)
    # --- setup points + skip counter
    num_pc1_points = int(analysis_df['principal component 1'].shape[0])
    num_timepoints = analysis_df.time.iloc[-1] # this is a series. each monomer point is saved with a time position. we grab the last value to know what the last monomer timepoint is
    n_skip = 1 # 1 = no skips
    if num_timepoints > 200:
        n_skip = math.floor(num_timepoints / 200) # skip data points for faster plotting
        print("High Volume Plot. Setting n_skip = ", n_skip)
    # --- compile flat arr of colors (feels weird but works)
    pc1_color_lists = []
    for fiber_idx in range(num_pc1_points // num_timepoints): # segment by fiber
        pc1_color_lists.append(color_list_generator(num_timepoints, fiber_idx))
    pc1_color_list = [c for cl in pc1_color_lists for c in cl] # flattens
    # --- scatter
    for i in range(num_pc1_points)[::n_skip]:
        ax.scatter(
            analysis_df.loc[i, "principal component 1"],
            analysis_df.loc[i, "principal component 2"],
            c=[pc1_color_list[i]],
            s=70,
        )
    plt.show()

# --- for many simulations overlapping
def plot_study_dfs(analysis_dfs: List[pd.DataFrame], title: str, figsize=6):
    """
    Plot multiple analysis dataframes atop each other. Increment colors by simulator count
    """
    fig, ax = plt.subplots(figsize=(figsize, figsize))
    # --- List of parameters or conditions under which the PCA was run.
    ax.set_xlabel("PC1", loc="left")
    ax.set_ylabel("PC2", loc="bottom")
    ax.set_title(title, fontsize=10)
    # --- for each simulation, plot..
    for analysis_idx, analysis_df in enumerate(analysis_dfs):
        # --- setup points + skip counter
        num_pc1_points = int(analysis_df['principal component 1'].shape[0])
        num_timepoints = analysis_df.time.iloc[-1] # this is a series. each monomer point is saved with a time position. we grab the last value to know what the last monomer timepoint is
        n_skip = 1 # 1 = no skips
        if num_timepoints > 200:
            n_skip = math.floor(num_timepoints / 200) # skip data points for faster plotting
            print("High Volume Plot. Setting n_skip = ", n_skip)

        # --- create a list for colors/markers
        pc1_color_lists = []
        pc1_marker_lists = []
        for fiber_idx in range(num_pc1_points // num_timepoints):
            pc1_color_lists.append(color_list_generator(num_timepoints, analysis_idx)) # all fibers for this sim will be same color, but we'll have opacity shift
            pc1_marker_lists.append(marker_list_generator(num_timepoints, fiber_idx)) # for each fiber, fill a list of symbol tags

        # --- scatter (flatten the lists so they pair with the scattered points)
        pc1_point_colors = [c for cl in pc1_color_lists for c in cl]
        pc1_point_markers = [m for ml in pc1_marker_lists for m in ml]
        for i in range(num_pc1_points)[::n_skip]:
            ax.scatter(
                analysis_df.loc[i, "principal component 1"],
                analysis_df.loc[i, "principal component 2"],
                c=[pc1_point_colors[i]],
                s=50,
                marker=pc1_point_markers[i]
            )
    plt.show()



# %%
# ### Explained Variance: Setup

def plot_explained_variance_visualization(pca: PCA):
    print(f"Explained Variance Visualization")
    plt.figure(figsize=(8, 4))
    # Plot the individual explained variance as a bar chart
    # The height of the bar represents the variance ratio for each principal component.
    plt.bar(range(pca.n_components_), pca.explained_variance_ratio_, alpha=0.5, align='center',
            label='Individual explained variance')
    # Overlay the cumulative explained variance as a step plot
    # np.cumsum computes the cumulative sum of explained variance ratios.
    # 'where=mid' places the step change in the middle of the x-tick.
    plt.step(range(pca.n_components_), np.cumsum(pca.explained_variance_ratio_), where='mid',
             label='Cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


# %%
# Data Loading: Combiled Subsample
subsamples_df = pd.read_csv(f"{data_directory}/dataframes/combined_actin_compression_dataset_subsampled.csv")
study_dfs = study_subsamples_loader(subsamples_df)


# %%
# # Plot: Graph per Simulation


velocities_to_plot = list(set(map(lambda df: df.param_velocity.values[0], study_dfs))) # grabs all unique velocities, and sort low->high
velocities_to_plot.sort()

for velocity in velocities_to_plot:
    study_dfs_to_plot = list(filter(lambda df: df.param_velocity.values[0] == velocity, study_dfs))
    print(f"Plotting {len(study_dfs_to_plot)} studies with velocity = {velocity}...")

    # PCA
    for st_df in study_dfs_to_plot:
        source = st_df['source'].values[0].upper()
        print(f"Plotting '{source}' PCA...")
        [pca_df] = get_study_pca_dfs([st_df], align=False)
        plot_study_df(pca_df, f"{source} (PAC): Aligned={False}, Velocity={float(velocity)}", figsize=4)
        [pca_aligned_df] = get_study_pca_dfs([st_df], align=True)
        plot_study_df(pca_aligned_df, f"{source} (PAC): Aligned={True},Velocity={float(velocity)}", figsize=4)

    # PCA: All Sims
    print(f"Plotting All PCA...")
    pca_dfs = get_study_pca_dfs(study_dfs_to_plot, align=False)
    plot_study_dfs(pca_dfs, f"ALL (PAC): Aligned={False}, Velocity={float(velocity)}", figsize=4)
    pca_aligned_dfs = get_study_pca_dfs(study_dfs_to_plot, align=True)
    plot_study_dfs(pca_aligned_dfs, f"ALL (PAC): Aligned={True}, Velocity={float(velocity)}", figsize=4)

    # PACMAP
    for st_df in study_dfs_to_plot:
        source = st_df['source'].values[0].upper()
        print(f"Plotting '{source}' PaCMAP...")
        [pacmap_df] = get_study_pacmap_dfs([st_df], align=False)
        plot_study_df(pacmap_df, f"{source} (PaCMAP): Aligned={False},Velocity={float(velocity)}", figsize=4)
        [pacmap_aligned_df] = get_study_pacmap_dfs([st_df], align=True)
        plot_study_df(pacmap_aligned_df, f"{source} (PaCMAP): Aligned={True},Velocity={float(velocity)}", figsize=4)

    # PACMAP: All Sims
    print(f"Plotting All PaCMAP...")
    pca_dfs = get_study_pacmap_dfs(study_dfs_to_plot, align=False)
    plot_study_dfs(pca_dfs, f"ALL (PaCMAP): Aligned={False}, Velocity={float(velocity)}", figsize=4)
    pca_aligned_dfs = get_study_pacmap_dfs(study_dfs_to_plot, align=True)
    plot_study_dfs(pca_aligned_dfs, f"ALL (PaCMAP): Aligned={True}, Velocity={float(velocity)}", figsize=4)





