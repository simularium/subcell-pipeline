# %%
import math
from typing import List
import numpy as np
import pacmap
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D  # This is necessary for 3D plotting
import seaborn as sns
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
color_list = [
    "blue",
    "red",
    "green",
    "purple",
    "orange",
    "cyan",
    "black",
    "gray",
    "yellow",
    "pink",
]

def color_fader(c1, c2, mix=0) -> str:
    """Get a colour gradient to represent time"""
    c1 = np.array(mpl.colors.to_rgba(c1))
    c2 = np.array(mpl.colors.to_rgba(c2))
    return mpl.colors.to_rgba((1 - mix) * c1 + mix * c2)

def color_list_generator(num_of_vals_to_make: int, idx: int = 0) -> list[str]:
    """For each fiber, we'll want a different color for intensity to differentiate when aligning"""
     # TODO: adjust the alpha on the choices above based on idx
    c1 = "white"
    # select new colors via incrementing idx so we can color fibers differently (cycling through color_list)
    c2 = color_list[idx % len(color_list)]
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
def align_fibers(fibers: np.ndarray, align_to_fiber = np.ndarray) -> np.ndarray:
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
    ref = (align_to_fiber if align_to_fiber is not None else fibers[-1][0]).copy().reshape((points_per_fiber, 3))
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
            # --- coordinate scaler (coordinate systems scale differences don't allow shared space)
            param_coordinate_scaler = 1
            if sim_name == "readdy":
                param_coordinate_scaler = 0.001
            # --- append
            df = pd.DataFrame({
                'fibers': [np.array(fibers)], # need as np array for shape property
                'source': sim_name,
                'param_fibers': len(fibers),
                'param_timepoints': num_timepoints,
                'param_fiber_monomers': num_monomers,
                'param_dimensions': 3,
                'param_velocity': param_velocity,
                'param_coordinate_scaler': param_coordinate_scaler
            })
            study_dfs.append(df)
    return study_dfs


# --- reshaping datasets for analysis
def get_study_df_ref_fiber(study_df: pd.DataFrame) -> np.ndarray:
    """
    Grab first fiber from the study dataset. We're not going to transfrom the fiber, that's for the fiber align function
    """
    return study_df.fibers[0][-1][0]

def prep_study_df_fibers(study_df: pd.DataFrame, align: bool = False, fiber_for_alignment = None):
    """
    Function for getting a consistently (re)shaped dataset for PCA/PACMAP analysis
    """
    num_pca_samples = int(study_df.param_fibers) * int(study_df.param_timepoints)
    num_pca_features = int(study_df.param_fiber_monomers) * int(study_df.param_dimensions)
    fibers = study_df.fibers[0] if align == False else align_fibers(study_df.fibers[0], align_to_fiber=fiber_for_alignment)
    fibers_scaled = fibers * study_df.param_coordinate_scaler[0] # scale the coordinates for aligning multiple sims
    fibers_scaled_flattened = np.ravel(fibers_scaled)
    pca_dataset = fibers_scaled_flattened.reshape((num_pca_samples, num_pca_features))
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

def get_study_pca_dfs(study_dfs: List[pd.DataFrame], align: bool) -> [List[pd.DataFrame], PCA]:
    """
    Get a PCA analysis dataframe from a fiber study dataframe.
    """
    pca_space = PCA(n_components=2)
    # --- create our embedding space by fitting the combined data (dataset func flattens/orders our monomer datapoints)
    fiber_for_alignment = get_study_df_ref_fiber(study_dfs[0])
    study_datasets = [prep_study_df_fibers(study_df, align, fiber_for_alignment) for study_df in study_dfs]
    combined_prepared_data = np.vstack(study_datasets) # needing to create a 1 dim arr, but errs if I just pass in [study_datasets], TODO: better understand this
    pca_space.fit(combined_prepared_data)
    # --- transform the data
    pca_transformed_study_data = [pca_space.transform(data) for data in study_datasets]
    # --- scale/normalize the data
    pca_results = []
    for study_idx, pca_transformed_study_data in enumerate(pca_transformed_study_data):
        pca_df = pd.DataFrame(
            data=pca_transformed_study_data,
            columns=["principal component 1", "principal component 2"],
        )
        pca_df["label"] = study_dfs[study_idx].param_velocity
        pca_df["time"] = int(study_dfs[study_idx].param_timepoints)
        pca_df["source"] = study_dfs[study_idx].source
        pca_results.append(pca_df)
    return pca_results, pca_space


# %%
# ### PACMAP: Setup

def get_study_pacmap_dfs(study_dfs: List[pd.DataFrame], align: bool) -> List[pd.DataFrame]:
    """
    Get a PACMAP analysis dataframe from a fiber study dataframe.
    """
    n_neighbors = sum(map(lambda df: df.param_fibers.values[0], study_dfs)) # per fiber? meaning, all study fiber counts?
    pacmap_space = pacmap.PaCMAP(n_components=2, n_neighbors=n_neighbors)
    # --- create the embedding space
    fiber_for_alignment = get_study_df_ref_fiber(study_dfs[0])
    study_datasets = [prep_study_df_fibers(study_df, align, fiber_for_alignment) for study_df in study_dfs]
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
            data=pacmap_transformed_study_data[start:end],
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
def plot_study_df(analysis_df: pd.DataFrame, title: str, figsize=6, pca_space: PCA = None):
    """
    Plot a PCA analysis dataframe
    """
    fig, ax = plt.subplots(figsize=(figsize, figsize))
    # --- List of parameters or conditions under which the PCA was run.
    ax.set_xlabel(f"PC1{f': {pca_space.explained_variance_ratio_[0]}' if pca_space != None else ''}", loc="left")
    ax.set_ylabel(f"PC2{f': {pca_space.explained_variance_ratio_[1]}' if pca_space != None else ''}", loc="bottom")
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
def plot_study_dfs(analysis_dfs: List[pd.DataFrame], title: str, figsize=6, pca_space: PCA = None):
    """
    Plot multiple analysis dataframes atop each other. Increment colors by simulator count
    """
    fig, ax = plt.subplots(figsize=(figsize, figsize))
    # --- List of parameters or conditions under which the PCA was run.
    ax.set_xlabel(f"PC1{f': {pca_space.explained_variance_ratio_[0]}' if pca_space != None else ''}", loc="left")
    ax.set_ylabel(f"PC2{f': {pca_space.explained_variance_ratio_[1]}' if pca_space != None else ''}", loc="bottom")
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

# --- quick way to see what colors are being used for what sims
def plot_study_dfs_legend(study_dfs: List[pd.DataFrame]):
    for idx, study_df in enumerate(study_dfs):
        print(f"{study_df.source[0]}: {color_list[idx]}")


# %%
# # Inverse Transformations Funcs: Exmaning Differences in Original/Transformed Data
def calc_pca_component_distributions(pca_space: PCA, pca_transform_dataframes: List[pd.DataFrame]) -> List[dict]:
    # print(f"[calc_pca_component_distributions] PCA DFs: {pca_transform_dataframes}")
    pca_sets = []
    for pca_transform_dataframe in pca_transform_dataframes:
        for idx, label in enumerate(["principal component 1", "principal component 2"]):
            pc_mean = np.mean(pca_transform_dataframe[label].values)
            pc_std = abs(np.std(pca_transform_dataframe[label].values))
            # --- setup PC1/2 vals for analysis
            df = pd.DataFrame({
                "label": label,
                "values": [
                    dict(input=[pc_mean - pc_std * 2, 0] if idx == 0 else [0, pc_mean - pc_std * 2], fiber=None),
                    dict(input=[pc_mean - pc_std, 0] if idx == 0 else [0, pc_mean - pc_std], fiber=None),
                    dict(input=[pc_mean, 0] if idx == 0 else [0, pc_mean], fiber=None),
                    dict(input=[pc_mean + pc_std, 0] if idx == 0 else [0, pc_mean + pc_std], fiber=None),
                    dict(input=[pc_mean + pc_std * 2, 0] if idx == 0 else [0, pc_mean + pc_std * 2], fiber=None),
                ],
                # feels duplicative, but helpful to have at different levels for different contexts :/
                "source": pca_transform_dataframe.source[0]
            })
            # --- append source so we can reference later in legends
            for idx, val in enumerate(df['values']):
                df['values'][idx]['source'] = pca_transform_dataframe.source[0]
            # --- invert transform fibers
            for idx, val in enumerate(df['values']):
                projected_fiber = pca_space.inverse_transform([val['input'][0], val['input'][1]]) # duplicative, but just want to make clear the interface
                df['values'][idx]['fiber'] = projected_fiber.reshape(-1, 3) # transformation is a flat
            # --- append to set
            pca_sets.append(df)
    return pca_sets

def plot_inverse_transform_pca(pca_sets: List[pd.DataFrame], title_prefix: str):
    # print(f"[plot_inverse_transform_pca] PCA Sets: {len(pca_sets)}")
    # 2D Layout
    fig, axs = plt.subplots(2, 5, sharex=True, sharey=True, figsize=(24, 12))
    for idx_pca, pca_set in enumerate(pca_sets):
        for idx_val, val in enumerate(pca_set['values']):
            # including % 2 so if we have multiple sims, we overlay plots. only going to have 2 rows because they represent PC1/PC2
            # dividing alpha by idx_pca to clarify different simulators
            axs[idx_pca % 2, idx_val].plot(val['fiber'][:, 0], val['fiber'][:, 0], c='#cccccc', alpha=(1 / (idx_pca + 1)))
            axs[idx_pca % 2, idx_val].plot(val['fiber'][:, 0], val['fiber'][:, 1], c='b', alpha=(1 / (idx_pca + 1)), label=f"[{val['source']}] Fiber Y")
            axs[idx_pca % 2, idx_val].plot(val['fiber'][:, 0], val['fiber'][:, 2], c='g', alpha=(1 / (idx_pca + 1)), label=f"[{val['source']}] Fiber Z")
            axs[idx_pca % 2, idx_val].set_xlabel("x")
            axs[idx_pca % 2, idx_val].set_ylabel("y")
            # prefix title for sim name / velocity params
            axs[idx_pca % 2, idx_val].set_title(f"[{title_prefix}] PC1={str(val['input'][0])[0:6]} PC2={str(val['input'][1])[0:6]}", fontsize=14)
            axs[idx_pca % 2, idx_val].legend()
    plt.tight_layout()
    plt.show()
    # 3D plot of fiber positions
    fig = plt.figure(figsize=(18, 10))
    pca_sets_by_comp = [
        list(filter(lambda pca_set: pca_set['label'][0] == "principal component 1", pca_sets)),
        list(filter(lambda pca_set: pca_set['label'][0] == "principal component 2", pca_sets)),
    ]
    for idx_pca_comp, pca_set_by_comp in enumerate(pca_sets_by_comp):
        ax = fig.add_subplot(1, 2, idx_pca_comp + 1, projection='3d')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f"[{title_prefix}] {pca_set_by_comp[0]['label'][0]}", fontsize=14)
        for idx_pca, pca_set in enumerate(pca_set_by_comp):
            for idx_val, val in enumerate(pca_set['values']):
                c = color_list[idx_val % len(color_list)] # this function is confusing as shit 
                x, y, z = zip(*val['fiber'])  # TIL you can pass in a tuple of values, and this resolves the legend issue + visualization
                ax.plot(x, y, z, c=c, alpha=(1 / (idx_pca + 1)), label=f"[{pca_set.source[0]}] {str(val['input'][0])[0:6]}, {str(val['input'][1])[0:6]}")
        ax.legend()
    plt.tight_layout()
    plt.show()


# %%
# Data Loading: Combiled Subsample
subsamples_df = pd.read_csv(f"{data_directory}/dataframes/combined_actin_compression_dataset_subsampled.csv")
study_dfs = study_subsamples_loader(subsamples_df)


# %%
# # Plot: Per Simulation
velocities_to_plot = list(set(map(lambda df: df.param_velocity.values[0], study_dfs))) # grabs all unique velocities, and sort low->high
velocities_to_plot.sort()

for velocity in velocities_to_plot:
    study_dfs_to_plot = list(filter(lambda df: df.param_velocity.values[0] == velocity, study_dfs))
    print(f"Plotting {len(study_dfs_to_plot)} studies with velocity = {velocity}...")

    # PCA
    for st_df in study_dfs_to_plot:
        source = st_df['source'].values[0].upper()
        print(f"Plotting '{source}' PCA...")
        # [pca_df], pca_space = get_study_pca_dfs([st_df], align=False)
        # plot_study_df(pca_df, f"{source} (PCA): Aligned={False}, Velocity={float(velocity)}", figsize=6, pca_space=pca_space)
        [pca_aligned_df], pca_aligned_space = get_study_pca_dfs([st_df], align=True)
        # plot_study_df(pca_aligned_df, f"{source} (PCA): Aligned={True},Velocity={float(velocity)}", figsize=6, pca_space=pca_aligned_space)
        # --- inverse transforms
        print(f"Plotting '{source}' PCA Inverted Transformation...")
        pca_component_dists = calc_pca_component_distributions(pca_space=pca_aligned_space, pca_transform_dataframes=[pca_aligned_df])
        plot_inverse_transform_pca(pca_sets=pca_component_dists, title_prefix=f"{source} / {float(velocity)}")

    # PCA: All Sims
    print(f"Plotting All PCA...")
    # plot_study_dfs_legend(study_dfs_to_plot)
    # pca_dfs, pca_space = get_study_pca_dfs(study_dfs_to_plot, align=False)
    # plot_study_dfs(pca_dfs, f"ALL (PCA): Aligned={False}, Velocity={float(velocity)}", figsize=6, pca_space=pca_space)
    pca_aligned_dfs, pca_aligned_space = get_study_pca_dfs(study_dfs_to_plot, align=True)
    # plot_study_dfs(pca_aligned_dfs, f"ALL (PCA): Aligned={True}, Velocity={float(velocity)}", figsize=6, pca_space=pca_aligned_space)
    # --- inverse transforms
    pca_component_dists = calc_pca_component_distributions(pca_space=pca_aligned_space, pca_transform_dataframes=pca_aligned_dfs)
    plot_inverse_transform_pca(pca_sets=pca_component_dists, title_prefix=f"ALL / {float(velocity)}")

    # # PACMAP
    # for st_df in study_dfs_to_plot:
    #     source = st_df['source'].values[0].upper()
    #     print(f"Plotting '{source}' PaCMAP...")
    #     # [pacmap_df] = get_study_pacmap_dfs([st_df], align=False)
    #     # plot_study_df(pacmap_df, f"{source} (PaCMAP): Aligned={False},Velocity={float(velocity)}", figsize=6)
    #     [pacmap_aligned_df] = get_study_pacmap_dfs([st_df], align=True)
    #     plot_study_df(pacmap_aligned_df, f"{source} (PaCMAP): Aligned={True},Velocity={float(velocity)}", figsize=6)

    # # PACMAP: All Sims
    # print(f"Plotting All PaCMAP...")
    # plot_study_dfs_legend(study_dfs_to_plot)
    # # pca_dfs = get_study_pacmap_dfs(study_dfs_to_plot, align=False)
    # # plot_study_dfs(pca_dfs, f"ALL (PaCMAP): Aligned={False}, Velocity={float(velocity)}", figsize=6)
    # pca_aligned_dfs = get_study_pacmap_dfs(study_dfs_to_plot, align=True)
    # plot_study_dfs(pca_aligned_dfs, f"ALL (PaCMAP): Aligned={True}, Velocity={float(velocity)}", figsize=6)


# %%
# # Plot: All PCA
all_pca_aligned_dfs, all_pca_aligned_space = get_study_pca_dfs(study_dfs, align=True)
# plot_study_dfs(all_pca_aligned_dfs, f"ALL (PCA): Aligned={True}, Velocity=All", figsize=12, pca_space=all_pca_aligned_space)
# all_pca_aligned_dfs = get_study_pacmap_dfs(study_dfs, align=True)
# plot_study_dfs(all_pca_aligned_dfs, f"ALL (PaCMAP): Aligned={True}, Velocity=All", figsize=12)

# --- inverse transforms
all_pca_component_dists = calc_pca_component_distributions(pca_space=all_pca_aligned_space, pca_transform_dataframes=all_pca_aligned_dfs)
plot_inverse_transform_pca(pca_sets=all_pca_component_dists, title_prefix=f"ALL/ALL")

