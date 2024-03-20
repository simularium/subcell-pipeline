# %%
import math
from typing import List
import numpy as np
import os
import pacmap
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D  # This is necessary for 3D plotting
import seaborn as sns
from scipy.spatial.transform import Rotation
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


# %%
# ### Init Fns/Helpers
# --- data refs
data_directory = "../data"
figures_directory = "../data/figures"

# --- directory setup (if data, data/figures don't exist)
if not os.path.exists(data_directory):
    os.makedirs(data_directory)
if not os.path.exists(figures_directory):
    os.makedirs(figures_directory)

# --- math funcs
# Get RMSD between 2 curves
def rmsd(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
    return np.sqrt(((((vec1 - vec2) ** 2)) * 3).mean())

def normalize(value, min_value, max_value):
    return (value - min_value) / (max_value - min_value)

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
marker_list = [
    "o",
    "s",
    "D",
    "*",
    "P",
]

def color_fader(c1, c2, mix=0) -> str:
    """Get a colour gradient to represent time"""
    c1 = np.array(mpl.colors.to_rgb(c1))
    c2 = np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_rgb((1 - mix) * c1 + mix * c2) # dropping to rgb because alpha is being used by metrics now

def color_list_generator(num_of_vals_to_make: int, idx: int = 0, c1_override: str = None, c2_override: str = None) -> list[str]:
    """For each fiber, we'll want a different color for intensity to differentiate when aligning"""
     # TODO: adjust the alpha on the choices above based on idx
    c1 = c1_override or "white"
    # select new colors via incrementing idx so we can color fibers differently (cycling through color_list)
    c2 = c2_override or color_list[idx % len(color_list)]
    # return range of values
    return [color_fader(c1, c2, i / num_of_vals_to_make) for i in range(num_of_vals_to_make)]

def marker_list_generator(num_of_vals_to_make: int, idx: int = 0) -> list[str]:
    return [marker_list[idx % len(marker_list)]] * num_of_vals_to_make

def source_to_idx(source: str) -> int:
    return {
        "cytosim": 0,
        "readdy": 1,
    }[source]

# --- alignments
def find_best_fit_lines_theta_from_fibers(fiber_at_timepoint: np.ndarray) -> float:
    """Find the best fit line for a fiber's monomers (x,y,z)."""
    # --- fit line
    x = fiber_at_timepoint[:, 0]
    y = fiber_at_timepoint[:, 1]
    z = fiber_at_timepoint[:, 2]
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    # --- get theta
    theta = np.arctan(m)
    return theta


def align_fibers(fibers: np.ndarray) -> np.ndarray:
    """Rotationally align the given fibers around the x-axis to the fiber with the greatest magnitude."""
    # v1 - get angle to align each fiber at the last time point
    # v2 - change this function so it finds highest magnitude point, gets rotation theta, and for each monomer timepoint rotates accordingly to be in same plane
    fibers_mapped = fibers.copy()
    # for fibers, fit a line and calculate the theta
    # print("[align_fibers] best_fit_theta: ", best_fit_theta)
    # for each fiber at each time point, align all monomers to that highest magnitude point
    for fiber_idx, fiber in enumerate(fibers):
        for fiber_timepoint_idx, fiber_timepoint in enumerate(fiber):
            # legacy issue: data was prepped in a flattened way, so reforming 3D monomers datastructure, and will flatten again after when saving
            fiber_timepoint_reshaped = fiber_timepoint.reshape(-1, 3)
            best_fit_theta = find_best_fit_lines_theta_from_fibers(fiber_timepoint_reshaped)
            # for each monomer in the timepoint's fiber, calculate the rotation and map the new position
            for fiber_timepoint_monomer_idx, fiber_timepoint_monomer in enumerate(fiber_timepoint_reshaped):
                # ... calc theta of this monomer
                monomer_theta = np.arctan(fiber_timepoint_monomer[1] / fiber_timepoint_monomer[2])
                if np.isnan(monomer_theta):
                    monomer_theta = 0
                # ... calc difference between highest magnitude theta and this monomer's theta
                monomer_theta_diff = best_fit_theta - monomer_theta
                # ... map new y/z values to this fiber_timepoint_monomer's based on rotation
                new_monomer_y = fiber_timepoint_monomer[1] * np.cos(monomer_theta_diff) - fiber_timepoint_monomer[2] * np.sin(monomer_theta_diff)
                new_monomer_z = fiber_timepoint_monomer[1] * np.sin(monomer_theta_diff) + fiber_timepoint_monomer[2] * np.cos(monomer_theta_diff)
                fiber_timepoint_reshaped[fiber_timepoint_monomer_idx] = [fiber_timepoint_monomer[0], new_monomer_y, new_monomer_z]
            # legacy: saving again as a flat array for the rest of the application
            fiber_timepoint_reshaped_flattened = fiber_timepoint_reshaped.reshape(-1)
            fibers_mapped[fiber_idx][fiber_timepoint_idx] = fiber_timepoint_reshaped_flattened
    # return
    return fibers_mapped

# %%
# ### Data Preppers

# --- metrics
repeat_time_metrics_list = ["time", "NON_COPLANARITY", "PEAK_ASYMMETRY", "AVERAGE_PERP_DISTANCE", "TOTAL_FIBER_TWIST", "CALC_BENDING_ENERGY", "CONTOUR_LENGTH"]

# --- load in prepped CSV of subsamples (pre-processed for timepoints, monomers)
def study_subsamples_loader(subsamples_df: pd.DataFrame):
    study_dfs = []
    num_timepoints = 66 # 200 is the subsample size, but we want to focus PCA latent space on data seeing most transformation
    num_monomers = 200
    for sim_name, sim_df in subsamples_df.groupby("simulator"):
        for param_velocity, velocity_df in sim_df.groupby("velocity"):
            fibers = [] # 1 fiber per simulation, so treating "repeats" as fibers
            fibers_metrics = []
            for repeat, repeat_df in velocity_df.groupby("repeat"):
                fiber_timepoints = []
                fiber_timepoints_metrics = []
                for time, time_df in repeat_df.groupby("time"):
                    if len(fiber_timepoints) >= num_timepoints:
                        break
                    fiber_timepoints.append(time_df[["xpos", "ypos", "zpos"]].values.flatten()) # flattening to match data patterns
                    # HACK: we should do this all w/ dataframes but the refactor is painful so going to try keeping metrics in parallel
                    fiber_timepoints_metrics.append(time_df[repeat_time_metrics_list])
                fibers.append(fiber_timepoints) # 1001 timepoints x flattened(50 monomers x 3 dimensions)
                fibers_metrics.append(fiber_timepoints_metrics)
            # --- coordinate scaler (coordinate systems scale differences don't allow shared space)
            param_coordinate_scaler = 1
            if sim_name == "readdy":
                param_coordinate_scaler = 0.001
            # --- append
            df = pd.DataFrame({
                "fibers": [np.array(fibers)], # need as np array for shape property, should be (-1, 600)
                "fibers_metrics": [fibers_metrics],
                "source": sim_name,
                "param_fibers": len(fibers),
                "param_timepoints": num_timepoints,
                "param_fiber_monomers": num_monomers,
                "param_velocity": param_velocity,
                "param_coordinate_scaler": param_coordinate_scaler
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
    num_pca_features = int(study_df.param_fiber_monomers) * 3
    fibers = study_df.fibers[0].copy() if align == False else align_fibers(study_df.fibers[0])
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
        pca_df["param_velocity"] = study_dfs[study_idx].param_velocity
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
        pacmap_df["source"] = study_df.source
        pacmap_df["param_velocity"] = study_df.param_velocity
        pacmap_results.append(pacmap_df)
        start = end
    return pacmap_results, pacmap_space


# %%
# ### Plotter Setup

# --- for single simulation
def plot_study_df(analysis_df: pd.DataFrame, pca_space: PCA, study_df: pd.DataFrame, title: str, figsize=8):
    """
    Plot a PCA analysis dataframe
    """
    for metric in repeat_time_metrics_list:
        # TODO: subplot setup
        fig, ax = plt.subplots(figsize=(figsize, figsize) if type(figsize) == int else figsize)
        # --- List of parameters or conditions under which the PCA was run.
        ax.set_xlabel(f"PC1{f': {pca_space.explained_variance_ratio_[0]}' if pca_space != None else ''}", loc="left")
        ax.set_ylabel(f"PC2{f': {pca_space.explained_variance_ratio_[1]}' if pca_space != None else ''}", loc="bottom")
        ax.set_title(title + f",Metric={metric}", fontsize=10)
        # --- setup points + skip counter
        num_pc1_points = int(analysis_df['principal component 1'].shape[0])
        num_timepoints = analysis_df.time.iloc[-1] # this is a series. each monomer point is saved with a time position. we grab the last value to know what the last monomer timepoint is
        # --- flattened metrics (bc points are flattened when processed ex: 5 fiber x 200 timepoint becomes 1000 points)
        flattened_metrics = [repeat_time[metric].values[0] for repeat_list in study_df.fibers_metrics[0] for repeat_time in repeat_list]
        metric_min = min(flattened_metrics)
        metric_max = max(flattened_metrics)
        # --- compile flat arr of colors (feels weird but works)
        pc1_color_lists = []
        for fiber_idx in range(num_pc1_points // num_timepoints): # segment by fiber
            pc1_color_lists.append(color_list_generator(num_timepoints, fiber_idx, c1_override="black" if metric != "time" else None, c2_override="black" if metric != "time" else None))
        pc1_color_list = [c for cl in pc1_color_lists for c in cl] # flattens
        # --- scatter
        for i in range(num_pc1_points):
            ax.scatter(
                analysis_df.loc[i, "principal component 1"],
                analysis_df.loc[i, "principal component 2"],
                c=[pc1_color_list[i]],
                alpha=normalize(flattened_metrics[i], metric_min, metric_max),
                s=70,
            )
        plt.show()

# --- for many simulations overlapping
def plot_study_dfs(analysis_dfs: List[pd.DataFrame], pca_space: PCA, study_dfs: pd.DataFrame, title: str, figsize=8):
    """
    Plot multiple analysis dataframes atop each other. Increment colors by simulator count
    """
    for metric in repeat_time_metrics_list:
        fig, ax = plt.subplots(figsize=(figsize, figsize) if type(figsize) == int else figsize)
        # --- List of parameters or conditions under which the PCA was run.
        ax.set_xlabel(f"PC1{f': {pca_space.explained_variance_ratio_[0]}' if pca_space != None else ''}", loc="left")
        ax.set_ylabel(f"PC2{f': {pca_space.explained_variance_ratio_[1]}' if pca_space != None else ''}", loc="bottom")
        ax.set_title(title + f",Metric={metric}", fontsize=10)
        legend_handles = []
        # --- for each simulation, plot..
        for analysis_idx, analysis_df in enumerate(analysis_dfs):
            # --- setup points + skip counter
            num_pc1_points = int(analysis_df['principal component 1'].shape[0])
            num_timepoints = analysis_df.time.iloc[-1] # this is a series. each monomer point is saved with a time position. we grab the last value to know what the last monomer timepoint is
            # --- flattened metrics (bc points are flattened when processed ex: 5 fiber x 200 timepoint becomes 1000 points)
            flattened_metrics = [repeat_time[metric].values[0] for repeat_list in study_dfs[analysis_idx].fibers_metrics[0] for repeat_time in repeat_list]
            metric_min = min(flattened_metrics)
            metric_max = max(flattened_metrics)
            # --- create a list for colors/markers
            pc1_color_lists = []
            pc1_marker_lists = []
            for fiber_idx in range(num_pc1_points // num_timepoints):
                if metric == "time":
                    pc1_color_lists.append(color_list_generator(num_timepoints, analysis_idx)) # all fibers for this sim will be same color, but we'll have opacity shift
                    pc1_marker_lists.append(marker_list_generator(num_timepoints, fiber_idx)) # for each fiber, fill a list of symbol tags
                else:
                    pc1_color_lists.append(color_list_generator(num_timepoints, analysis_idx, c1_override="black" if metric != "time" else None, c2_override="black" if metric != "time" else None)) # all fibers for this sim will be same color, but we'll have opacity shift
                    pc1_marker_lists.append(marker_list_generator(num_timepoints, analysis_idx)) # for each fiber, fill a list of symbol tags
            # --- scatter (flatten the lists so they pair with the scattered points)
            pc1_point_colors = [c for cl in pc1_color_lists for c in cl]
            pc1_point_markers = [m for ml in pc1_marker_lists for m in ml]
            for i in range(num_pc1_points):
                if i == 0:
                    if metric == "time":
                        legend_handles.append(mpatches.Patch(color=color_list[analysis_idx], label=f"{analysis_df.source[0]}/{analysis_df.param_velocity[0]}"))
                    else:
                        legend_handles.append(mpatches.Patch(color="white", label=f"[{marker_list[analysis_idx % len(marker_list)]}] {analysis_df.source[0]}/{analysis_df.param_velocity[0]}"))
                ax.scatter(
                    analysis_df.loc[i, "principal component 1"],
                    analysis_df.loc[i, "principal component 2"],
                    c=[pc1_point_colors[i]],
                    alpha=normalize(flattened_metrics[i], metric_min, metric_max),
                    s=50,
                    marker=pc1_point_markers[i],
                )
        plt.legend(handles=legend_handles) # using handles so we can control how many legend items appear
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
            # --- setup PC1/2 vals for analysis (appending source/params to refence for plot legend labels)
            df = pd.DataFrame({
                "label": label,
                "values": [
                    dict(
                        input=[pc_mean - pc_std * 2, 0] if idx == 0 else [0, pc_mean - pc_std * 2],
                        fiber=None,
                        param_velocity=pca_transform_dataframe.param_velocity[0],
                        source=pca_transform_dataframe.source[0]),
                    dict(
                        input=[pc_mean - pc_std, 0] if idx == 0 else [0, pc_mean - pc_std],
                        fiber=None,
                        param_velocity=pca_transform_dataframe.param_velocity[0],
                        source=pca_transform_dataframe.source[0]),
                    dict(
                        input=[pc_mean, 0] if idx == 0 else [0, pc_mean],
                        fiber=None,
                        param_velocity=pca_transform_dataframe.param_velocity[0],
                        source=pca_transform_dataframe.source[0]),
                    dict(
                        input=[pc_mean + pc_std, 0] if idx == 0 else [0, pc_mean + pc_std],
                        fiber=None,
                        param_velocity=pca_transform_dataframe.param_velocity[0],
                        source=pca_transform_dataframe.source[0]),
                    dict(
                        input=[pc_mean + pc_std * 2, 0] if idx == 0 else [0, pc_mean + pc_std * 2],
                        fiber=None,
                        param_velocity=pca_transform_dataframe.param_velocity[0],
                        source=pca_transform_dataframe.source[0]),
                ],
                # feels duplicative, but helpful to have at different levels for different contexts :/
                "source": pca_transform_dataframe.source[0]
            })
            # --- invert transform fibers
            for idx, val in enumerate(df['values']):
                projected_fiber = pca_space.inverse_transform([val['input'][0], val['input'][1]]) # duplicative, but just want to make clear the interface
                df['values'][idx]['fiber'] = projected_fiber.reshape(-1, 3) # transformation is a flat
            # --- append to set
            pca_sets.append(df)
    return pca_sets

def plot_inverse_transform_pca(pca_sets: List[pd.DataFrame], title_prefix: str, color_by: str = None):
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
    for idx_pca_comp, pca_set_by_component in enumerate(pca_sets_by_comp):
        ax = fig.add_subplot(1, 2, idx_pca_comp + 1, projection='3d')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f"[{title_prefix}] {pca_set_by_component[0]['label'][0]}", fontsize=14)
        for idx_pca, pca_set in enumerate(pca_set_by_component):
            for idx_val, val in enumerate(pca_set['values']):
                c = color_list[source_to_idx(pca_set.source[0]) if color_by == "source" else (idx_val % len(color_list))] # this function is confusing as shit 
                x, y, z = zip(*val['fiber'])  # TIL you can pass in a tuple of values, and this resolves the legend issue + visualization
                alpha = min(max(0.1, (1 / (idx_pca + 1))), 0.8) # ensuring a floor of 0.1 and max of 0.8 to prevent dominating lines
                ax.plot(x, y, z, c=c, alpha=alpha, label=f"[{pca_set.source[0]}] {str(val['input'][0])[0:6]}, {str(val['input'][1])[0:6]}")
        ax.legend()
    plt.tight_layout()
    plt.show()


# %%
# # Histograms
def plot_pca_histogram(pca_sets: List[pd.DataFrame]):
    sources = list(set([pca_set.source[0] for pca_set in pca_sets]))
    bin_max=1.5
    bin_size=0.15
    fig, axs = plt.subplots(len(sources) * 2, figsize=(10, 10 * len(sources)))
    idx_plt = 0
    for source_idx, source in enumerate(sources):
        source_pca_sets = list(filter(lambda pca: pca.source[0] == source, pca_sets))
        # for each simulator/source, plot a PC1/PC2
        for pc in ["principal component 1", "principal component 2"]:
            axs[idx_plt].set_title(f"{source} - {pc}")
            # --- defaults for binns
            bins = np.arange(-bin_max, bin_max, bin_size)
            num_bins = int(2 * bin_max / bin_size)
            # --- loop over each dataset
            dataset_midpoints = []
            labels = []
            for pca_data in source_pca_sets:
                binned_data = pd.cut(pca_data[pc], bins)
                midpoints = binned_data.apply(lambda x: x.mid).dropna()
                dataset_midpoints.append(midpoints)
                labels.append(f"{source}/{pca_data.param_velocity[0]}")
            # --- plot
            axs[idx_plt].hist(dataset_midpoints,
                bins=num_bins,
                stacked=True,
                label=labels,
                color=color_list[source_idx] if len(dataset_midpoints) == 1 else None, # if many datasets combined, let it auto do colors
                alpha=1 if pc == "principal component 1" else 0.65) # doing this to make it visually easy to see PC1/PC2
            axs[idx_plt].set_ylabel("Frequency")
            axs[idx_plt].legend()
            idx_plt += 1
    # --- graph
    plt.tight_layout()
    plt.show()


# %%
# Data Loading: Combiled Subsample
subsamples_filename = "combined_actin_compression_metrics_all_velocities_and_repeats_subsampled.csv"
subsamples_df = pd.read_csv(f"{data_directory}/dataframes/{subsamples_filename}")
study_dfs = study_subsamples_loader(subsamples_df)


# %%
# # PLOT PCAS: BY VELOCITY
for velocity in np.sort(subsamples_df['velocity'].unique()):
    study_dfs_to_plot = list(filter(lambda df: df.param_velocity.values[0] == velocity, study_dfs))
    print(f"Plotting {len(study_dfs_to_plot)} studies with velocity = {velocity}...")

    # PCA
    for st_df in study_dfs_to_plot:
        source = st_df["source"].values[0]
        print(f"Plotting '{source}' PCA...")
        # [pca_unaligned_df], pca_unaligned_space = get_study_pca_dfs([st_df], align=False)
        [pca_aligned_df], pca_aligned_space = get_study_pca_dfs([st_df], align=True)
        # --- pca histogram
        # plot_pca_histogram(pca_sets=[pca_aligned_df])
        # --- pca scatter plot
        plot_study_df(analysis_df=pca_aligned_df, pca_space=pca_aligned_space, study_df=st_df, title=f"{source} (PCA): Aligned=True,Velocity={float(velocity)}", figsize=(8, 8))
        # --- pca inverse transforms: unaligned
        # pca_unaligned_component_dists = calc_pca_component_distributions(pca_space=pca_unaligned_space, pca_transform_dataframes=[pca_unaligned_df])
        # plot_inverse_transform_pca(pca_sets=pca_unaligned_component_dists, title_prefix=f"{source} / {float(velocity)} / Unaligned")
        # --- pca inverse transforms: aligned
        pca_component_dists = calc_pca_component_distributions(pca_space=pca_aligned_space, pca_transform_dataframes=[pca_aligned_df])
        plot_inverse_transform_pca(pca_sets=pca_component_dists, title_prefix=f"{source} / {float(velocity)} / Aligned")

    # PCA: ALL SIMS
    print(f"Plotting All PCA...")
    # --- pca scatter plot
    pca_aligned_dfs, pca_aligned_space = get_study_pca_dfs(study_dfs_to_plot, align=True)
    plot_study_dfs(analysis_dfs=pca_aligned_dfs, pca_space=pca_aligned_space, study_dfs=study_dfs_to_plot, title=f"ALL (PCA): Aligned=True,Velocity={float(velocity)}", figsize=(8, 8))
    # --- pca histograms
    # plot_pca_histogram(pca_sets=pca_aligned_dfs)
    # --- pca inverse transforms (apparently this isn't helpful)


# %%
# # PLOT PCAS BY SIM/SOURCE
sources_to_plot = list(set([pca_set.source[0] for pca_set in study_dfs]))

for source in sources_to_plot:
    study_dfs_to_plot = list(filter(lambda pca: pca.source[0] == source, study_dfs))
    print(f"Plotting {len(study_dfs_to_plot)} studies with source = {source}...")
    # --- pca
    pca_aligned_dfs, pca_aligned_space = get_study_pca_dfs(study_dfs_to_plot, align=True)
    # --- pca: plot
    plot_study_dfs(analysis_dfs=pca_aligned_dfs, pca_space=pca_aligned_space, study_dfs=study_dfs_to_plot, title=f"{source} (PCA): Aligned=True,Velocity=All", figsize=(8, 8))
    # --- histograms
    # plot_pca_histogram(pca_sets=pca_aligned_dfs)
    # --- inverse transform
    pca_component_dists = calc_pca_component_distributions(pca_space=pca_aligned_space, pca_transform_dataframes=pca_aligned_dfs)
    plot_inverse_transform_pca(pca_sets=pca_component_dists, title_prefix=f"{source}")


# %%
# # PLOT PCAS BY ALL
print(f"Plotting all {len(study_dfs)} studies...")
# --- pca
pca_aligned_dfs_all, pca_aligned_space_all = get_study_pca_dfs(study_dfs, align=True)
# --- pca: plot
# plot_study_dfs(analysis_dfs=pca_aligned_dfs_all, pca_space=pca_aligned_space_all, study_dfs=study_dfs, title=f"ALL (PCA): Aligned=True,Velocity=All", figsize=(8, 8))
# --- histograms
# plot_pca_histogram(pca_sets=pca_aligned_dfs_all)
# --- inverse transform
pca_component_dists_all = calc_pca_component_distributions(pca_space=pca_aligned_space_all, pca_transform_dataframes=pca_aligned_dfs_all)
plot_inverse_transform_pca(pca_sets=pca_component_dists_all, title_prefix="ALL")
plot_inverse_transform_pca(pca_sets=pca_component_dists_all, title_prefix="ALL", color_by="source")

