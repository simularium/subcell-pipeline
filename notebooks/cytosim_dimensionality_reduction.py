# %%
from datetime import date
import math
from typing import List
import numpy as np
import os
from pydash import snake_case
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from scipy.optimize import minimize


# %%
# ### Init Fns/Helpers
# --- directories/files
dir_data = "../data/"
dir_figures = "../data/figures/"

if not os.path.exists(dir_data):
    os.makedirs(dir_data)
if not os.path.exists(dir_figures):
    os.makedirs(dir_figures)

def figure_filename(strs: List[str]) -> str:
    today_str = date.today().strftime("%Y-%m-%d")
    # create a directory for today's date
    if not os.path.exists(dir_figures + snake_case(today_str)):
        os.makedirs(dir_figures + snake_case(today_str))
    return dir_figures + snake_case(today_str) + "/" + snake_case(today_str + "_" + "_".join(strs)) + ".png"

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
    }.get(source, 6)

def velocity_to_idx(velocity: str) -> int:
    velocitry_str = str(velocity)
    return {
        "4.7": 0,
        "15.0": 1,
        "47.0": 2,
        "150.0": 3,
    }.get(velocitry_str, 0)

def velocity_to_alpha(velocity: str) -> int:
    velocitry_str = str(velocity)
    return {
        "4.7": 0.2,
        "15.0": 0.4,
        "47.0": 0.7,
        "150.0": 1,
    }.get(velocitry_str, 0)

# --- alignments
def calculate_fiber_rotation(fiber: np.ndarray) -> float:
    """Return radians needed to get to a target angle, using most distant monomer"""
    # calculate monomer distances, and get index of furthest away
    monomers_by_distance = np.sqrt(np.sum(fiber[:,1:] ** 2, axis=1))
    max_distance_monomer_index = np.argmax(monomers_by_distance)
    # get the radians of that monomer using arctan
    angle_rad = np.arctan2(fiber[max_distance_monomer_index, 2], fiber[max_distance_monomer_index, 1])
    # return the diff in radians from 90 deg (pi/2), so all fibers will align vertically
    target_rotation_rad = np.pi / 2 # this is 90 deg, can use 0 to go horizontal
    rotation_angle_rad_to_0 = target_rotation_rad - angle_rad
    return rotation_angle_rad_to_0

def align_fibers(fibers: np.ndarray) -> np.ndarray:
    """Rotationally align the given fibers around the x-axis to the fiber with the greatest magnitude."""
    aligned_fibers = []
    # for each fiber...
    for fiber_idx, fiber in enumerate(fibers.copy()):
        print(f"aligning fiber: {fiber_idx}")
        aligned_fiber = []
        # for each timepoint...
        for fiber_timepoint_idx, fiber_timepoint in enumerate(fiber):
            # print(f"aligning fiber timepoint: {fiber_idx}/{fiber_timepoint_idx}")
            fiber_timepoint_shaped = np.array(fiber_timepoint).reshape(-1, 3)
            # determine rotation angle for fiber's timepoint (how do we get cos0/sin1?)
            rotation_angle_rads = calculate_fiber_rotation(fiber_timepoint_shaped)
            # get rotation matrix (we're ignorning X axis, which is why this is a 2x2)
            c, s = np.cos(-rotation_angle_rads), np.sin(-rotation_angle_rads)
            rotation_matrix = np.array(((c, -s), (s, c)))
            # rotate fiber, and reshape the y/z values (index 1/2)
            fiber_rotated = fiber_timepoint_shaped.copy()
            fiber_rotated[:,1:] = np.dot(fiber_timepoint_shaped[:,1:], rotation_matrix)
            aligned_fiber.append(fiber_rotated.reshape(-1))
        aligned_fibers.append(np.array(aligned_fiber))
    return np.array(aligned_fibers)


# %%
# ### Data Preppers

# --- metrics
repeat_time_metrics_list = ["time", "NON_COPLANARITY", "PEAK_ASYMMETRY", "TOTAL_FIBER_TWIST", "CALC_BENDING_ENERGY", "CONTOUR_LENGTH"]

# --- load in prepped CSV of subsamples (pre-processed for timepoints, monomers)
def study_subsamples_loader(subsamples_df: pd.DataFrame):
    study_dfs = []
    num_timepoints = 200 # 200 is the subsample size, but we want to focus PCA latent space on data seeing most transformation
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
                    try:
                        fiber_timepoints_metrics.append(time_df[repeat_time_metrics_list])
                    except Exception as err:
                        pass # if metrics don't exist, the append errors but just let things continue
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
def prep_study_df_fibers(study_df: pd.DataFrame, align: bool = False):
    """Function for getting a consistently (re)shaped dataset for PCA/PACMAP analysis"""
    # calc dimensions for transformation
    num_pca_samples = int(study_df.param_fibers) * int(study_df.param_timepoints)
    num_pca_features = int(study_df.param_fiber_monomers) * 3
    # align or dont
    fibers = study_df.fibers[0].copy() if align == False else align_fibers(fibers=study_df.fibers[0])
    # normalize scale across simulators
    fibers_scaled = fibers * study_df.param_coordinate_scaler[0] # scale the coordinates for aligning multiple sims
    # reshape
    fibers_scaled_flattened = np.ravel(fibers_scaled)
    pca_dataset = fibers_scaled_flattened.reshape((num_pca_samples, num_pca_features))
    return pca_dataset


# %%
# ### PCA: Setup
def get_study_pca_dfs(study_dfs: List[pd.DataFrame], align: bool) -> [List[pd.DataFrame], PCA]:
    """
    Get a PCA analysis dataframe from a fiber study dataframe.
    """
    pca_space = PCA(n_components=2)
    # --- create our embedding space by fitting the combined data (dataset func flattens/orders our monomer datapoints)
    study_fibers = [prep_study_df_fibers(study_df, align) for study_df in study_dfs]
    combined_prepared_data = np.vstack(study_fibers) # needing to create a 1 dim arr, but errs if I just pass in [study_fibers], TODO: better understand this
    pca_space.fit(combined_prepared_data)
    # --- transform the data
    pca_transformed_study_data = [pca_space.transform(data) for data in study_fibers]
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

def get_pca_distribution_df(dfs: List[pd.DataFrame]) -> List[pd.DataFrame]:
    pc1_maxs = [max(df.loc[:, "principal component 1"]) for df in dfs]
    pc1_mins = [min(df.loc[:, "principal component 1"]) for df in dfs]
    pc2_maxs = [max(df.loc[:, "principal component 2"]) for df in dfs]
    pc2_mins = [min(df.loc[:, "principal component 2"]) for df in dfs]
    pc1_mean = np.mean(pc1_maxs + pc1_mins)
    pc2_mean = np.mean(pc2_maxs + pc2_mins)
    pc1_std = np.std(pc1_maxs + pc1_mins)
    pc2_std = np.std(pc1_maxs + pc1_mins)
    pca_standards_df = [pd.DataFrame(
        data=[[pc1_std * -2, pc2_std * -2],[pc1_std * -1, pc2_std * -1],[pc1_mean, pc2_mean],[pc1_std, pc2_std], [pc1_std * 2, pc2_std * 2]],
        columns=["principal component 1", "principal component 2"],
    )]
    pca_standards_df[0]["source"] = "ALL"
    pca_standards_df[0]["param_velocity"] = "ALL"
    return pca_standards_df


# %%
# ### Plotter Setup

# --- for many simulations overlapping
def plot_study_dfs(analysis_dfs: List[pd.DataFrame], pca_space: PCA, study_dfs: pd.DataFrame, title: str, figsize:int = 8, alpha_by: str = None, color_by: str = None, skip_metrics: bool = False, only_plot_source: str = None):
    """
    Plot multiple analysis dataframes atop each other. Increment colors by simulator count
    """
    for metric in repeat_time_metrics_list:
        # --- skip metrics that aren't time
        if skip_metrics == True and metric != "time":
            continue
        # --- setup fig
        fig, ax = plt.subplots(figsize=(figsize, figsize) if type(figsize) == int else figsize)
        # ax.set_xlim(-0.9, 0.9)
        # ax.set_ylim(-0.9, 0.9)
        # --- List of parameters or conditions under which the PCA was run.
        ax.set_xlabel(f"PC1{f': {pca_space.explained_variance_ratio_[0]}' if pca_space != None else ''}", loc="left")
        ax.set_ylabel(f"PC2{f': {pca_space.explained_variance_ratio_[1]}' if pca_space != None else ''}", loc="bottom")
        ax.set_title(title + f",Metric={metric}", fontsize=10)
        legend_handles = []
        # --- for each simulation, plot..
        for analysis_idx, analysis_df in enumerate(analysis_dfs):
            # --- check if only plotting a set number of sources
            if only_plot_source != None and analysis_df.source[0] != only_plot_source:
                continue
            # --- setup points + skip counter
            num_pc1_points = int(analysis_df['principal component 1'].shape[0])
            num_timepoints = analysis_df.time.iloc[-1] # this is a series. each monomer point is saved with a time position. we grab the last value to know what the last monomer timepoint is
            # --- flattened metrics (bc points are flattened when processed ex: 5 fiber x 200 timepoint becomes 1000 points)
            flattened_metrics = [repeat_time[metric].values[0] for repeat_list in study_dfs[analysis_idx].fibers_metrics[0] for repeat_time in repeat_list]
            metric_min = min(flattened_metrics) if len(flattened_metrics) > 0 else 1
            metric_max = max(flattened_metrics) if len(flattened_metrics) > 0 else 1
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
                # --- per point visualizations
                alpha = normalize(flattened_metrics[i], metric_min, metric_max) if len(flattened_metrics) > 0 else 0.8
                color = [pc1_point_colors[i]]
                marker = pc1_point_markers[i]
                if color_by == "source":
                    color = color_list[source_to_idx(analysis_df.source[0])]
                    marker = marker_list[velocity_to_idx(analysis_df.param_velocity[0])]
                if alpha_by == "param_velocity" and metric == "time":
                    alpha = velocity_to_alpha(analysis_df.param_velocity[0])
                # --- setup legend on first point
                if i == 0:
                    if metric == "time":
                        label = f"{analysis_df.source[0]}/{analysis_df.param_velocity[0]} [{marker}]"
                        legend_handles.append(mpatches.Patch(color=color[0], label=label))
                    else:
                        label = f"{analysis_df.source[0]}/{analysis_df.param_velocity[0]} [{metric}]"
                        legend_handles.append(mpatches.Patch(color=color[0] if color_by == "source" else "white", label=label))
                ax.scatter(
                    analysis_df.loc[i, "principal component 1"],
                    analysis_df.loc[i, "principal component 2"],
                    c=color,
                    alpha=alpha,
                    s=4,
                    marker=marker,
                )
        plt.legend(handles=legend_handles) # using handles so we can control how many legend items appear
        plt.savefig(figure_filename(["pca", title, metric]))
        plt.show()


# %%
# # Inverse Transformations Funcs: Exmaning Differences in Original/Transformed Data
def calc_pca_component_distributions(pca_space: PCA, pca_transform_dataframes: List[pd.DataFrame]) -> List[dict]:
    # print(f"[calc_pca_component_distributions] PCA DFs: {pca_transform_dataframes}")
    pca_sets = []
    # grouping by 
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

def plot_inverse_transform_pca(pca_sets: List[pd.DataFrame], title_prefix: str, color_by: str = None, include_2d: bool = False, include_3d: bool = True):
    # print(f"[plot_inverse_transform_pca] PCA Sets: {len(pca_sets)}")
    # 2D Layout
    if include_2d != False:
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
        plt.savefig(figure_filename(["inverse_transform_2d", title_prefix]))
        plt.show()
    # 3D plot of fiber positions
    if include_3d != False:
        # hacky, but w/e it works well enough for now
        elev=10
        for angle in [90, 60, 30, 0]:
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
                ax.set_title(f"[{title_prefix},Deg={angle}] {pca_set_by_component[0]['label'][0]}", fontsize=14)
                legend_handles = []
                legend_labels = []
                for idx_pca_val, pca_set_val in enumerate(pca_set_by_component):
                    for idx_inverse_val, inverse_val in enumerate(pca_set_val['values']):
                        c = color_list[idx_inverse_val % len(color_list)]
                        x, y, z = zip(*inverse_val['fiber'])  # TIL you can pass in a tuple of values, and this resolves the legend issue + visualization
                        if color_by == "source":
                            c = color_list[source_to_idx(inverse_val['source'])]
                            label = f"[{pca_set_val.source[0]}]" # the pc1/2 values aren't helpful, but seeing velocity distinctions are
                            legend_labels.append(label)
                        if color_by == "param_velocity":
                            c = color_list[velocity_to_idx(str(inverse_val['param_velocity']))]
                            label = f"[{pca_set_val.source[0]}] {str(inverse_val['param_velocity'])}" # the pc1/2 values aren't helpful, but seeing velocity distinctions are
                            legend_labels.append(label)
                        if color_by == "distribution":
                            c = color_fader("blue", "red", mix=idx_inverse_val / 5)
                            legend_labels = ["-2σ", "-1σ", "mean", "+1σ", "+2σ"]
                        alpha = 0.9 # min(max(0.1, (1 / (idx_pca_val + 1))), 0.8) # ensuring a floor of 0.1 and max of 0.8 to prevent dominating lines
                        line, = ax.plot(x,y,z,c=c,alpha=alpha)
                        legend_handles.append(line)
                ax.legend(handles=legend_handles, labels=legend_labels)
                # --- render with angle to get better sense of transform
                ax.view_init(elev=elev, azim=angle)
            plt.tight_layout()
            plt.savefig(figure_filename(["inverse_transform_3d", title_prefix, f"{angle}deg"]))
            plt.show()

# --- visualize aligned fibers to sanity check our underlying data is right (only grab first few time points of each fiber)
def plot_subsample_fibers_dfs(study_dfs: List[pd.DataFrame], align: bool = False):
    # DATA
    plot = []
    color_increment = 0
    for study_df in study_dfs:
        study_df_fibers_slice = study_df.fibers[0]
        study_df_fibers = align_fibers(fibers=study_df_fibers_slice) if align == True else study_df_fibers_slice
        for fiber_idx, fiber in enumerate(study_df_fibers):
            # if fiber_idx > 1: continue
            for idx_fiber_timepoint, fiber_timepoint in enumerate(fiber.reshape(-1, 200, 3)): # seems to have 66 loops at 200/3
                # if idx_fiber_timepoint > 20: continue
                x, y, z = zip(*fiber_timepoint.reshape(-1, 3))
                plot.append([x, y, z, 0.6, color_list[color_increment % len(color_list)]])
            color_increment += 1
    # PLOT
    for proj_type in ["persp", "ortho"]:
        fig = plt.figure(figsize=(18, 10))
        # do a subplot for each angle
        for idx_angle, angle in enumerate([0, 90]):
            ax = fig.add_subplot(1, 2, idx_angle + 1, projection='3d')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f"[Subsamples Aligned={align},Deg={angle}]", fontsize=14)
            # --- for each plot point, plot
            for p in plot:
                ax.plot(p[0], p[1], p[2], alpha=p[3], c=p[4])
            # --- render with angle to get better sense of transform
            ax.set_proj_type(proj_type)
            ax.view_init(elev=0 if proj_type == "ortho" else 20, azim=angle if proj_type == "ortho" else 45 + angle)
        plt.tight_layout()
        plt.savefig(figure_filename(["subsample_fibers_3d", f"aligned={align}" f"proj_type={proj_type}", f"{angle}deg"]))
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
    plt.savefig(figure_filename(["histogram"]))
    plt.show()


# %%
# Data Loading: Combiled Subsample
# subsamples_filename = "combined_actin_compression_dataset_all_velocities_and_repeats_subsampled.csv"
subsamples_filename = "combined_actin_compression_metrics_all_velocities_and_repeats_subsampled_with_metrics.csv"
subsamples_df = pd.read_csv(f"{dir_data}dataframes/{subsamples_filename}")
study_dfs = study_subsamples_loader(subsamples_df)


# %%
# # PLOT

print(f"Plotting all {len(study_dfs)} studies...")

for align in [True]:
    # --- verify fibers alignment
    plot_subsample_fibers_dfs(study_dfs=study_dfs[-2:], align=align)

for align in [True]:
    # --- pca
    pca_aligned_dfs_all, pca_aligned_space_all = get_study_pca_dfs(study_dfs, align=align)
    # --- inverse transform: min/maxs
    pca_distribution_df = get_pca_distribution_df(pca_aligned_dfs_all)
    pca_component_dists_standards = calc_pca_component_distributions(pca_space=pca_aligned_space_all, pca_transform_dataframes=pca_distribution_df)
    plot_inverse_transform_pca(pca_sets=pca_component_dists_standards, title_prefix=f"ALL: Aligned={align}", color_by="distribution")
    # --- inverse transform: all fibers
    # pca_component_dists_all = calc_pca_component_distributions(pca_space=pca_aligned_space_all, pca_transform_dataframes=pca_aligned_dfs_all)
    # plot_inverse_transform_pca(pca_sets=pca_component_dists_all, title_prefix="ALL", color_by="source")
    # --- pca: plot
    plot_study_dfs(analysis_dfs=pca_aligned_dfs_all, pca_space=pca_aligned_space_all, study_dfs=study_dfs, title=f"ALL (PCA): Aligned={align},Velocity=All", figsize=(11, 11), color_by="source", skip_metrics=False)
    plot_study_dfs(analysis_dfs=pca_aligned_dfs_all, pca_space=pca_aligned_space_all, study_dfs=study_dfs, title=f"cytosim (PCA): Aligned=True,Velocity=All", figsize=(11, 11), color_by="source", alpha_by="param_velocity", skip_metrics=True, only_plot_source="cytosim")
    plot_study_dfs(analysis_dfs=pca_aligned_dfs_all, pca_space=pca_aligned_space_all, study_dfs=study_dfs, title=f"readdy (PCA): Aligned=True,Velocity=All", figsize=(11, 11), color_by="source", alpha_by="param_velocity", skip_metrics=True, only_plot_source="readdy")
    # --- histograms
    # plot_pca_histogram(pca_sets=pca_aligned_dfs_all)
