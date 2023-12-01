# %%
import math
import numpy as np
import pacmap
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.spatial.transform import Rotation
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# %%
# ### Init Fns/Helpers
# --- data refs
data_directory = "../data"

# --- math funcs
# Get RMSD between 2 curves
def rmsd(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
    return np.sqrt(((((vec1 - vec2) ** 2)) * 3).mean())

# --- color funcs
def colorFader(c1, c2, mix=0) -> str:
    """Get a colour gradient to represent time"""
    c1 = np.array(mpl.colors.to_rgba(c1))
    c2 = np.array(mpl.colors.to_rgba(c2))
    return mpl.colors.to_rgba((1 - mix) * c1 + mix * c2)

def color_list_generator (num: int, idx: int = 0) -> list[str]:
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
    return [colorFader(c1, c2, i / num) for i in range(num)]

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
# ### Study Data: Microscope

# The shape of all_actin_vectors_all_time is 101 timepoints * 5 fibers * 4 simulations * 3 dimensions * 500 monomer points
raw_microscopy_array = np.loadtxt(f"{data_directory}/microscope_dimensionality_positions.txt")
# go from raw data and reshape from (101, 4, 5, 3, 500) to (4, 5, 101, 500, 3)
raw_microscopy_fibers_array = raw_microscopy_array.reshape(101, 4, 5, 1500) # 3 * 500 combining monomers/dimensions bc align code expects that
raw_microscope_study_fibers_array = raw_microscopy_fibers_array.transpose(1, 2, 0, 3) # numbers refer to prior positions -> (sims, fibers, timepoints, dims/monos)


# %%
# each batch of fibers has a velocity param. as we loop through, we'll reference the param below. This is unfortunately a hard coded situation
microscope_study_params = [
    { "velocity": 4.7 }, # run 1
    { "velocity": 15 }, # run 2
    { "velocity": 47 }, # run 3
    { "velocity": 150 }, # run 4
]
microscope_study_dataframes = []
for i, fibers in enumerate(raw_microscope_study_fibers_array):
    param_fiber_monomers = 500
    param_timepoints = 101
    df = pd.DataFrame({
        'fibers': [fibers], # fibers -> timepoints -> monomers (dims)
        'source': 'microscopy', # when we ingest other datasets, this will change to simulator name
        'param_fibers': 5,
        'param_fiber_monomers': param_fiber_monomers,
        'param_timepoints': param_timepoints, # at each time point, expect the monomer positions
        'param_dimensions': 3, # seems obvious i know, but i want the named variable for clarity in other operations
        'param_velocity': microscope_study_params[i].get("velocity")
    })
    microscope_study_dataframes.append(df)

# %%
# ### Study Data: Simulation Cytosim

# cytosim_study_dataframes = []
# cytosim_df = pd.read_csv(f"{data_directory}/dataframes/cytosim_actin_compression_all_velocities_and_repeats.csv", index_col=0)

# # %%
# print(cytosim_df.shape)
# print(cytosim_df.keys())


# %%
# ### Study Data: Simulation Readdy

readdy_df = pd.read_csv(f"{data_directory}/dataframes/readdy_actin_compression_all_velocities_and_repeats.csv", index_col=0)
readdy_df = readdy_df.rename(columns={"id": "fiber_id"})

# %%
readdy_study_dataframes = []
for param_velocity, velocity_df in readdy_df.groupby('velocity'):
    param_fiber_monomers = 50
    param_timepoints = 1001
    fibers = [] # 1 fiber per simulation, so treating 'repeats' as fibers
    for repeat, repeat_df in velocity_df.groupby('repeat'):
        fiber_timepoints = []
        for time, time_df in repeat_df.groupby('time'):
            fiber_timepoints.append(time_df[['xpos', 'ypos', 'zpos']].values.flatten()) # flattening to match data patterns
        fibers.append(fiber_timepoints) # 1001 timepoints x flattened(50 monomers x 3 dimensions)
    # --- append "study"
    df = pd.DataFrame({
        'fibers': [fibers],
        'source': 'readdy', # when we ingest other datasets, this will change to simulator name
        'param_fibers': len(fibers),
        'param_timepoints': param_timepoints, # at each time point, expect the monomer positions
        'param_fiber_monomers': param_fiber_monomers,
        'param_dimensions': 3, # seems obvious i know, but i want the named variable for clarity in other operations
        'param_velocity': param_velocity
    })
    readdy_study_dataframes.append(df)


# %%
# ### Data Preppers

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

def get_study_fibers_pca_df(study_df: pd.DataFrame, align: bool) -> pd.DataFrame:
    """
    Get a PCA analysis dataframe from a fiber study dataframe.
    """
    # --- compute
    pca_dataset = get_study_df_analysis_dataset(study_df, align)
    pca = PCA(n_components=2)
    pca_samples_components = pca.fit_transform(pca_dataset)
    # --- setup dataframe for plotting
    pca_df = pd.DataFrame(
        data=pca_samples_components,
        columns=["principal component 1", "principal component 2"],
    )
    pca_df["label"] = study_df.param_velocity
    pca_df["time"] = int(study_df.param_timepoints)
    # --- return
    return pca_df


def plot_study_fibers_pca(pca_df: pd.DataFrame, study_df: pd.DataFrame, title: str, figsize=6):
    """
    Plot a PCA analysis dataframe
    """
    fig, ax = plt.subplots(figsize=(figsize, figsize))
    # --- List of parameters or conditions under which the PCA was run.
    ax.set_xlabel("PC1", loc="left")
    ax.set_ylabel("PC2", loc="bottom")
    ax.set_title(title, fontsize=10)
    # --- compile flat arr of colors (505 points for pc1, which is 101 chunks. this feels a little weird but it works)
    num_pc1_points = int(pca_df['principal component 1'].shape[0])
    num_fiber_points = int(study_df.param_timepoints)
    pc1_color_lists = []
    for idx in range(num_pc1_points // num_fiber_points):
        # incrimenting idx by 1 per set of 101 points across the full count of pc1 points
        pc1_color_lists.append(color_list_generator(num_fiber_points, idx))
    pc1_color_list = [c for cl in pc1_color_lists for c in cl] # flattens
    # --- scatter
    for i in range(num_pc1_points):
        ax.scatter(
            pca_df.loc[i, "principal component 1"],
            pca_df.loc[i, "principal component 2"],
            c=[pc1_color_list[i]],
            s=70,
        )


# %%
# ### PACMAP: Setup

def get_study_fibers_pacmap_df(study_df: pd.DataFrame, align: bool) -> pd.DataFrame:
    """
    Get a PACMAP analysis dataframe from a fiber study dataframe.     
    """
    # --- compute
    pacmap_dataset = get_study_df_analysis_dataset(study_df, align)
    embedding = pacmap.PaCMAP(n_components=2, n_neighbors=int(study_df.param_fibers))
    pacmap_samples_components = embedding.fit_transform(pacmap_dataset, init="pca")
    # --- setup dataframe for plotting
    pacmap_df = pd.DataFrame(
        data=pacmap_samples_components,
        columns=["principal component 1", "principal component 2"],
    )
    pacmap_df["label"] = study_df.param_velocity
    pacmap_df["time"] = int(study_df.param_timepoints) # range(num_pacmap_samples)
    # --- return
    return pacmap_df


def plot_study_fibers_pacmap(pacmap_df: np.ndarray, study_df: pd.DataFrame, title: str, figsize=6):
    """
    Plot a PACMAP analysis dataframe
    """
    fig, ax = plt.subplots(figsize=(figsize, figsize))
    # --- List of parameters or conditions under which the PCA was run.
    ax.set_xlabel("PC1", loc="left")
    ax.set_ylabel("PC2", loc="bottom")
    ax.set_title(title, fontsize=10)
    # --- compile flat arr of colors (505 points for pc1, which is 101 chunks. this feels a little weird but it works)
    num_pc1_points = int(pacmap_df['principal component 1'].shape[0])
    num_fiber_points = int(study_df.param_timepoints)
    pc1_color_lists = []
    for idx in range(num_pc1_points // num_fiber_points):
        # incrimenting idx by 1 per set of 101 points across the full count of pc1 points
        pc1_color_lists.append(color_list_generator(num_fiber_points, idx))
    pc1_color_list = [c for cl in pc1_color_lists for c in cl] # flattens
    # --- scatter
    for i in range(num_pc1_points):
        ax.scatter(
            pacmap_df.loc[i, "principal component 1"],
            pacmap_df.loc[i, "principal component 2"],
            c=[pc1_color_list[i]],
            s=70,
        )


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


# %%
# # Plot: Legend for Color Transitions/Steps
microscope_timepoint_step_range = int(microscope_study_dataframes[0].param_timepoints) # grabbing a sim timepoint count for sake of a legend
plot_timepoint_sepectrum(color_list_generator(microscope_timepoint_step_range), "Simulation Time Intervals")


# %%
# # Plot: Fibers - Microscopy

for idx, microscope_study_df in enumerate(microscope_study_dataframes):
    print(f"Plotting microscope_study_df #{idx}...")
    # PCA
    # --- not aligned
    pca_df = get_study_fibers_pca_df(microscope_study_df, align=False)
    plot_study_fibers_pca(pca_df, microscope_study_df, f"PAC: Aligned={False},Velocity={float(microscope_study_df.param_velocity)}", figsize=4)
    # # --- aligned (has no impact on PCA)
    # PACMAP
    # --- not aligned
    pacmap_df = get_study_fibers_pacmap_df(microscope_study_df, align=False)
    plot_study_fibers_pacmap(pacmap_df, microscope_study_df, f"PaCMAP: Aligned={False},Velocity={float(microscope_study_df.param_velocity)}", figsize=4)
    # --- aligned
    pacmap_aligned_df = get_study_fibers_pacmap_df(microscope_study_df, align=True)
    plot_study_fibers_pacmap(pacmap_aligned_df, microscope_study_df, f"PaCMAP: Aligned={True},Velocity={float(microscope_study_df.param_velocity)}", figsize=4)


# %%
# # Plot: Fibers - ReaDDy

for idx, readdy_study_df in enumerate(readdy_study_dataframes):
    print(f"Plotting readdy_study_df #{idx}...")
    # PCA
    # --- not aligned
    pca_df = get_study_fibers_pca_df(readdy_study_df, align=False)
    plot_study_fibers_pca(pca_df, readdy_study_df, f"PAC: Aligned={False},Velocity={float(readdy_study_df.param_velocity)}", figsize=4)
    # --- aligned (TODO)
    # PACMAP
    # --- not aligned
    pacmap_df = get_study_fibers_pacmap_df(readdy_study_df, align=False)
    plot_study_fibers_pacmap(pacmap_df, readdy_study_df, f"PaCMAP: Aligned={False},Velocity={float(readdy_study_df.param_velocity)}", figsize=4)

