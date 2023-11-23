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
def align(fibers: np.ndarray) -> np.ndarray:
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
# ### Run Params
NUM_TIMEPOINTS = 101 # what's the unit of time?
NUM_FIBERS = 5
NUM_SIMULATIONS = 4
NUM_MONOMER_DIMS = 3
NUM_MONOMERS = 500
NUM_FIBERS_BY_SIMULATION_BY_TIME_POINTS = NUM_TIMEPOINTS * NUM_FIBERS * NUM_SIMULATIONS # 101 * 5 * 4
NUM_MONOMERS_WITH_DIMS = NUM_MONOMER_DIMS * NUM_MONOMERS

# Param changing between simulations: velocity
SIMULATION_1_VELOCITY = 4.7
SIMULATION_2_VELOCITY = 15
SIMULATION_3_VELOCITY = 47
SIMULATION_4_VELOCITY = 150


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

plot_timepoint_sepectrum(color_list_generator(NUM_TIMEPOINTS), "Simulation Step")


# %%
# ### Simulations

# The shape of all_actin_vectors_all_time is 101 timepoints * 5 fibers * 4 simulations * 3 dimensions * 500 monomer points
raw_microscopy_array = np.loadtxt(f"{data_directory}/cytosim_dimensionality_positions.txt")
# go from raw data and reshape from (101, 4, 5, 3, 500) to (4, 5, 101, 500, 3)
fibers_array = raw_microscopy_array.reshape(101, 4, 5, 1500) # 3 * 500 combining monomers/dimensions bc align code expects that
simulation_fibers_array = fibers_array.transpose(1, 2, 0, 3) # numbers refer to prior positions
simulation_fibers_params = [
    { "velocity": SIMULATION_1_VELOCITY },
    { "velocity": SIMULATION_2_VELOCITY },
    { "velocity": SIMULATION_3_VELOCITY },
    { "velocity": SIMULATION_4_VELOCITY },
]

simulation_dataframes = []
for i, fibers in enumerate(simulation_fibers_array):
    df = pd.DataFrame({
        'fibers': [fibers],
        # --- break down of vars
        'num_fibers': 5,
        'num_timepoints': 101,
        'num_monomers': 500,
        'num_dimensions': 3, # seems obvious i know, but i want the named variable for clarity in other operations
        # --- simulation params
        'param_velocity': simulation_fibers_params[i].get("velocity")
    })
    simulation_dataframes.append(df)

def get_sim_df_analysis_dataset(sim_df: pd.DataFrame, aligned: bool = False):
    num_pca_samples = int(sim_df.num_fibers) * int(sim_df.num_timepoints)
    num_pca_features = int(sim_df.num_monomers) * int(sim_df.num_dimensions)
    fibers = sim_df.fibers[0] if aligned == False else align(sim_df.fibers[0])
    simulation_fibers_flattened = np.ravel(fibers)
    pca_dataset = simulation_fibers_flattened.reshape((num_pca_samples, num_pca_features))
    return pca_dataset


# %%
# ### PCA: Setup

def get_simulation_fibers_pca_df(sim_df: pd.DataFrame, aligned: bool) -> pd.DataFrame:
    """
    Get a PCA analysis dataframe from a fiber simulation dataframe.
    """
    # --- compute
    pca_dataset = get_sim_df_analysis_dataset(sim_df, aligned)
    pca = PCA(n_components=2)
    pca_samples_components = pca.fit_transform(pca_dataset)
    # --- setup dataframe for plotting
    pca_df = pd.DataFrame(
        data=pca_samples_components,
        columns=["principal component 1", "principal component 2"],
    )
    pca_df["label"] = sim_df.param_velocity
    pca_df["time"] = int(sim_df.num_timepoints)
    # --- return
    return pca_df


def plot_simulation_fibers_pca(pca_df: pd.DataFrame, sim_df: pd.DataFrame, title: str, figsize=6):
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
    num_fiber_points = int(sim_df.num_timepoints)
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

def get_simulation_fibers_pacmap_df(sim_df: pd.DataFrame, aligned: bool) -> pd.DataFrame:
    """
    Get a PACMAP analysis dataframe from a fiber simulation dataframe.     
    """
    # --- compute
    pacmap_dataset = get_sim_df_analysis_dataset(sim_df, aligned)
    embedding = pacmap.PaCMAP(n_components=2, n_neighbors=int(sim_df.num_fibers))
    pacmap_samples_components = embedding.fit_transform(pacmap_dataset, init="pca")
    # --- setup dataframe for plotting
    pacmap_df = pd.DataFrame(
        data=pacmap_samples_components,
        columns=["principal component 1", "principal component 2"],
    )
    pacmap_df["label"] = sim_df.param_velocity
    pacmap_df["time"] = int(sim_df.num_timepoints) # range(num_pacmap_samples)
    # --- return
    return pacmap_df


def plot_simulation_fibers_pacmap(pacmap_df: np.ndarray, sim_df: pd.DataFrame, title: str, figsize=6):
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
    num_fiber_points = int(sim_df.num_timepoints)
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
# Plot

for sim_df in simulation_dataframes:
    # PCA
    # --- not aligned
    pca_df = get_simulation_fibers_pca_df(sim_df, aligned=False)
    plot_simulation_fibers_pca(pca_df, sim_df, f"PAC: Aligned={False},Velocity={float(sim_df.param_velocity)}", figsize=4)
    # FYI alignment has no impact on PCA, but it does affect PACMAP
    # # --- aligned
    # pca_aligned_df = get_simulation_fibers_pca_df(sim_df, aligned=True)
    # plot_simulation_fibers_pca(pca_aligned_df, sim_df, f"Aligned={True},Velocity={float(sim_df.param_velocity)}", figsize=4)

    # PACMAP
    # --- not aligned
    pacmap_df = get_simulation_fibers_pacmap_df(sim_df, aligned=False)
    plot_simulation_fibers_pacmap(pacmap_df, sim_df, f"PaCMAP: Aligned={False},Velocity={float(sim_df.param_velocity)}", figsize=4)
    # --- aligned
    pacmap_aligned_df = get_simulation_fibers_pacmap_df(sim_df, aligned=True)
    plot_simulation_fibers_pacmap(pacmap_aligned_df, sim_df, f"PaCMAP: Aligned={True},Velocity={float(sim_df.param_velocity)}", figsize=4)

