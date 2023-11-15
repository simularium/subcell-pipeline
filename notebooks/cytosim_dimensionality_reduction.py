# %%
import numpy as np
import pacmap
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# %%
# SETUP
# --- data refs
data_directory = "../data"
# --- math funcs
# Get RMSD between 2 curves
def rmsd(vec1, vec2):
    vec1 = vec1.swapaxes(0, 1).copy()
    vec2 = vec2.swapaxes(0, 1).copy()
    rmsd = np.sqrt(((((vec1 - vec2) ** 2)) * 3).mean())
    return rmsd
# --- visualization funcs
# Get a colour gradient to represent time
c1 = "cyan"
c2 = "blue"
colors = []
def colorFader(
    c1, c2, mix=0
):  # fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1 = np.array(mpl.colors.to_rgb(c1))
    c2 = np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1 - mix) * c1 + mix * c2)
for i in range(101):
    colors.append(colorFader(c1, c2, i / 101))

# %%
# Load in all actin positions at all % of time

# The shape of all_actin_vectors_all_time is 101 timepoints * 20 fibers * 3 dimensions * 500 monomer points
x = np.loadtxt(f"{data_directory}/cytosim_dimensionality_positions.txt")
all_actin_vectors_all_time = x.reshape(101, 20, 3, 500)
# print(all_actin_vectors_all_time)


# %%
# Reshape the np array to use scipy rotation. Also select a ref fiber between (2000-2020, i.e. last time point) to
# align all the fibers into the same plane
x = np.array(all_actin_vectors_all_time)
a = np.ravel(x)
# TODO: this feels like a dangerous/weird way to select a ref fiber.
fibers = a.reshape((2020, 3, 500))


# %%
NUM_TIMEPOINTS = 101 # what's the unit of time?
NUM_FIBERS = 20
NUM_MONOMER_DIMS = 3
NUM_MONOMERS = 500
NUM_FIBERS_BY_TIME_POINTS = NUM_TIMEPOINTS * NUM_FIBERS
NUM_MONOMERS_WITH_DIMS = NUM_MONOMER_DIMS * NUM_MONOMERS

# While normalizing data is important for PCA, our data are spatial points. We don't want to transform it
USE_SCALAR_SCALED_DATA = False

# why are we reshaping this? (101 timepoints * 20 fibers) * (3 dimensions * 500 monomer points)
pca_dataset = np.array(fibers)
pca_dataset = pca_dataset.reshape((NUM_FIBERS_BY_TIME_POINTS, NUM_MONOMERS_WITH_DIMS))
if USE_SCALAR_SCALED_DATA == True:
    # scaling because that's important to do before PCA?
    scaler = StandardScaler()
    x = scaler.fit_transform(pca_dataset)
else:
    # not scaling because its what we did before
    x = pca_dataset
# pca'ing
pca_inst = PCA(n_components=2)
pcs = pca_inst.fit_transform(x) # not a fan of this re-writing of a vague x variable

print(f"Explained Variance Ratio: {pca_inst.explained_variance_ratio_}")

# %%
# ### Run Params
# Param changing between simulations: velocity
SIMULATION_1_VELOCITY = 4.7
SIMULATION_2_VELOCITY = 15
SIMULATION_3_VELOCITY = 47
SIMULATION_4_VELOCITY = 150

sims = []
for i in range(2020):
    if i % 20 < 5:
        sims.append(SIMULATION_1_VELOCITY)
    elif i % 20 < 10:
        sims.append(SIMULATION_2_VELOCITY)
    elif i % 20 < 15:
        sims.append(SIMULATION_4_VELOCITY)
    else:
        sims.append(SIMULATION_3_VELOCITY)

time = []
for i in range(NUM_FIBERS_BY_TIME_POINTS):
    # every 20th number, we slide the scale of colors bc time step
    t = i // 20
    time.append(t)


# %%
# ### PCA: DataFrame

pca_dataframe = pd.DataFrame(
    data=pcs,
    columns=["principal component 1", "principal component 2"],
)
pca_dataframe["label"] = sims
pca_dataframe["time"] = time


# %%
# ### PCA: Plot

# could make this function have better param selections maybe?
def plot_pca(dataframe, param_label, param_simulation_run_vars):
    print(f"PCA Plot (Scaled Data: {USE_SCALAR_SCALED_DATA})")
    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
    time_points = range(0, NUM_TIMEPOINTS)
    # List of parameters or conditions under which the PCA was run.
    for i in range(2):
        for j in range(2):
            ax[i][j].set_xlabel("PC1", loc="left")
            ax[i][j].set_ylabel("PC2", loc="bottom")
            ax[i][j].set_title(f"{param_label} = {param_simulation_run_vars[2 * i + j]}", fontsize=10)
    for i in range(len(time_points)):
        for j in range(len(param_simulation_run_vars)):
            col = colors[i]
            indicesToKeep = (dataframe["time"] == time_points[i]) & (
                dataframe["label"] == param_simulation_run_vars[j]
            )
            ax[j // 2][j % 2].scatter(
                dataframe.loc[indicesToKeep, "principal component 1"],
                dataframe.loc[indicesToKeep, "principal component 2"],
                c=col,
                s=70,
            )


# %%
# ### PACMAP: DataFrame

embedding = pacmap.PaCMAP(n_components=2, n_neighbors=5)
x_transformed = embedding.fit_transform(x, init="pca")
y = np.array(sims)
pacmap_dataframe = pd.DataFrame(
    data=x_transformed, columns=["pacmap component 1", "pacmap component 2"]
)
pacmap_dataframe["label"] = sims
pacmap_dataframe["time"] = time

# %%
# ### PACMAP: Plot

def plot_pacmap(dataframe, param_label, param_simulation_run_vars):
    print(f"PacMap Visualization (Scaled Data: {USE_SCALAR_SCALED_DATA})")
    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
    targets = range(0, 101)
    ax[1][1].set_xlabel("PC1")
    ax[1][1].set_ylabel("PC2")
    for i in range(2):
        for j in range(2):
            ax[i][j].set_xlabel("PC1", loc="left")
            ax[i][j].set_ylabel("PC2", loc="bottom")
            ax[i][j].set_title(f"{param_label} = {param_simulation_run_vars[2 * i + j]}", fontsize=10)
    for i in range(len(targets)):
        for j in range(len(param_simulation_run_vars)):
            col = colors[i]
            indicesToKeep = (dataframe["time"] == targets[i]) & (
                dataframe["label"] == param_simulation_run_vars[j]
            )
            ax[j // 2][j % 2].scatter(
                dataframe.loc[indicesToKeep, "pacmap component 1"],
                dataframe.loc[indicesToKeep, "pacmap component 2"],
                c=col,
                s=70,
            )


# %%
# ### Explained Variance Visualization

def plot_explained_variance_visualization(pca: PCA):
    print(f"Explained Variance Visualization (Scaled Data: {USE_SCALAR_SCALED_DATA})")
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
plot_pca(pca_dataframe, "Sim. Velocity", [SIMULATION_1_VELOCITY, SIMULATION_2_VELOCITY, SIMULATION_3_VELOCITY, SIMULATION_4_VELOCITY])

# %%
plot_pacmap(pacmap_dataframe, "Sim. Velocity", [SIMULATION_1_VELOCITY, SIMULATION_2_VELOCITY, SIMULATION_3_VELOCITY, SIMULATION_4_VELOCITY])
