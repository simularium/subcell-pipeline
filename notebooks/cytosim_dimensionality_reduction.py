# %%
import numpy as np
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

NUM_TIMEPOINTS = 101
NUM_FIBERS = 20
NUM_MONOMER_DIMS = 3
NUM_MONOMERS = 500
NUM_FIBERS_BY_TIME_POINTS = NUM_TIMEPOINTS * NUM_FIBERS
NUM_MONOMERS_WITH_DIMS = NUM_MONOMER_DIMS * NUM_MONOMERS

USE_SCALAR_SCALED_DATA = True

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
principalDf = pd.DataFrame(
    data=pcs,
    columns=["principal component 1", "principal component 2"],
)

print(principalDf)

# %%
# ### Run Params
# We have 4 different run params I believe: 4.7, 15, 47, 150, what they stand for idk.

# what are these? simulation velocity?
SIMULATION_1_VELOCITY = 4.7
SIMULATION_2_VELOCITY = 15
SIMULATION_3_VELOCITY = 47
SIMULATION_4_VELOCITY = 150

sims = []
for i in range(2020):
    if i % 20 < 5:
        sims.append(4.7)
    elif i % 20 < 10:
        sims.append(15)
    elif i % 20 < 15:
        sims.append(150)
    else:
        sims.append(47)

time = []
for i in range(NUM_FIBERS_BY_TIME_POINTS):
    # every 20th number, we slide the scale of colors bc time step
    t = i // 20
    time.append(t)

# these need to be dimensions of 2020 or throws
# this also feels weird. are we reconciling/aligning reshaped data based on prior data positions?
principalDf["label"] = sims
principalDf["time"] = time


# %%
# ### PCA Plot
print(f"PCA Plot (Scaled Data: {USE_SCALAR_SCALED_DATA})")

# could make this function have better param selections maybe?
def pca_plot(dataframe, param_label, param_simulation_run_vars):
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

pca_plot(principalDf, "Sim. Velocity(?)", [SIMULATION_1_VELOCITY, SIMULATION_2_VELOCITY, SIMULATION_3_VELOCITY, SIMULATION_4_VELOCITY])


# %%
# ### Explained Variance Visualization
print(f"Explained Variance Visualization (Scaled Data: {USE_SCALAR_SCALED_DATA})")

def explained_variance_visualization(pca: PCA):
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

explained_variance_visualization(pca_inst)


# %%
# ### Biplot Visualization
print(f"Biplot (Scaled Data: {USE_SCALAR_SCALED_DATA})")

def biplot(score, coeff, labels=None):
    # `score`: PCA scores, the transformed dataset with reduced dimensions.
    # `coeff`: PCA loadings, the importance of each feature on the components.
    # `labels`: Feature names for plotting (optional).
    # Extract the scores for the first two principal components.
    xs = score[:,0]  # Scores for PC1
    ys = score[:,1]  # Scores for PC2
    # Determine the number of variables based on the loadings shape.
    n = coeff.shape[0]  # Number of original features
    # Normalize the scale of the scatter plot to ensure it fits well in the figure.
    # This prevents the arrows from being too short or too long.
    scalex = 1.0/(xs.max() - xs.min())  # Scaling factor for x-axis
    scaley = 1.0/(ys.max() - ys.min())  # Scaling factor for y-axis
    # Create a scatter plot of the principal components.
    plt.scatter(xs * scalex, ys * scaley)  # Plot the normalized scores
    # Loop over the number of variables to plot the PCA loadings (arrows).
    for i in range(n):
        # Draw an arrow from the origin (0,0) to the end point of the loading vector.
        plt.arrow(0, 0, coeff[i,0], coeff[i,1], color='r', alpha=0.5)
        # Add labels to the arrows. If no labels provided, use 'Var1', 'Var2', etc.
        if labels is None:
            # If labels are not provided, name them as Var1, Var2, and so on.
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color='g', ha='center', va='center')
        else:
            # If labels are provided, use them and place the text slightly beyond the arrow's tip.
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color='g', ha='center', va='center')
    # Label the axes with the principal components.
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid()

biplot(pcs, np.transpose(pca_inst.components_[0:2, :]))
