# %% imports
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from subcell_analysis.compression_analysis import COMPRESSIONMETRIC
from subcell_analysis.compression_workflow_runner import compression_metrics_workflow

metrics = [
    COMPRESSIONMETRIC.NON_COPLANARITY,
    COMPRESSIONMETRIC.PEAK_ASYMMETRY,
    COMPRESSIONMETRIC.TOTAL_FIBER_TWIST,
    COMPRESSIONMETRIC.CALC_BENDING_ENERGY,
    COMPRESSIONMETRIC.CONTOUR_LENGTH,
    COMPRESSIONMETRIC.AVERAGE_PERP_DISTANCE,
    COMPRESSIONMETRIC.COMPRESSION_RATIO,
]

options = {
    "signed": True,
}

# %% load dataframes
df_folder = Path("../data/dataframes")
# %%
tomogram_df = pd.read_csv(df_folder / "tomogram_cme_all_unbranched_coordinates.csv")

print(tomogram_df.shape)
tomogram_df


# %%
n_monomer_points = 200
# %%
cols_to_interp = ["X", "Y", "Z"]


def create_unique_identifier(row):
    # Extract the unique part of the dataset name (e.g., '2018August_Tomo27')
    unique_part = row["dataset"]

    # Combine it with the 'fil' column
    return f"{row['fil']}__{unique_part}"


# Apply the function to each row
tomogram_df["unique_identifier"] = tomogram_df.apply(create_unique_identifier, axis=1)

# Display the dataframe
tomogram_df
df_subsample_list = []
# %%

# Interpolate the data to have the same number of points for each filament
df_tmp = pd.DataFrame()
for repeat, df_repeat in tomogram_df.groupby("unique_identifier"):
    df_tmp = pd.DataFrame()
    if df_repeat.shape[0] < 3:
        continue
    df_tmp["monomer_ids"] = np.arange(n_monomer_points)
    df_tmp["time"] = 0
    df_tmp["velocity"] = 0
    df_tmp["repeat"] = repeat
    for col in cols_to_interp:
        df_tmp[col] = np.interp(
            np.linspace(0, 1, n_monomer_points),
            np.linspace(0, 1, df_repeat.shape[0]),
            df_repeat[col].values,
        )
    df_subsample_list.append(df_tmp)
df_tmp

# %%
df_reg_list = []
tomogram_df
# small fils means less than 4 points
small_fils = []
for repeat, df_repeat in tomogram_df.groupby("unique_identifier"):
    df_tmp = pd.DataFrame()
    print(df_repeat.shape[0])
    if df_repeat.shape[0] < 3:
        continue
    df_tmp["monomer_ids"] = np.arange(df_repeat.shape[0])
    df_tmp["time"] = 0
    df_tmp["velocity"] = 0
    df_tmp["repeat"] = repeat
    df_tmp["X"], df_tmp["Y"], df_tmp["Z"] = (
        df_repeat["X"].values,
        df_repeat["Y"].values,
        df_repeat["Z"].values,
    )
    df_reg_list.append(df_tmp)
df_reg = pd.concat(df_reg_list)
df_reg.reset_index(inplace=True, drop=True)
df_reg.rename(columns={"X": "xpos", "Y": "ypos", "Z": "zpos"}, inplace=True)
# print(small_fils)
# remove all rows from df_reg with unique_identifier in small_fils
df_reg = df_reg[~df_reg["repeat"].isin(small_fils)]
df_reg
# %%
df_subsampled = pd.concat(df_subsample_list)
df_subsampled.reset_index(inplace=True, drop=True)
df_subsampled["simulator"] = "tomogram"
df_subsampled.to_csv(df_folder / "tomogram_subsampled.csv", index=False)
df_subsampled


# %%
def plot_filaments(chosen_filaments):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for filament_id, filament in chosen_filaments.groupby("repeat"):
        ax.plot(filament["X"], filament["Y"], filament["Z"], marker="o")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    return fig


plot_filaments(df_subsampled)

df_subsampled.rename(columns={"X": "xpos", "Y": "ypos", "Z": "zpos"}, inplace=True)
# convert plot_filaments to interactive plot
# %%
chosen_metrics = [
    COMPRESSIONMETRIC.CONTOUR_LENGTH,
    COMPRESSIONMETRIC.AVERAGE_PERP_DISTANCE,
    COMPRESSIONMETRIC.CALC_BENDING_ENERGY,
    COMPRESSIONMETRIC.NON_COPLANARITY,
    COMPRESSIONMETRIC.PEAK_ASYMMETRY,
    COMPRESSIONMETRIC.TOTAL_FIBER_TWIST,
    COMPRESSIONMETRIC.COMPRESSION_RATIO,
]
for repeat, df_repeat in df_subsampled.groupby("repeat"):
    df_repeat = compression_metrics_workflow(
        df_repeat, metrics_to_calculate=chosen_metrics, **options
    )
    for metric in chosen_metrics:
        df_subsampled.loc[df_repeat.index, metric.value] = df_repeat[metric.value]
# %%
df_subsampled
df_subsampled.to_csv(df_folder / "tomogram_subsampled_with_metrics.csv", index=False)
# %%
i = 0
for repeat, df_repeat in df_reg.groupby("repeat"):
    df_repeat = compression_metrics_workflow(
        df_repeat, metrics_to_calculate=chosen_metrics, **options
    )
    for metric in chosen_metrics:
        df_reg.loc[df_repeat.index, metric.value] = df_repeat[metric.value]
df_reg
# %%
df_reg["CONTOUR_LENGTH"].describe()
df_subsampled["CONTOUR_LENGTH"].describe()
# %%
# %%
df_subsampled.to_csv(df_folder / "tomogram_subsampled_with_metrics.csv", index=False)
df_reg.to_csv(df_folder / "tomogram_data_with_metrics.csv", index=False)

# %%
df_reg
# %%
