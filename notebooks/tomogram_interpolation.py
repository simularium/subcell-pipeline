# %% imports
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from subcell_analysis.compression_analysis import COMPRESSIONMETRIC
from subcell_analysis.compression_workflow_runner import compression_metrics_workflow

metrics = [
    COMPRESSIONMETRIC.NON_COPLANARITY,
    COMPRESSIONMETRIC.PEAK_ASYMMETRY,
    COMPRESSIONMETRIC.TOTAL_FIBER_TWIST,
    COMPRESSIONMETRIC.CALC_BENDING_ENERGY,
    COMPRESSIONMETRIC.CONTOUR_LENGTH,
    COMPRESSIONMETRIC.AVERAGE_PERP_DISTANCE,
]

options = {
    "signed": True,
}

# %% load dataframes
df_folder = Path("../data/dataframes")
# %%
tomogram_df = pd.read_csv(
    df_folder / "tomogram_cme_all_unbranched_coordinates.csv"
)

print(tomogram_df.shape)
tomogram_df
# %%
n_monomer_points = 200
# %%
cols_to_interp = ["xpos", "ypos", "zpos"]

df_subsample_list = []

# %%
for repeat, df_repeat in tomogram_df.groupby("repeat"):
    print(f"repeat: {repeat}")
    df_tmp = pd.DataFrame()
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
# %%
df_subsampled = pd.concat(df_subsample_list)
df_subsampled.reset_index(inplace=True, drop=True)
df_subsampled["simulator"]= "tomogram"
df_subsampled.to_csv(df_folder / "tomogram_subsampled.csv", index=False)
df_subsampled
# %%
def plot_filaments(chosen_filaments):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for filament_id, filament in chosen_filaments.groupby('repeat'):
        ax.plot(filament['xpos'], filament['ypos'], filament['zpos'], marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    return fig
plot_filaments(df_subsampled)


# convert plot_filaments to interactive plot
# %%
for repeat, df_repeat in df_subsampled.groupby("repeat"):
    df_repeat = compression_metrics_workflow(
        df_repeat, metrics_to_calculate=metrics, **options
    )
    for metric in metrics:
        df_subsampled.loc[df_repeat.index, metric.value] = df_repeat[metric.value]
# %%
