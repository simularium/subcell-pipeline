# %% [markdown]
# # Analyze actin CME tomography data

# %% [markdown]
"""
Notebook contains steps for loading, processing, and analyzing segmented
cryo-electron tomography of CME-associated actin filaments.

Data is compiled from:

> D Serwas, M Akamatsu, A Moayed, K Vegesna, R Vasan, JM Hill, J Schöneberg, KM
Davies, P Rangamani, DG Drubin. (2022). Mechanistic insights into actin force
generation during vesicle formation from cryo-electron tomography.
_Developmental Cell_, 57(9), P1132-1145.e5. DOI: 10.1016/j.devcel.2022.04.012

- [Load tomography datasets](#load-tomography-datasets)
- [Plot branched tomography fibers](#plot-branched-tomography-fibers)
- [Plot unbranched tomography fibers](#plot-unbranched-tomography-fibers)
- [Define sampling settings](#define-sampling-settings)
- [Sample tomography data](#sample-tomography-data)
- [Plot sampled tomography fibers](#plot-sampled-tomography-fibers)
"""

# %%
if __name__ != "__main__":
    raise ImportError("This module is a notebook and is not meant to be imported")

# %%
import pandas as pd

from subcell_pipeline.analysis.tomography_data.tomography_data import (
    get_branched_tomography_data,
    get_unbranched_tomography_data,
    plot_tomography_data_by_dataset,
    sample_tomography_data,
)

# %% [markdown]
"""
## Load tomography datasets

Each dataset contains x, y, and z positions of segmented actin filaments from
cryo-electron tomography.
"""

# %%
# Dataset name
name = "actin_cme_tomography"

# S3 bucket for input and output files
bucket = "s3://subcell-working-bucket"

# Data repository for downloading tomography data
repository = "https://raw.githubusercontent.com/RangamaniLabUCSD/actincme/master/PolarityAnalysis/"

# Conversion factor from pixels to um for this dataset
tomography_scale_factor: float = 0.0006

# Folders and names of branched actin datasets
branched_datasets = [
    ("2018August_Tomo27", "TomoAugust_27_earlyCME"),
    ("2018June_Tomo14_Early_Invagination", "2018June_Tomo14_Early_Invagination"),
    ("2018June_Tomo14_Late_Invagination", "2018June_Tomo14_Late_Invagination"),
    ("2018June_Tomo26", "2018June_Tomo26_CME_Invagination"),
    ("2018March", "2018March_Late_Invagination"),
    ("2018November_32", "TomoNovember_32_Vesicle"),
]

# Folders and names of unbranched actin datasets
unbranched_datasets = [
    ("2018August_Tomo27", "TomoAugust_27_earlyCME"),
    ("2018June_Tomo14_Early_Invagination", "2018June_Tomo14_Early_Invagination"),
    ("2018June_Tomo14_Late_Invagination", "2018June_Tomo14_Late_Invagination"),
    ("2018June_Tomo26", "2018June_Tomo26_CME_Invagination"),
    ("2018November_32", "TomoNovember_32_Vesicle"),
]

# %%
branched_df = get_branched_tomography_data(
    bucket=bucket,
    name=name,
    repository=repository,
    datasets=branched_datasets,
    scale_factor=tomography_scale_factor,
)
unbranched_df = get_unbranched_tomography_data(
    bucket=bucket,
    name=name,
    repository=repository,
    datasets=unbranched_datasets,
    scale_factor=tomography_scale_factor,
)

# %% [markdown]
"""
## Plot branched tomography fibers
"""

# %%
plot_tomography_data_by_dataset(
    branched_df, bucket, f"{name}/{name}_plots_branched_%s.png"
)

# %% [markdown]
"""
## Plot unbranched tomography fibers
"""

# %%
plot_tomography_data_by_dataset(
    unbranched_df, bucket, f"{name}/{name}_plots_unbranched_%s.png"
)

# %% [markdown]
"""
## Define sampling settings

Defines the settings used for subsampling tomography data points.
"""

# %%
# Number of monomer points per fiber
n_monomer_points = 20

# Minimum number of points for valid fiber
minimum_points = 3

# True to recalculate the sampled tomography data, False otherwise.
recalculate = True

# %% [markdown]
"""
## Sample tomography data

Sample monomer points for each unique segmented tomography fiber. Fibers with
less than the specified minimum number of segmented points are excluded from the
sampling.
"""

# %%
sampled_key = f"{name}/{name}_coordinates_sampled.csv"
all_tomogram_df = pd.concat([branched_df, unbranched_df])
sampled_data = sample_tomography_data(
    all_tomogram_df,
    bucket,
    sampled_key,
    n_monomer_points,
    minimum_points,
    recalculate=recalculate,
)

# %% [markdown]
"""
## Plot sampled tomography fibers
"""

# %%
plot_tomography_data_by_dataset(
    sampled_data, bucket, f"{name}/{name}_plots_all_sampled_%s.png"
)
