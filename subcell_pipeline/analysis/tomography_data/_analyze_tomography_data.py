# %% [markdown]
# # Analyze tomography data

# %% [markdown]
"""
Notebook contains steps for loading, processing, and analyzing segmented
cryo-electron tomography of actin filaments.

- [Load tomography datasets](#load-tomography-datasets)
"""

# %%
from subcell_pipeline.analysis.tomography_data.tomography_data import (
    get_branched_tomography_data,
    get_unbranched_tomography_data,
)

# %% [markdown]
"""
## Load tomography datasets

Tomogram data is compiled from data published in:

> D Serwas, M Akamatsu, A Moayed, K Vegesna, R Vasan, JM Hill, J Schöneberg, KM
Davies, P Rangamani, DG Drubin. (2022). Mechanistic insights into actin force
generation during vesicle formation from cryo-electron tomography.
_Developmental Cell_, 57(9), P1132-1145.e5. DOI: 10.1016/j.devcel.2022.04.012

Each dataset contains x, y, and z positions of segmented actin filaments from
cryo-electron tomography.
"""

# %%
# S3 bucket for input and output files
bucket = "s3://subcell-working-bucket"

# Data repository for downloading tomography data
repository = "https://raw.githubusercontent.com/RangamaniLabUCSD/actincme/master/PolarityAnalysis/"

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

# Spatial conversion scaling factor (pixels to um)
scale_factor = 0.00006

# %%
branched_df = get_branched_tomography_data(
    bucket, repository, branched_datasets, scale_factor
)
unbranched_df = get_unbranched_tomography_data(
    bucket, repository, unbranched_datasets, scale_factor
)
