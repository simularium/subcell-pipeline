# %% [markdown]
# # Analyze persistence length of simulation results

# %% [markdown]
"""
Notebook contains steps to calculate persistence length 
for baseline simulations where there was no compression.

- [Define parameters](#define-parameters)
- [Calculate persistence length for Cytosim](#calculate-persistence-length-for-cytosim)
- [Calculate persistence length for ReaDDy](#calculate-persistence-length-for-readdy)
- [Plot persistence length](#plot-persistence-length)
"""

# %%
if __name__ != "__main__":
    raise ImportError("This module is a notebook and is not meant to be imported")

# %%
from subcell_pipeline.analysis.persistence_length.persistence_length import (
    get_persistence_length_data,
    plot_persistence_length,
)

# %% [markdown]
"""
## Define parameters

Calculate persistence length using the `ACTIN_NO_COMPRESSION` simulation series, 
which simulates a single actin fiber with a free barbed end across five replicates.
"""

# %%
# Name of the simulation series
no_compression_series_name: str = "ACTIN_NO_COMPRESSION"

# S3 bucket Cytosim for input files
cytosim_bucket: str = "s3://cytosim-working-bucket"

# S3 bucket ReaDDy for input files
readdy_bucket: str = "s3://readdy-working-bucket"

# S3 bucket for output files
bucket = "s3://subcell-working-bucket"

# Random seeds for simulations
random_seeds: list[int] = [1, 2, 3, 4, 5]

# %% [markdown]
"""
## Calculate persistence length for Cytosim
"""

# %%
pl_data_cytosim = get_persistence_length_data(
    bucket=cytosim_bucket,
    series_name=no_compression_series_name,
    condition_keys=[""],
    random_seeds=random_seeds,
)

# %% [markdown]
"""
## Calculate persistence length for ReaDDy
"""

# %%
pl_data_readdy = get_persistence_length_data(
    bucket=readdy_bucket,
    series_name=no_compression_series_name,
    condition_keys=[""],
    random_seeds=random_seeds,
)

# %% [markdown]
"""
## Plot persistence length
"""

# %%
plot_persistence_length(dict(pl_data_cytosim, **pl_data_readdy), bucket)
