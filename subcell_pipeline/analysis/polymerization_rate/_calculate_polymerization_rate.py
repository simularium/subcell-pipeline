# %% [markdown]
# # Calculate polymerization rate for fiber-membrane simulations

# %%
if __name__ != "__main__":
    raise ImportError("This module is a notebook and is not meant to be imported")

import matplotlib.pyplot as plt

# %%
from subcell_pipeline.simulation.readdy.loader import ReaddyLoader
from subcell_pipeline.simulation.readdy.parser import BOX_SIZE
from subcell_pipeline.simulation.readdy.post_processor import ReaddyPostProcessor

# %% [markdown]
"""
## Calculate polymerization rates of readdy-simulated actin filaments
"""

# %%
# ## Load data
time_inc = 1
timestep = 1
pickle_key = ""

h5_file_path = "../../../inputs/brownian_ratchet_binding_site_baseline_0.h5"

readdy_loader = ReaddyLoader(
    h5_file_path=h5_file_path,
    time_inc=time_inc,
    timestep=timestep,
    pickle_location=None,
    pickle_key="",
)

post_processor = ReaddyPostProcessor(readdy_loader.trajectory(), box_size=BOX_SIZE)

# %%
times = post_processor.times()
fiber_chain_ids = post_processor.linear_fiber_chain_ids(polymer_number_range=5)

fiber_lengths = [len(fiber) for tp in fiber_chain_ids for fiber in tp]

# %% [markdown]
# ## Visualize actin

# %% [markdown]
# ## Plot length vs time
plt.plot(fiber_lengths)

# %%
