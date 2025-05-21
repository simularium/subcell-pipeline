# %% [markdown]
# # Calculate polymerization rate for fiber-membrane simulations
#
# Expecting 500 nm/s in physiological systems.

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

# %% [markdown]
# ## Load data

# %%
time_inc = 1  # how many timesteps of data to load
timestep = 0.1  # model timestep in ns

condition = "baseline"

h5_file_paths = {
    "baseline": "../../../inputs/brownian_ratchet_binding_site_baseline_0.h5",
    "membrane": "../../../inputs/brownian_ratchet_binding_site_membrane_0.h5",
}

pickle_location = "/home/jessica/Documents/subcell-pipeline/"
pickle_keys = {
    "baseline": "inputs/brownian_ratchet_binding_site_baseline_0.pkl",
    "membrane": "inputs/brownian_ratchet_binding_site_membrane_0.pkl",
}

readdy_loader = ReaddyLoader(
    h5_file_path=h5_file_paths[condition],
    time_inc=time_inc,
    timestep=timestep,
    pickle_location=pickle_location,
    pickle_key=pickle_keys[condition],
)

post_processor = ReaddyPostProcessor(readdy_loader.trajectory(), box_size=BOX_SIZE)

# %%
times = post_processor.times()
fiber_chain_ids = post_processor.linear_fiber_chain_ids(polymer_number_range=5)
fiber_positions, _ = post_processor.linear_fiber_axis_positions(fiber_chain_ids)

# %%
from subcell_pipeline.analysis.compression_metrics.polymer_trace import (
    get_contour_length_from_trace,
    get_end_to_end_distance,
)

# %%
contour_lengths = [
    get_contour_length_from_trace(fiber) for tp in fiber_positions for fiber in tp
]
end_to_end_distances = [
    get_end_to_end_distance(fiber) for tp in fiber_positions for fiber in tp
]

# %%
plt.plot(contour_lengths, label="contour")
plt.plot(end_to_end_distances, label="end-to-end")
plt.title("Polymer length vs time")
plt.xlabel("Time (ns)")
plt.ylabel("Fiber length (nm)")
plt.legend()

# %%
from subcell_pipeline.analysis.polymerization_rate.polymerization_rate import (
    calculate_average_polymerization_rate,
)

# %%
length_scale = 1  # nm / unit
time_scale = 0.1  # ns / timestep
avg_polymerization_rate = calculate_average_polymerization_rate(
    contour_lengths, times, length_scale, time_scale
)
avg_polymerization_rate_end_to_end = calculate_average_polymerization_rate(
    end_to_end_distances, times, length_scale, time_scale
)

print(
    f"Average polymerization rate (contour length): {avg_polymerization_rate * 1E9:0.2f} nm/s"
)
print(
    f"Average polymerization rate (end-to-end): {avg_polymerization_rate_end_to_end * 1E9:0.2f} nm/s"
)

# %%
# TODO: distance from surface
