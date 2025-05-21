"""Methods to calculate polymerization rate."""

import numpy as np


def calculate_average_polymerization_rate(
    fiber_lengths: np.ndarray, times: np.ndarray, length_scale: float, time_scale: float
) -> float:
    """
     Calculate average polymerization rate of fiber.

    Parameters
    ----------
    fiber_lengths
        Array of fiber lengths in units.
    times
        Array of times in timesteps.
    length_scale
        Length scaling factor in nm/unit.
    time_scale
        Time scaling factor in ns/timestep.

    Returns
    -------
    :
        Average polymerization rate
    """

    change_in_length = (fiber_lengths[-1] - fiber_lengths[0]) * length_scale
    change_in_time = (times[-1] - times[0]) * time_scale

    return change_in_length / change_in_time
