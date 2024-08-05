"""Methods for vector operations."""

import numpy as np

from subcell_pipeline.analysis.compression_metrics.constants import ABSOLUTE_TOLERANCE


def get_unit_vector(vector: np.ndarray) -> np.ndarray:
    """
    Calculate the unit vector from a given vector.

    Parameters
    ----------
    vector
        Array containing the x,y,z positions of the vector.

    Returns
    -------
    :
        Unit vector.
    """
    if np.linalg.norm(vector) < ABSOLUTE_TOLERANCE or np.isnan(vector).any():
        return np.array([0, 0, 0])
    else:
        vec_length = np.linalg.norm(vector)
        return vector / vec_length


def get_end_to_end_unit_vector(polymer_trace: np.ndarray) -> np.ndarray:
    """
    Calculate the unit vector of the end-to-end axis of a polymer trace.

    Parameters
    ----------
    polymer_trace
        Array containing the x,y,z positions of the polymer trace.

    Returns
    -------
    :
        Unit vector of the end-to-end axis of the polymer trace.
    """
    assert len(polymer_trace) > 1, "Polymer trace must have at least 2 points"
    assert polymer_trace.shape[1] == 3, "Polymer trace must have 3 columns"

    end_to_end_axis = polymer_trace[-1] - polymer_trace[0]

    return get_unit_vector(end_to_end_axis)
