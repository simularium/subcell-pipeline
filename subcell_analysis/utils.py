#!/usr/bin/env python

from typing import Tuple

import numpy as np


ABSOLUTE_TOLERANCE = 1e-6

def get_unit_vector(vector: np.array) -> Tuple[np.array, float]:
    if np.linalg.norm(vector) < ABSOLUTE_TOLERANCE or np.isnan(vector).any():
        return np.array([0, 0, 0]), 0
    else:
        vec_length = np.linalg.norm(vector)
        return vector / vec_length, vec_length
