"""Constants for parsing ReaDDy simulations."""


import numpy as np


# particle types correspond to types from simularium/readdy-models
ACTIN_START_PARTICLE_PHRASE = "pointed"
ACTIN_PARTICLE_TYPES = [
    "actin#",
    "actin#ATP_",
    "actin#mid_",
    "actin#mid_ATP_",
    "actin#fixed_",
    "actin#fixed_ATP_",
    "actin#mid_fixed_",
    "actin#mid_fixed_ATP_",
    "actin#barbed_",
    "actin#barbed_ATP_",
    "actin#fixed_barbed_",
    "actin#fixed_barbed_ATP_",
]

# measured from crystal structure
IDEAL_ACTIN_POSITIONS = np.array(
    [
        [24.738, 20.881, 26.671],
        [27.609, 24.061, 27.598],
        [30.382, 21.190, 25.725],
    ]
)
IDEAL_ACTIN_VECTOR_TO_AXIS = np.array(
    [-0.01056751, -1.47785105, -0.65833209]
)
