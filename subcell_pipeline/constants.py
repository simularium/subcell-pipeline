"""Constants for parsing simulations."""

from typing import Dict, List, Union

import numpy as np

from simulariumio import DisplayData, DISPLAY_TYPE


LOCAL_DOWNLOADS_PATH: str = "aws_downloads/"

COLUMN_NAMES: List[str] = [
    "fiber_id",
    "xpos",
    "ypos",
    "zpos",
    "xforce",
    "yforce",
    "zforce",
    "segment_curvature",
    "time",
    "fiber_point",
]

COLUMN_DTYPES: Dict[str, Union[float, int]] = {
    "fiber_id": int,
    "xpos": float,
    "ypos": float,
    "zpos": float,
    "xforce": float,
    "yforce": float,
    "zforce": float,
    "segment_curvature": float,
    "time": float,
    "fiber_point": int,
}

BOX_SIZE: np.ndarray = np.array(3 * [600.0])

READDY_TIMESTEP: float = 0.1  # ns

READDY_SAVED_FRAMES: int = 1000

READDY_TOTAL_STEPS: Dict[str, int] = {
    "ACTIN_NO_COMPRESSION" : 1e7,
    "ACTIN_COMPRESSION_VELOCITY_0047" : 3.2e8,
    "ACTIN_COMPRESSION_VELOCITY_0150" : 1e8, 
    "ACTIN_COMPRESSION_VELOCITY_0470" : 3.2e7, 
    "ACTIN_COMPRESSION_VELOCITY_1500" : 1e7,
}

# particle types correspond to types from simularium/readdy-models
ACTIN_START_PARTICLE_PHRASE: str = "pointed"
ACTIN_PARTICLE_TYPES: List[str] = [
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
IDEAL_ACTIN_POSITIONS: np.ndarray = np.array(
    [
        [24.738, 20.881, 26.671],
        [27.609, 24.061, 27.598],
        [30.382, 21.190, 25.725],
    ]
)
IDEAL_ACTIN_VECTOR_TO_AXIS: np.ndarray = np.array(
    [-0.01056751, -1.47785105, -0.65833209]
)

CYTOSIM_SCALE_FACTOR: float = 1000.0


def READDY_DISPLAY_DATA() -> Dict[str, DisplayData]:
    extra_radius = 1.5
    actin_radius = 2.0 + extra_radius
    n_polymer_numbers = 5
    result = {}
    for i in range(1, n_polymer_numbers + 1):
        result.update(
            {
                f"actin#{i}": DisplayData(
                    name="actin",
                    display_type=DISPLAY_TYPE.SPHERE,
                    radius=actin_radius,
                    color="#bf9b30",
                ),
                f"actin#mid_{i}": DisplayData(
                    name="actin#mid",
                    display_type=DISPLAY_TYPE.SPHERE,
                    radius=actin_radius,
                    color="#bf9b30",
                ),
                f"actin#fixed_{i}": DisplayData(
                    name="actin#fixed",
                    display_type=DISPLAY_TYPE.SPHERE,
                    radius=actin_radius,
                    color="#bf9b30",
                ),
                f"actin#mid_fixed_{i}": DisplayData(
                    name="actin#mid_fixed",
                    display_type=DISPLAY_TYPE.SPHERE,
                    radius=actin_radius,
                    color="#bf9b30",
                ),
                f"actin#ATP_{i}": DisplayData(
                    name="actin#ATP",
                    display_type=DISPLAY_TYPE.SPHERE,
                    radius=actin_radius,
                    color="#ffbf00",
                ),
                f"actin#mid_ATP_{i}": DisplayData(
                    name="actin#mid_ATP",
                    display_type=DISPLAY_TYPE.SPHERE,
                    radius=actin_radius,
                    color="#ffbf00",
                ),
                f"actin#fixed_ATP_{i}": DisplayData(
                    name="actin#fixed_ATP",
                    display_type=DISPLAY_TYPE.SPHERE,
                    radius=actin_radius,
                    color="#ffbf00",
                ),
                f"actin#mid_fixed_ATP_{i}": DisplayData(
                    name="actin#mid_fixed_ATP",
                    display_type=DISPLAY_TYPE.SPHERE,
                    radius=actin_radius,
                    color="#ffbf00",
                ),
                f"actin#barbed_{i}": DisplayData(
                    name="actin#barbed",
                    display_type=DISPLAY_TYPE.SPHERE,
                    radius=actin_radius,
                    color="#ffdc73",
                ),
                f"actin#barbed_ATP_{i}": DisplayData(
                    name="actin#barbed_ATP",
                    display_type=DISPLAY_TYPE.SPHERE,
                    radius=actin_radius,
                    color="#ffdc73",
                ),
                f"actin#fixed_barbed_{i}": DisplayData(
                    name="actin#fixed_barbed",
                    display_type=DISPLAY_TYPE.SPHERE,
                    radius=actin_radius,
                    color="#ffdc73",
                ),
                f"actin#fixed_barbed_ATP_{i}": DisplayData(
                    name="actin#fixed_barbed_ATP",
                    display_type=DISPLAY_TYPE.SPHERE,
                    radius=actin_radius,
                    color="#ffdc73",
                ),
                f"actin#pointed_{i}": DisplayData(
                    name="actin#pointed",
                    display_type=DISPLAY_TYPE.SPHERE,
                    radius=actin_radius,
                    color="#a67c00",
                ),
                f"actin#pointed_ATP_{i}": DisplayData(
                    name="actin#pointed_ATP",
                    display_type=DISPLAY_TYPE.SPHERE,
                    radius=actin_radius,
                    color="#a67c00",
                ),
                f"actin#pointed_fixed_{i}": DisplayData(
                    name="actin#pointed_fixed",
                    display_type=DISPLAY_TYPE.SPHERE,
                    radius=actin_radius,
                    color="#a67c00",
                ),
                f"actin#pointed_fixed_ATP_{i}": DisplayData(
                    name="actin#pointed_fixed_ATP",
                    display_type=DISPLAY_TYPE.SPHERE,
                    radius=actin_radius,
                    color="#a67c00",
                ),
            },
        )
    return result
