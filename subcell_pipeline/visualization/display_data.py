"""Methods for visualization display data."""

from simulariumio import DISPLAY_TYPE, DisplayData


def get_readdy_display_data() -> dict[str, DisplayData]:
    """
    Get DisplayData for ReaDDy actin simulations.

    Each actin monomer uses the `SPHERE` display type with the following colors:

    ======================= =======
    Label                   Color
    ======================= =======
    actin                   #bf9b30
    actin#mid               #bf9b30
    actin#fixed             #bf9b30
    actin#mid_fixed         #bf9b30
    actin#ATP               #ffbf00
    actin#mid_ATP           #ffbf00
    actin#fixed_ATP         #ffbf00
    actin#mid_fixed_ATP     #ffbf00
    actin#barbed            #ffdc73
    actin#barbed_ATP        #ffdc73
    actin#fixed_barbed      #ffdc73
    actin#fixed_barbed_ATP  #ffdc73
    actin#pointed           #a67c00
    actin#pointed_ATP       #a67c00
    actin#pointed_fixed     #a67c00
    actin#pointed_fixed_ATP #a67c00
    ======================= =======

    Returns
    -------
    :
        Map of names to DisplayData objects.
    """

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
