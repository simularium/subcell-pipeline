import numpy as np
from simulariumio import (
    AgentData,
    DisplayData,
    MetaData,
    TrajectoryConverter,
    TrajectoryData,
    UnitData,
)


def generate_trajectory_converter_for_fiber_points(
    fiber_points: list[np.ndarray],
    type_names: list[str],
    meta_data: MetaData,
    display_data: dict[str, DisplayData],
    time_units: UnitData,
    spatial_units: UnitData,
    fiber_radius: float = 0.5,
) -> TrajectoryConverter:
    """
    Generate a TrajectoryConverter for the given fiber points.

    Parameters
    ----------
    fiber_points
        List of fibers, where each fiber has the shape (timesteps x points x 3).
    type_names
        List of type names.
    meta_data
        Simularium metadata object.
    display_data
        Map of type names to Simularium display data objects.
    time_units
        Time unit data.
    spatial_units
        Spatial unit data.

    Returns
    -------
    :
        Simularium trajectory converter.
    """

    # build subpoints array with correct dimensions
    n_fibers = len(fiber_points)
    total_steps = fiber_points[0].shape[0]
    n_points = fiber_points[0].shape[1]
    subpoints = np.zeros((total_steps, n_fibers, n_points, 3))
    for time_ix in range(total_steps):
        for fiber_ix in range(n_fibers):
            subpoints[time_ix][fiber_ix] = fiber_points[fiber_ix][time_ix]
    subpoints = subpoints.reshape((total_steps, n_fibers, 3 * n_points))

    # convert to simularium
    traj_data = TrajectoryData(
        meta_data=meta_data,
        agent_data=AgentData(
            times=np.arange(total_steps),
            n_agents=n_fibers * np.ones(total_steps),
            viz_types=1001 * np.ones((total_steps, n_fibers)),  # fiber viz type = 1001
            unique_ids=np.array(total_steps * [list(range(n_fibers))]),
            types=total_steps * [type_names],
            positions=np.zeros((total_steps, n_fibers, 3)),
            radii=fiber_radius * np.ones((total_steps, n_fibers)),
            n_subpoints=3 * n_points * np.ones((total_steps, n_fibers)),
            subpoints=subpoints,
            display_data=display_data,
        ),
        time_units=time_units,
        spatial_units=spatial_units,
    )
    return TrajectoryConverter(traj_data)
