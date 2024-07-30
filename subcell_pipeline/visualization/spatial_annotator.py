"""Methods for adding spatial annotations to visualizations."""

import numpy as np
from simulariumio import DISPLAY_TYPE, DimensionData, DisplayData, TrajectoryData
from simulariumio.constants import VIZ_TYPE


def _added_dimensions_for_fibers(
    traj_data: TrajectoryData, data: list[list[np.ndarray]]
) -> DimensionData:
    """
    Get a DimensionData with deltas for each dimension of AgentData.

    Used when adding fiber annotation data.

    Data shape = [timesteps, fibers, np.array(points, 3)] (assumed to be jagged)
    """
    total_steps = len(data)
    max_fibers = 0
    max_points = 0
    for time_ix in range(total_steps):
        n_fibers = len(data[time_ix])
        if n_fibers > max_fibers:
            max_fibers = n_fibers
        for fiber_ix in range(n_fibers):
            n_points = len(data[time_ix][fiber_ix])
            if n_points > max_points:
                max_points = n_points
    current_dimensions = traj_data.agent_data.get_dimensions()
    return DimensionData(
        total_steps=0,
        max_agents=max_fibers,
        max_subpoints=(3 * max_points) - current_dimensions.max_subpoints,
    )


def add_fiber_annotation_agents(
    traj_data: TrajectoryData,
    fiber_points: list[list[np.ndarray]],
    type_name: str = "fiber",
    fiber_width: float = 0.5,
    color: str = "#eaeaea",
) -> TrajectoryData:
    """
    Add agent data for fiber annotations.

    Parameters
    ----------
    traj_data
        Trajectory data to add the fibers to.
    fiber_points
        List of lists of arrays (shape = n x 3) containing the x,y,z positions
        of control points for each fiber at each time.
    type_name
        Agent type name to use for the new fibers.
    fiber_width
        Width to draw the fibers.
    color
        Color for the new fibers.

    Returns
    -------
    :
        Updated trajectory data.
    """

    total_steps = len(fiber_points)
    new_agent_data = traj_data.agent_data.get_copy_with_increased_buffer_size(
        _added_dimensions_for_fibers(traj_data, fiber_points)
    )
    max_used_uid = max(list(np.unique(traj_data.agent_data.unique_ids)))
    for time_ix in range(total_steps):
        start_ix = int(traj_data.agent_data.n_agents[time_ix])
        n_fibers = len(fiber_points[time_ix])
        end_ix = start_ix + n_fibers
        for fiber_ix in range(n_fibers):
            agent_ix = start_ix + fiber_ix
            new_agent_data.unique_ids[time_ix][agent_ix] = max_used_uid + fiber_ix + 1
            new_agent_data.n_subpoints[time_ix][agent_ix] = 3 * len(
                fiber_points[time_ix][fiber_ix]
            )
            new_agent_data.subpoints[time_ix][agent_ix] = fiber_points[time_ix][
                fiber_ix
            ].flatten()
        new_agent_data.n_agents[time_ix] += n_fibers
        new_agent_data.viz_types[time_ix][start_ix:end_ix] = n_fibers * [VIZ_TYPE.FIBER]
        new_agent_data.types[time_ix] += n_fibers * [type_name]
        new_agent_data.radii[time_ix][start_ix:end_ix] = n_fibers * [fiber_width]
    new_agent_data.display_data[type_name] = DisplayData(
        name=type_name,
        display_type=DISPLAY_TYPE.FIBER,
        color=color,
    )
    traj_data.agent_data = new_agent_data
    return traj_data


def _added_dimensions_for_spheres(data: list[np.ndarray]) -> DimensionData:
    """
    Get a DimensionData with deltas for each dimension of AgentData.

    Used when adding sphere annotation data.

    Data shape = [timesteps, np.array(spheres, 3)] (assumed to be jagged)
    """
    total_steps = len(data)
    max_spheres = 0
    for time_ix in range(total_steps):
        n_spheres = len(data[time_ix])
        if n_spheres > max_spheres:
            max_spheres = n_spheres
    return DimensionData(
        total_steps=0,
        max_agents=max_spheres,
    )


def add_sphere_annotation_agents(
    traj_data: TrajectoryData,
    sphere_positions: list[np.ndarray],
    type_name: str = "sphere",
    radius: float = 1.0,
    rainbow_colors: bool = False,
    color: str = "#eaeaea",
) -> TrajectoryData:
    """
    Add agent data for sphere annotations.

    Parameters
    ----------
    traj_data
        Trajectory data to add the spheres to.
    sphere_positions
        List of x,y,z positions of spheres to visualize at each time.
    type_name
        Agent type name to use for the new spheres.
    radius
        Radius to draw the spheres
    rainbow_colors
        True to color new spheres in rainbow order, False otherwise.
    color
       Color for the new fibers (if rainbow_colors is False).

    Returns
    -------
    :
        Updated trajectory data.
    """

    total_steps = len(sphere_positions)
    new_agent_data = traj_data.agent_data.get_copy_with_increased_buffer_size(
        _added_dimensions_for_spheres(sphere_positions)
    )
    max_used_uid = max(list(np.unique(traj_data.agent_data.unique_ids)))
    max_spheres = 0
    for time_ix in range(total_steps):
        start_ix = int(traj_data.agent_data.n_agents[time_ix])
        n_spheres = len(sphere_positions[time_ix])
        if n_spheres > max_spheres:
            max_spheres = n_spheres
        end_ix = start_ix + n_spheres
        new_agent_data.unique_ids[time_ix][start_ix:end_ix] = np.arange(
            max_used_uid + 1, max_used_uid + 1 + n_spheres
        )
        new_agent_data.n_agents[time_ix] += n_spheres
        new_agent_data.viz_types[time_ix][start_ix:end_ix] = n_spheres * [
            VIZ_TYPE.DEFAULT
        ]
        new_agent_data.types[time_ix] += [
            f"{type_name}#{ix}" for ix in range(n_spheres)
        ]
        new_agent_data.positions[time_ix][start_ix:end_ix] = sphere_positions[time_ix][
            :n_spheres
        ]
        new_agent_data.radii[time_ix][start_ix:end_ix] = n_spheres * [radius]

    colors = ["#0000ff", "#00ff00", "#ffff00", "#ff0000", "#ff00ff"]
    for ix in range(max_spheres):
        tn = f"{type_name}#{ix}"
        new_agent_data.display_data[tn] = DisplayData(
            name=tn,
            display_type=DISPLAY_TYPE.SPHERE,
            color=colors[ix % len(colors)] if rainbow_colors else color,
        )
    traj_data.agent_data = new_agent_data
    return traj_data
