#!/usr/bin/env python

from typing import List

import numpy as np
from simulariumio import DISPLAY_TYPE, DimensionData, DisplayData, TrajectoryData
from simulariumio.constants import VIZ_TYPE


class SpatialAnnotator:
    @staticmethod
    def _added_dimensions_for_fibers(
        traj_data: TrajectoryData,
        data: List[List[np.ndarray]],
    ) -> DimensionData:
        """
        Get a DimensionData with the deltas for each dimension
        of AgentData when adding the given fiber data.

        Data shape = [timesteps, fibers, np.array(points, 3)]
        (assumed to be jagged)
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

    @staticmethod
    def add_fiber_agents(
        traj_data: TrajectoryData,
        fiber_points: List[List[np.ndarray]],
        type_name: str = "fiber",
        fiber_width: float = 0.5,
        color: str = "#eaeaea",
    ) -> TrajectoryData:
        """
        Add agent data for fibers.


        Parameters
        ----------
        traj_data: TrajectoryData
            Trajectory data to add the fibers to.
        fiber_points: List[List[np.ndarray (shape = n x 3)]]
            List of lists of arrays containing
            the x,y,z positions of control points
            for each fiber at each time.
        type_name: str (optional)
            Agent type name to use for the new fibers.
            Default: "fiber"
        fiber_width: float (optional)
            Width to draw the fibers.
            Default: 0.5
        color: str (optional)
            Color for the new fibers.
            Default: "#eaeaea"
        """
        total_steps = len(fiber_points)
        new_agent_data = traj_data.agent_data.get_copy_with_increased_buffer_size(
            SpatialAnnotator._added_dimensions_for_fibers(traj_data, fiber_points)
        )
        max_used_uid = max(list(np.unique(traj_data.agent_data.unique_ids)))
        for time_ix in range(total_steps):
            start_ix = int(traj_data.agent_data.n_agents[time_ix])
            n_fibers = len(fiber_points[time_ix])
            end_ix = start_ix + n_fibers
            for fiber_ix in range(n_fibers):
                agent_ix = start_ix + fiber_ix
                new_agent_data.unique_ids[time_ix][agent_ix] = (
                    max_used_uid + fiber_ix + 1
                )
                new_agent_data.n_subpoints[time_ix][agent_ix] = 3 * len(
                    fiber_points[time_ix][fiber_ix]
                )
                new_agent_data.subpoints[time_ix][agent_ix] = fiber_points[time_ix][
                    fiber_ix
                ].flatten()
            new_agent_data.n_agents[time_ix] += n_fibers
            new_agent_data.viz_types[time_ix][start_ix:end_ix] = n_fibers * [
                VIZ_TYPE.FIBER
            ]
            new_agent_data.types[time_ix] += n_fibers * [type_name]
            new_agent_data.radii[time_ix][start_ix:end_ix] = n_fibers * [fiber_width]
        new_agent_data.display_data[type_name] = DisplayData(
            name=type_name,
            display_type=DISPLAY_TYPE.FIBER,
            color=color,
        )
        traj_data.agent_data = new_agent_data
        return traj_data

    @staticmethod
    def _added_dimensions_for_spheres(
        data: List[np.ndarray],
    ) -> DimensionData:
        """
        Get a DimensionData with the deltas for each dimension
        of AgentData when adding the given sphere data.

        Data shape = [timesteps, np.array(spheres, 3)]
        (assumed to be jagged)
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

    @staticmethod
    def add_sphere_agents(
        traj_data: TrajectoryData,
        sphere_positions: List[np.ndarray],
        type_name: str = "sphere",
        radius: float = 1.0,
        color: str = "#eaeaea",
    ) -> TrajectoryData:
        """
        Add agent data for fibers.


        Parameters
        ----------
        traj_data: TrajectoryData
            Trajectory data to add the spheres to.
        sphere_positions: List[np.ndarray (shape = n x 3)]
            List of x,y,z positions of spheres to visualize
            at each time.
        type_name: str (optional)
            Agent type name to use for the new spheres.
            Default: "sphere"
        radius: float (optional)
            Radius to draw the spheres.
            Default: 1.
        color: str (optional)
            Color for the new spheres.
            Default: "#eaeaea"
        """
        total_steps = len(sphere_positions)
        new_agent_data = traj_data.agent_data.get_copy_with_increased_buffer_size(
            SpatialAnnotator._added_dimensions_for_spheres(sphere_positions)
        )
        max_used_uid = max(list(np.unique(traj_data.agent_data.unique_ids)))
        for time_ix in range(total_steps):
            start_ix = int(traj_data.agent_data.n_agents[time_ix])
            n_spheres = len(sphere_positions[time_ix])
            end_ix = start_ix + n_spheres
            new_agent_data.unique_ids[time_ix][start_ix:end_ix] = np.arange(
                max_used_uid + 1, max_used_uid + 1 + n_spheres
            )
            new_agent_data.n_agents[time_ix] += n_spheres
            new_agent_data.viz_types[time_ix][start_ix:end_ix] = n_spheres * [
                VIZ_TYPE.DEFAULT
            ]
            new_agent_data.types[time_ix] += n_spheres * [type_name]
            new_agent_data.positions[time_ix][start_ix:end_ix] = sphere_positions[
                time_ix
            ][:n_spheres]
            new_agent_data.radii[time_ix][start_ix:end_ix] = n_spheres * [radius]
        new_agent_data.display_data[type_name] = DisplayData(
            name=type_name,
            display_type=DISPLAY_TYPE.SPHERE,
            color=color,
        )
        traj_data.agent_data = new_agent_data
        return traj_data
