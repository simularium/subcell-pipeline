#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List
import math

import numpy as np

from .readdy_data import FrameData


class ReaddyPostProcessor:
    def __init__(
        self, 
        trajectory: List[FrameData],
        box_size: np.ndarray,
        periodic_boundary: bool = False,
    ):
        """
        Get different views of the ReaDDy trajectory 
        for different analysis purposes.
        
        Parameters
        ----------
        trajectory: List[FrameData]
            A trajectory of ReaDDy data from 
            ReaddyLoader(h5_file_path).trajectory().
        box_size: np.ndarray (shape = 3)
            The size of the XYZ dimensions of the simulation volume.
        periodic_boundary: bool (Optional)
            Was there a periodic boundary in this simulation?
            Default: False
        """
        self.trajectory = trajectory
        self.box_size = box_size
        self.periodic_boundary = periodic_boundary

    def _id_for_neighbor_of_types(
        self,
        time_ix: int,
        particle_id: int,
        neighbor_types: List[str],
        exclude_ids: List[int] = None,
    ) -> int:
        """
        Get the id for the first neighbor 
        with a type_name in neighbor_types
        at the given time index.
        """
        particles = self.trajectory[time_ix].particles
        for neighbor_id in particles[particle_id].neighbor_ids:
            if exclude_ids is not None and neighbor_id in exclude_ids:
                continue
            for neighbor_type in neighbor_types:
                if neighbor_type == particles[neighbor_id].type_name:
                    return neighbor_id
        return -1

    def _ids_for_chain_of_types(
        self,
        time_ix: int,
        start_particle_id,
        chain_particle_types,
        current_polymer_number,
        chain_length=0,
        last_particle_id=None,
        result=None,
    ) -> List[int]:
        """
        Starting from the particle with start_particle_id,
        get ids for a chain of particles with chain_particle_types 
        in the given frame of data, 
        avoiding the particle with last_particle_id.
        If chain_length = 0, return entire chain.
        """
        if result is None:
            result = [start_particle_id]
        if chain_length == 1:
            return result
        neighbor_id = self._id_for_neighbor_of_types(
            time_ix=time_ix,
            particle_id=start_particle_id,
            chain_particle_types=chain_particle_types[current_polymer_number + 1],
            exclude_ids=[last_particle_id] if last_particle_id is not None else [],
        )
        if neighbor_id is None:
            return result
        result.append(neighbor_id)
        return self._ids_for_chain_of_types(
            time_ix=time_ix,
            start_particle_id=neighbor_id,
            chain_particle_types=chain_particle_types,
            current_polymer_number=(current_polymer_number + 1) % len(chain_particle_types),
            chain_length=chain_length - 1 if chain_length > 0 else 0,
            last_particle_id=start_particle_id,
            result=result,
        )

    def _non_periodic_position(self, position1: np.ndarray, position2: np.ndarray) -> np.ndarray:
        """
        If the distance between two positions is greater than box_size,
        move the second position across the box.
        """
        if not self.periodic_boundary:
            return position2
        result = np.copy(position2)
        for dim in range(3):
            if abs(position2[dim] - position1[dim]) > self.box_size[dim] / 2.0:
                result[dim] -= position2[dim] / abs(position2[dim]) * self.box_size[dim]
        return result

    @staticmethod
    def _vector_is_invalid(vector: np.ndarray) -> bool:
        """
        check if any of a 3D vector's components are NaN.
        """
        return math.isnan(vector[0]) or math.isnan(vector[1]) or math.isnan(vector[2])

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        """
        normalize a vector.
        """
        if vector[0] == 0 and vector[1] == 0 and vector[2] == 0:
            return vector
        return vector / np.linalg.norm(vector)

    @staticmethod
    def _orientation_from_positions(positions: np.ndarray) -> np.ndarray:
        """
        orthonormalize and cross the vectors from a particle position
        to prev and next particle positions to get a basis local to the particle.
        
        positions = [
            prev particle's position, 
            this particle's position, 
            next particle's position
        ]
        """
        v1 = ReaddyPostProcessor._normalize(positions[0] - positions[1])
        v2 = ReaddyPostProcessor._normalize(positions[2] - positions[1])
        v2 = ReaddyPostProcessor._normalize(v2 - (np.dot(v1, v2) / np.dot(v1, v1)) * v1)
        v3 = np.cross(v2, v1)
        return np.array(
            [[v1[0], v2[0], v3[0]], [v1[1], v2[1], v3[1]], [v1[2], v2[2], v3[2]]]
        )

    def _rotation(self, positions: np.ndarray, ideal_positions: np.ndarray) -> np.ndarray:
        """
        get the difference in the particles's current orientation
        compared to the initial orientation as a rotation matrix.
        
        positions = [
            prev particle's position, 
            this particle's position, 
            next particle's position
        ]
        """
        positions[0] = self._non_periodic_position(positions[1], positions[0])
        positions[2] = self._non_periodic_position(positions[1], positions[2])
        return np.matmul(
            self._orientation_from_positions(positions), 
            np.linalg.inv(
                self._orientation_from_positions(ideal_positions)
            )
        )

    def _axis_position(
        self, 
        positions: np.ndarray, 
        ideal_positions: np.ndarray, 
        ideal_vector_to_axis: np.ndarray
    ) -> np.ndarray:
        """
        get the position on the fiber axis closest to a particle.
        
        positions = [
            prev particle's position, 
            this particle's position, 
            next particle's position
        ]
        """
        rotation = self._rotation(positions, ideal_positions)
        if rotation is None:
            return None
        vector_to_axis_local = np.squeeze(
            np.array(np.dot(rotation, ideal_vector_to_axis))
        )
        return positions[1] + vector_to_axis_local

    def _chain_axis_positions(
        self, 
        time_ix: int,
        chain_ids: List[int],
        ideal_positions: np.ndarray,
        ideal_vector_to_axis: np.ndarray,
    ) -> np.ndarray:
        """
        get the position on the fiber axis closest 
        to each particle in the chain.
        """
        result = []
        particles = self.trajectory[time_ix].particles
        for ix in range(1, len(chain_ids) - 1):
            positions = [
                particles[chain_ids[ix - 1]].position,
                particles[chain_ids[ix]].position,
                particles[chain_ids[ix + 1]].position,
            ]
            axis_pos = self._axis_position(
                positions, ideal_positions, ideal_vector_to_axis
            )
            if self._vector_is_invalid(axis_pos):
                break
            result.append(axis_pos)
        if len(result) < 2:
            return None
        return result

    @staticmethod
    def _control_points(
        axis_positions: List[np.ndarray],
        segment_length: float,
    ) -> np.ndarray:
        """
        get the control points along the fiber 
        defined by the axis positions, each is segment_length apart.
        """
        control_points = [axis_positions[0]]
        current_length = 0
        for axis_ix in range(1, len(axis_positions)):
            prev_pos = axis_positions[axis_ix - 1]
            v_segment = axis_positions[axis_ix] - prev_pos
            distance = np.linalg.norm(v_segment)
            if current_length + distance > segment_length:
                remaining_length = segment_length - current_length
                direction = ReaddyPostProcessor._normalize(v_segment)
                new_point = prev_pos + remaining_length * direction
                control_points.append(new_point)
                current_length = distance - remaining_length
            else:
                current_length += distance
        return np.array(control_points)

    def linear_fiber_points(
        self, 
        start_particle_phrases: List[str],
        other_particle_types: List[str],
        polymer_number_range: int,
        ideal_positions: np.ndarray,
        ideal_vector_to_axis: np.ndarray,
        segment_length: float,
    ) -> List[List[np.ndarray]]:
        """
        Get XYZ control points for each linear fiber
        at each timestep.
        
        Parameters
        ----------
        start_particle_phrases: List[str]
            List of phrases in particle type names
            for the first particles in the linear chain.
        other_particle_types: List[str]
            List of particle type names 
            (without polymer numbers at the end)
            for the particles other than the start particles.
        polymer_number_range: int
            How many numbers are used to represent the
            relative identity of particles in the chain?
        ideal_positions: np.ndarray (shape = 3 x 3)
            XYZ positions for 3 particles in an ideal chain.
        ideal_vector_to_axis: np.ndarray
            Vector from the second ideal position 
            to the axis of the fiber.
        segment_length: float
            Length of segments between control points
            on resulting fibers.

        Returns
        ----------
        fiber_points: List[List[np.ndarray (shape = n x 3)]]
            Array containing the x,y,z positions 
            of fiber points for each fiber at each time.
        """
        result = []
        chain_particle_types = []
        for i in range(polymer_number_range):
            chain_particle_types.append(
                [f"{type_name}{i + 1}" for type_name in other_particle_types]
            )
        for time_ix in range(len(self.trajectory)):
            particles = self.trajectory[time_ix].particles
            result.append([])
            for particle_id in particles:
                # check if this particle is the start of a chain
                is_start_particle = False
                for phrase in start_particle_phrases:
                    if phrase in particles[particle_id].type_name:
                        is_start_particle = True
                        break
                if not is_start_particle:
                    continue
                # get ids for particles in the chain
                chain_ids = self._ids_for_chain_of_types(
                    time_ix=time_ix,
                    start_particle_id=particle_id,
                    chain_particle_types=chain_particle_types,
                    current_polymer_number=int(particles[particle_id].type_name[-1]),
                )
                if len(chain_ids) < 2:
                    continue
                # get axis positions for the chain particles
                axis_positions = self._chain_axis_positions(
                    time_ix=time_ix, 
                    chain_ids=chain_ids, 
                    ideal_positions=ideal_positions, 
                    ideal_vector_to_axis=ideal_vector_to_axis,
                )
                if axis_positions is None:
                    continue
                # resample the fiber line to get the requested segment length
                control_points = self._control_points(axis_positions, segment_length)
                result[time_ix].append(control_points)
        return result
