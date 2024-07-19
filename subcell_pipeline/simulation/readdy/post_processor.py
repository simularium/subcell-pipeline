"""Class for post processing ReaDDy trajectories."""

import math
from typing import Optional

import numpy as np
from tqdm import tqdm

from subcell_pipeline.analysis.compression_metrics.polymer_trace import (
    get_contour_length_from_trace,
)
from subcell_pipeline.analysis.dimensionality_reduction.fiber_data import align_fiber
from subcell_pipeline.simulation.readdy.data_structures import FrameData

ACTIN_START_PARTICLE_PHRASE: list[str] = ["pointed"]
"""Phrases indicating actin start particle."""

ACTIN_PARTICLE_TYPES: list[str] = [
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
"""Actin particle types from simularium/readdy-models."""

IDEAL_ACTIN_POSITIONS: np.ndarray = np.array(
    [
        [24.738, 20.881, 26.671],
        [27.609, 24.061, 27.598],
        [30.382, 21.190, 25.725],
    ]
)
"""Ideal actin positions measured from crystal structure."""

IDEAL_ACTIN_VECTOR_TO_AXIS: np.ndarray = np.array(
    [-0.01056751, -1.47785105, -0.65833209]
)
"""Ideal actin vector to axis."""


class ReaddyPostProcessor:
    """Get different views of ReaDDy trajectory for different analysis purposes."""

    trajectory: list[FrameData]
    """ReaDDy data trajectory from ReaddyLoader(h5_file_path).trajectory()."""

    box_size: np.ndarray
    """The size of the XYZ dimensions of the simulation volume (shape = 3)."""

    periodic_boundary: bool
    """True if simulation had periodic boundary, False otherwise."""

    def __init__(
        self,
        trajectory: list[FrameData],
        box_size: np.ndarray,
        periodic_boundary: bool = False,
    ):
        self.trajectory = trajectory
        self.box_size = box_size
        self.periodic_boundary = periodic_boundary

    def times(self) -> np.ndarray:
        """
        Get simulation time at each timestep.

        Returns
        -------
        times
            Array of time stamps in simulation time for each timestep (shape =
            n_timesteps).
        """
        result = [trajectory.time for trajectory in self.trajectory]
        return np.array(result)

    def _id_for_neighbor_of_types(
        self,
        time_ix: int,
        particle_id: int,
        neighbor_types: list[str],
        exclude_ids: Optional[list[int]] = None,
    ) -> int:
        """
        Get the id for the first neighbor with a type_name in neighbor_types at
        the given time index.
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
        start_particle_id: int,
        chain_particle_types: list[list[str]],
        current_polymer_number: int,
        chain_length: int = 0,
        last_particle_id: Optional[int] = None,
        result: Optional[list[int]] = None,
    ) -> list[int]:
        """
        Get IDs for a chain of particles with chain_particle_types in the given
        frame of data, starting from the particle with start_particle_id and
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
            neighbor_types=chain_particle_types[current_polymer_number],
            exclude_ids=[last_particle_id] if last_particle_id is not None else [],
        )
        if neighbor_id < 0:
            return result
        result.append(neighbor_id)
        return self._ids_for_chain_of_types(
            time_ix=time_ix,
            start_particle_id=neighbor_id,
            chain_particle_types=chain_particle_types,
            current_polymer_number=(
                (current_polymer_number + 1) % len(chain_particle_types)
            ),
            chain_length=chain_length - 1 if chain_length > 0 else 0,
            last_particle_id=start_particle_id,
            result=result,
        )

    def _non_periodic_position(
        self, position1: np.ndarray, position2: np.ndarray
    ) -> np.ndarray:
        """
        If the distance between two positions is greater than box_size, move the
        second position across the box.
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
        """Check if any of a 3D vector's components are NaN."""
        return math.isnan(vector[0]) or math.isnan(vector[1]) or math.isnan(vector[2])

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        """Normalize a vector."""
        if vector[0] == 0 and vector[1] == 0 and vector[2] == 0:
            return vector
        return vector / np.linalg.norm(vector)

    @staticmethod
    def _orientation_from_positions(positions: np.ndarray) -> np.ndarray:
        """
        Orthonormalize and cross the vectors from a particle position to prev
        and next particle positions to get a basis local to the particle.

        The positions array is structured as:
            [
                prev particle's position,
                this particle's position,
                next particle's position,
            ]
        """
        v1 = ReaddyPostProcessor._normalize(positions[0] - positions[1])
        v2 = ReaddyPostProcessor._normalize(positions[2] - positions[1])
        v2 = ReaddyPostProcessor._normalize(v2 - (np.dot(v1, v2) / np.dot(v1, v1)) * v1)
        v3 = np.cross(v2, v1)
        return np.array(
            [[v1[0], v2[0], v3[0]], [v1[1], v2[1], v3[1]], [v1[2], v2[2], v3[2]]]
        )

    def _rotation(
        self, positions: np.ndarray, ideal_positions: np.ndarray
    ) -> np.ndarray:
        """
        Get the difference in the particles's current orientation compared to
        the initial orientation as a rotation matrix.

        The positions array is structured as:
            [
                prev particle's position,
                this particle's position,
                next particle's position,
            ]
        """
        positions[0] = self._non_periodic_position(positions[1], positions[0])
        positions[2] = self._non_periodic_position(positions[1], positions[2])
        return np.matmul(
            self._orientation_from_positions(positions),
            np.linalg.inv(self._orientation_from_positions(ideal_positions)),
        )

    def rotate_positions(
        self, positions: np.ndarray, rotation: np.ndarray
    ) -> np.ndarray:
        """
        Rotate an x,y,z position (or an array of them) around the x-axis
        with the given rotation matrix.
        """
        if len(positions.shape) > 1:
            result = np.dot(positions[:, 1:], rotation)
            return np.concatenate((positions[:, 0:1], result), axis=1)
        else:
            result = np.dot(positions[1:], rotation)
            return np.concatenate((positions[0:1], result), axis=0)

    def align_trajectory(
        self,
        fiber_points: list[list[np.ndarray]],
    ) -> tuple[np.ndarray, list[list[np.ndarray]]]:
        """
        Align the positions of particles in the trajectory
        so that the furthest point from the x-axis
        is aligned with the positive y-axis at the last time point.

        Parameters
        ----------
        fiber_points
            How many numbers are used to represent the relative identity of
            particles in the chain?
        start_particle_phrases
            List of phrases in particle type names for the first particles in
            the linear chain.
        other_particle_types
            List of particle type names (without polymer numbers at the end) for
            the particles other than the start particles.

        Returns
        -------
        positions
            Array (shape = timesteps x 1 x n x 3) containing the x,y,z positions
            of actin monomer particles at each timestep.
        fiber_points
            List of lists of arrays (shape = n x 3) containing the x,y,z positions
            of control points for each fiber at each time.
        """
        result = []
        _, rotation = align_fiber(fiber_points[-1][0])
        for time_ix in range(len(self.trajectory)):
            result.append([])
            for _, particle in self.trajectory[time_ix].particles.items():
                particle.position = self.rotate_positions(particle.position, rotation)
                result[time_ix].append(particle.position)
            fiber_points[time_ix][0] = self.rotate_positions(
                fiber_points[time_ix][0], rotation
            )
        return np.array(result), fiber_points

    def linear_fiber_chain_ids(
        self,
        polymer_number_range: int,
        start_particle_phrases: list[str] = ACTIN_START_PARTICLE_PHRASE,
        other_particle_types: list[str] = ACTIN_PARTICLE_TYPES,
    ) -> list[list[list[int]]]:
        """
        Get particle IDs for particles in each linear fiber at each timestep.

        Parameters
        ----------
        polymer_number_range
            How many numbers are used to represent the relative identity of
            particles in the chain?
        start_particle_phrases
            List of phrases in particle type names for the first particles in
            the linear chain.
        other_particle_types
            List of particle type names (without polymer numbers at the end) for
            the particles other than the start particles.

        Returns
        -------
        :
            List of lists of lists of the particle IDs for each particle for
            each fiber at each time.
        """
        result: list[list[list[int]]] = []
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
                result[time_ix].append(chain_ids)
        return result

    def linear_fiber_axis_positions(
        self,
        fiber_chain_ids: list[list[list[int]]],
        ideal_positions: np.ndarray = IDEAL_ACTIN_POSITIONS,
        ideal_vector_to_axis: np.ndarray = IDEAL_ACTIN_VECTOR_TO_AXIS,
    ) -> tuple[list[list[np.ndarray]], list[list[list[int]]]]:
        """
        Get XYZ axis positions for each particle in each linear fiber at each
        timestep.

        Parameters
        ----------
        fiber_chain_ids
            List of list of lists of particle IDs for each particle in each
            fiber at each time.
        ideal_positions
            XYZ positions for 3 particles in an ideal chain (shape = 3 x 3).
        ideal_vector_to_axis
            Vector from the second ideal position to the axis of the fiber
            (shape = 3).

        Returns
        -------
        axis_positions
            Lists of lists of arrays (shape = n x 3) containing the x,y,z
            positions of the closest point on the fiber axis to the position of
            each particle in each fiber at each time.
        new_chain_ids
            List of lists of lists of particle IDs matching the axis_positions
            for each particle in each fiber at each time.
        """
        result: list[list[np.ndarray]] = []
        ids: list[list[list[int]]] = []
        for time_ix in range(len(fiber_chain_ids)):
            result.append([])
            ids.append([])
            for fiber_ix in range(len(fiber_chain_ids[time_ix])):
                axis_positions = []
                new_ids = []
                particles = self.trajectory[time_ix].particles
                chain_ids = fiber_chain_ids[time_ix][fiber_ix]
                for particle_ix in range(1, len(chain_ids) - 1):
                    positions = [
                        particles[chain_ids[particle_ix - 1]].position,
                        particles[chain_ids[particle_ix]].position,
                        particles[chain_ids[particle_ix + 1]].position,
                    ]
                    pos_invalid = False
                    for pos in positions:
                        if self._vector_is_invalid(pos):
                            pos_invalid = True
                            break
                    if pos_invalid:
                        break
                    rotation = self._rotation(np.array(positions), ideal_positions)
                    if rotation is None:
                        break
                    vector_to_axis_local = np.squeeze(
                        np.array(np.dot(rotation, ideal_vector_to_axis))
                    )
                    axis_pos = positions[1] + vector_to_axis_local
                    if self._vector_is_invalid(axis_pos):
                        break
                    axis_positions.append(axis_pos)
                    new_ids.append(particle_ix)
                if len(axis_positions) < 2:
                    continue
                result[time_ix].append(np.array(axis_positions))
                ids[time_ix].append(new_ids)
        return result, ids

    def linear_fiber_normals(
        self,
        fiber_chain_ids: list[list[list[int]]],
        axis_positions: list[list[np.ndarray]],
        normal_length: float = 5,
    ) -> list[list[np.ndarray]]:
        """
        Get XYZ positions defining start and end points for normals for each
        particle in each fiber at each timestep.

        Parameters
        ----------
        fiber_chain_ids
            List of lists of lists of particle IDs for particles in each fiber
            at each time.
        axis_positions
            List of lists of arrays (shape = n x 3) containing the x,y,z
            positions of the closest point on the fiber axis to the position of
            each particle in each fiber at each time.
        normal_length
            Length of the resulting normal vectors in the trajectory's spatial
            units.

        Returns
        -------
        :
            List of lists of arrays (shape = 2 x 3) containing the x,y,z normals
            of each particle in each fiber at each time.
        """
        result: list[list[np.ndarray]] = []
        for time_ix in range(len(fiber_chain_ids)):
            result.append([])
            particles = self.trajectory[time_ix].particles
            for chain_ix in range(len(fiber_chain_ids[time_ix])):
                n_particles = len(fiber_chain_ids[time_ix][chain_ix])
                for particle_ix, particle_id in enumerate(
                    fiber_chain_ids[time_ix][chain_ix]
                ):
                    # Skip first and last particle
                    if particle_ix == 0 or particle_ix == n_particles - 1:
                        continue
                    position = particles[particle_id].position
                    axis_position = axis_positions[time_ix][chain_ix][particle_ix - 1]
                    direction = ReaddyPostProcessor._normalize(position - axis_position)
                    result[time_ix].append(
                        np.array(
                            [axis_position, axis_position + normal_length * direction]
                        )
                    )
        return result

    @staticmethod
    def linear_fiber_control_points(
        axis_positions: list[list[np.ndarray]],
        n_points: int,
    ) -> list[list[np.ndarray]]:
        """
        Resample the fiber line defined by each array of axis positions to get
        the requested number of points between XYZ control points for each
        linear fiber at each timestep.

        Parameters
        ----------
        axis_positions
            List of lists of arrays (shape = n x 3) containing the x,y,z
            positions of the closest point on the fiber axis to the position of
            each particle in each fiber at each time.
        n_points
            Number of control points (spaced evenly) on resulting fibers.

        Returns
        -------
        :
            List of lists of arrays (shape = n x 3) containing the x,y,z positions
            of control points for each fiber at each time.
        """
        if n_points < 2:
            raise Exception("n_points must be > 1 to define a fiber.")
        result: list[list[np.ndarray]] = []
        for time_ix in tqdm(range(len(axis_positions))):
            result.append([])
            contour_length = get_contour_length_from_trace(axis_positions[time_ix][0])
            segment_length = contour_length / float(n_points - 1)
            pt_ix = 1
            for positions in axis_positions[time_ix]:
                control_points = np.zeros((n_points, 3))
                control_points[0] = positions[0]
                current_position = np.copy(positions[0])
                leftover_length: float = 0
                for pos_ix in range(1, len(positions)):
                    v_segment = positions[pos_ix] - positions[pos_ix - 1]
                    direction = ReaddyPostProcessor._normalize(v_segment)
                    remaining_length = (
                        np.linalg.norm(v_segment).item() + leftover_length
                    )
                    # Rounding to 9 decimal places to avoid floating point error
                    # where the remaining length is very close to the segment
                    # length, causeing the final control point to be skipped.
                    while round(remaining_length, 9) >= round(segment_length, 9):
                        current_position += (
                            segment_length - leftover_length
                        ) * direction
                        control_points[pt_ix, :] = current_position[:]
                        pt_ix += 1
                        leftover_length = 0
                        remaining_length -= segment_length
                    current_position += (remaining_length - leftover_length) * direction
                    leftover_length = remaining_length
                result[time_ix].append(control_points)
        return result

    def fiber_bond_energies(
        self,
        fiber_chain_ids: list[list[list[int]]],
        ideal_lengths: dict[int, float],
        ks: dict[int, float],
        stride: int = 1,
    ) -> tuple[dict[int, np.ndarray], np.ndarray]:
        """
        Get the strain energy using the harmonic spring equation and the bond
        distance between particles with a given polymer number offset.

        Parameters
        ----------
        fiber_chain_ids
            List of lists of lists of particle IDs for particles in each fiber
            at each time.
        ideal_lengths
            Ideal bond length for each of the polymer number offsets.
        ks
            Bond energy constant for each of the polymer number offsets.
        stride
            Calculate bond energy every stride timesteps.

        Returns
        -------
        bond_energies
            Map of polymer number offset to array (shape = time x bonds) of bond
            energy for each bond at each time.
        filament_positions
            Array (shape = time x bonds) of position in the filament from the
            starting end for the first particle in each bond at each time.
        """
        energies: dict[int, list[list[float]]] = {}
        for offset in ideal_lengths:
            energies[offset] = []
        filament_positions: list[list[int]] = []
        for time_ix in range(0, len(self.trajectory), stride):
            for offset in ideal_lengths:
                energies[offset].append([])
            filament_positions.append([])
            particles = self.trajectory[time_ix].particles
            new_time = math.floor(time_ix / stride)
            for fiber_ix in range(len(fiber_chain_ids[time_ix])):
                fiber_ids = fiber_chain_ids[time_ix][fiber_ix]
                for ix in range(len(fiber_ids) - 2):
                    particle = particles[fiber_ids[ix]]
                    if "fixed" in particle.type_name:
                        continue
                    for offset in ideal_lengths:
                        offset_particle = particles[fiber_ids[ix + offset]]
                        if "fixed" in offset_particle.type_name:
                            continue
                        offset_pos = self._non_periodic_position(
                            particle.position, offset_particle.position
                        )
                        bond_stretch = (
                            np.linalg.norm(offset_pos - particle.position).item()
                            - ideal_lengths[offset]
                        )
                        energy = 0.5 * ks[offset] * bond_stretch * bond_stretch
                        if math.isnan(energy):
                            energy = 0.0
                        energies[offset][new_time].append(energy)
                    filament_positions[new_time].append(ix)
        return (
            {offset: np.array(energy) for offset, energy in energies.items()},
            np.array(filament_positions),
        )

    def edge_positions(self) -> list[list[np.ndarray]]:
        """
        Get the edges between particles as start and end positions.

        Returns
        -------
        :
            List of list of edges as position of each of the two connected particles
            for each edge at each time.
        """
        edges = []
        for frame in self.trajectory:
            edges.append([])
            for edge in frame.edge_ids:
                edges[-1].append(
                    np.array(
                        [
                            frame.particles[edge[0]].position,
                            frame.particles[edge[1]].position,
                        ]
                    )
                )
        return edges
