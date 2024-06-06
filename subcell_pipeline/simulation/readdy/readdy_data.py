#!/usr/bin/env python

from typing import Dict, List

import numpy as np


class TopologyData:
    uid: int
    type_name: str
    particle_ids: List[int]

    def __init__(self, uid: int, type_name: str, particle_ids: List[int]):
        """
        Data class representing a ReaDDy topology of connected particles.


        Parameters
        ----------
        uid: int
            Unique ID of the topology from ReaDDy.
        type_name: str
            ReaDDy type name of the topology.
        particle_ids: List[int]
            List of unique IDs of each particle in the topology.
        """
        self.uid = uid
        self.type_name = type_name
        self.particle_ids = particle_ids

    def __str__(self) -> str:
        """String with all data."""
        return (
            "Topology(\n"
            f"  id = {self.uid}\n"
            f"  type_name = {self.type_name}\n"
            f"  particles = {self.particle_ids}\n"
            ")"
        )


class ParticleData:
    uid: int
    type_name: str
    position: np.ndarray
    neighbor_ids: List[int]

    def __init__(
        self, uid: int, type_name: str, position: np.ndarray, neighbor_ids: List[int]
    ):
        """
        Data class representing a ReaDDy particle.


        Parameters
        ----------
        uid: int
            Unique ID of the particle from ReaDDy.
        type_name: str
            ReaDDy type name of the particle.
        position: np.ndarray
            XYZ position of the particle.
        neighbor_ids: List[int]
            List of unique IDs of each neighbor particle
            connected by an edge.
        """
        self.uid = uid
        self.type_name = type_name
        self.position = position
        self.neighbor_ids = neighbor_ids

    def __str__(self) -> str:
        """String with all data."""
        return (
            f"Particle(\n"
            f"  id = {self.uid}\n"
            f"  type_name = {self.type_name}\n"
            f"  position = {self.position}\n"
            f"  neighbors = {self.neighbor_ids}\n"
            ")"
        )


class FrameData:
    time: float
    topologies: Dict[int, TopologyData]
    particles: Dict[int, ParticleData]
    edges: List[np.ndarray]

    def __init__(
        self,
        time: float,
        topologies: Dict[int, TopologyData] = None,
        particles: Dict[int, ParticleData] = None,
        edges: List[np.ndarray] = None,
    ):
        """
        Data class representing one ReaDDy timestep.


        Parameters
        ----------
        time: float
            Current time of the simulation for this frame.
        topologies: Dict[int, TopologyData] (optional)
            Mapping of topology ID to a TopologyData for each topology.
            Default: {} (added by ReaddyLoader._shape_trajectory_data())
        particles: Dict[int, ParticleData] (optional)
            Mapping of particle ID to a ParticleData for each particle.
            Default: {} (added by ReaddyLoader._shape_trajectory_data())
        edges: List[np.ndarray (shape = 2 x 3)] (optional)
            List of edges as position of each of the two particles
            connected by the edge.
            Default: [] (added by ReaddyLoader._shape_trajectory_data())
        """
        self.time = time
        self.topologies = topologies if topologies is not None else {}
        self.particles = particles if particles is not None else {}
        self.edges = edges if edges is not None else []

    def __str__(self) -> str:
        """String with topology and particle data."""
        top_str = "\n"
        for top_id in self.topologies:
            top_str += f"{top_id} : \n{self.topologies[top_id]}\n"
        p_str = "\n"
        for p_id in self.particles:
            p_str += f"{p_id} : \n{self.particles[p_id]}\n"
        return (
            f"Frame(\n"
            f"  time={self.time}\n"
            f"  topologies=\n{top_str}\n\n"
            f"  particles=\n{p_str}\n\n"
            ")"
        )
