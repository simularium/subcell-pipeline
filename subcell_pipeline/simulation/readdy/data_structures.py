"""Data structures for ReaDDy simulations."""

from typing import Optional

import numpy as np


class TopologyData:
    """Data class representing a ReaDDy topology of connected particles."""

    uid: int
    """Unique ID of the topology from ReaDDy."""

    type_name: str
    """ReaDDy type name of the topology."""

    particle_ids: list[int]
    """List of unique IDs of each particle in the topology."""

    def __init__(self, uid: int, type_name: str, particle_ids: list[int]):
        self.uid = uid
        self.type_name = type_name
        self.particle_ids = particle_ids

    def __str__(self) -> str:
        return (
            "Topology(\n"
            f"  id = {self.uid}\n"
            f"  type_name = {self.type_name}\n"
            f"  particles = {self.particle_ids}\n"
            ")"
        )


class ParticleData:
    """Data class representing a ReaDDy particle."""

    uid: int
    """Unique ID of the particle from ReaDDy."""

    type_name: str
    """ReaDDy type name of the particle."""

    position: np.ndarray
    """XYZ position of the particle."""

    neighbor_ids: list[int]
    """List of unique IDs of each neighbor particle connected by an edge."""

    def __init__(
        self, uid: int, type_name: str, position: np.ndarray, neighbor_ids: list[int]
    ):
        self.uid = uid
        self.type_name = type_name
        self.position = position
        self.neighbor_ids = neighbor_ids

    def __str__(self) -> str:
        return (
            f"Particle(\n"
            f"  id = {self.uid}\n"
            f"  type_name = {self.type_name}\n"
            f"  position = {self.position}\n"
            f"  neighbors = {self.neighbor_ids}\n"
            ")"
        )


class FrameData:
    """Data class representing one ReaDDy timestep."""

    time: float
    """Current time of the simulation for this frame."""

    topologies: dict[int, TopologyData]
    """Mapping of topology ID to a TopologyData for each topology."""

    particles: dict[int, ParticleData]
    """Mapping of particle ID to a ParticleData for each particle."""

    edge_ids: list[list[int]]
    """List of edges, each is a list of the IDs of the two connected particles."""

    def __init__(
        self,
        time: float,
        topologies: Optional[dict[int, TopologyData]] = None,
        particles: Optional[dict[int, ParticleData]] = None,
        edge_ids: Optional[list[list[int]]] = None,
    ):
        self.time = time
        self.topologies = topologies if topologies is not None else {}
        self.particles = particles if particles is not None else {}
        self.edge_ids = edge_ids if edge_ids is not None else []

    def __str__(self) -> str:
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
