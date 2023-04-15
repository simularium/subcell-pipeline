#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Dict, List

import numpy as np


class TopologyData:
    uid: int
    type_name: str
    particle_ids: List[int]
    
    def __init__(self, uid: int, type_name: str, particle_ids: List[int]):
        """
        Data class representing a ReaDDy topology of connected particles.
        """
        self.uid = uid
        self.type_name = type_name
        self.particle_ids = particle_ids


class ParticleData:
    uid: int
    type_name: str
    position: np.ndarray
    neighbor_ids: List[int]
    
    def __init__(self, uid: int, type_name: str, position: np.ndarray, neighbor_ids: List[int]):
        """
        Data class representing a ReaDDy particle.
        """
        self.uid = uid
        self.type_name = type_name
        self.position = position
        self.neighbor_ids = neighbor_ids


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
        """
        self.time = time
        self.topologies = topologies if topologies is not None else {}
        self.particles = particles if particles is not None else {}
        self.edges = edges if edges is not None else []
