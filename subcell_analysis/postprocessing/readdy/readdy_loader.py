#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from typing import Any, List, Tuple

import tqdm
import pandas as pd
import numpy as np
import readdy

from .readdy_data import FrameData, TopologyData, ParticleData


class ReaddyLoader:
    def __init__(
        self, 
        h5_file_path: str,
        min_time_ix: int = 0,
        max_time_ix: int = -1,
        time_inc: int = 1,
        timestep: float = 100.,
        save_pickle_file: bool = False,
    ):
        """
        Load and shape data from a ReaDDy trajectory.
        
        Parameters
        ----------
        h5_file_path: str
            Path to the ReaDDy .h5 file. If a .dat pickle file exists 
            at this path, load from that instead.
        """
        self._readdy_trajectory = None
        self._trajectory = None
        self.h5_file_path = h5_file_path
        self.min_time_ix = min_time_ix
        self.max_time_ix = max_time_ix
        self.time_inc = time_inc
        self.timestep = timestep
        self.save_pickle_file = save_pickle_file

    def readdy_trajectory(self) -> readdy.Trajectory:
        """
        Lazy load the ReaDDy trajectory object.
        
        Returns
        ----------
        readdy_trajectory: readdy.Trajectory
            The ReaDDy trajectory object.
        """
        if self._readdy_trajectory is None:
            self._readdy_trajectory = readdy.Trajectory(self.h5_file_path)
        return self._readdy_trajectory

    @staticmethod
    def _edges(time_ix: int, topology_records: Any) -> List[Tuple[int,int]]:
        """
        After a simulation has finished, get all the edges 
        at the given time index as (particle1 id, particle2 id).
        
        topology_records from
        readdy.Trajectory(h5_file_path).read_observable_topologies()
        """
        result = []
        for top in topology_records[time_ix]:
            for e1, e2 in top.edges:
                if e1 <= e2:
                    ix1 = top.particles[e1]
                    ix2 = top.particles[e2]
                    result.append((ix1, ix2))
        return result

    def _shape_trajectory_data(self) -> List[FrameData]:
        """
        Shape data from a ReaDDy trajectory for analysis.
        """
        _, topology_records = self.readdy_trajectory.read_observable_topologies()
        times, types, ids, positions = self.readdy_trajectory.read_observable_particles()
        result = []
        for time_ix in tqdm(range(len(times))):
            if (
                time_ix < self.min_time_ix or 
                (self.max_time_ix >= 0 and time_ix > self.max_time_ix) or 
                time_ix % self.time_inc != 0
            ):
                continue
            frame = FrameData(time=self.timestep * time_ix)
            edges = ReaddyLoader._edges(time_ix, topology_records)
            for index, top in enumerate(topology_records[time_ix]):
                frame.topologies[index] = TopologyData(
                    uid=index,
                    type_name=top.type,
                    particle_ids=top.particles,
                )
            for p in range(len(ids[time_ix])):
                p_id = ids[time_ix][p]
                position = positions[time_ix][p]
                neighbor_ids = []
                for edge in edges:
                    if p_id == edge[0]:
                        neighbor_ids.append(edge[1])
                    elif p_id == edge[1]:
                        neighbor_ids.append(edge[0])
                frame.particles[ids[time_ix][p]] = ParticleData(
                    uid=ids[time_ix][p],
                    type_name=self.readdy_trajectory.species_name(types[time_ix][p]),
                    position=np.array([position[0], position[1], position[2]]),
                    neighbor_ids=neighbor_ids,
                )
            result.append(frame)
        return result

    def trajectory(self) -> List[FrameData]:
        """
        Lazy load the shaped trajectory.
        
        Returns
        ----------
        trajectory:  -> List[FrameData]
            The trajectory of data shaped for analysis.
        """
        if self._trajectory is not None:
            return self._trajectory
        pickle_file_path = self.h5_file_path + ".dat"
        if os.path.isfile(pickle_file_path):
            print("Loading pickle file for ReaDDy data")
            import pickle
            data = []
            with open(pickle_file_path, "rb") as f:
                while True:
                    try:
                        data.append(pickle.load(f))
                    except EOFError:
                        break
            self._trajectory = data[0]
        else:
            print("Loading ReaDDy data from h5 file...")
            self._trajectory = ReaddyLoader._shape_trajectory_data()
            if self.save_pickle_file:
                import pickle
                with open(pickle_file_path, "wb") as file:
                    pickle.dump(self._trajectory, file)
        return self._trajectory
