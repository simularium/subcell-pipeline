"""Class for loading and shaping ReaDDy trajectories."""

from typing import Any, Optional

import numpy as np
import readdy
from io_collection.keys.check_key import check_key
from io_collection.load.load_pickle import load_pickle
from io_collection.save.save_pickle import save_pickle
from tqdm import tqdm

from subcell_pipeline.simulation.readdy.data_structures import (
    FrameData,
    ParticleData,
    TopologyData,
)


class ReaddyLoader:
    """
    Load and shape data from a ReaDDy trajectory.

    Trajectory is loaded from the simulation output h5 file of the .dat pickle
    file. If a .dat pickle location and key are provided, the loaded trajectory
    is saved to the given location for faster reloads.
    """

    _readdy_trajectory: Optional[readdy.Trajectory]
    """ReaDDy trajectory object."""

    _trajectory: Optional[list[FrameData]]
    """List of FrameData for trajectory."""

    h5_file_path: str
    """Path to the ReaDDy .h5 file or .dat pickle file."""

    min_time_ix: int
    """First time index to include."""

    max_time_ix: int
    """Last time index to include."""

    time_inc: int
    """Include every time_inc timestep."""

    timestep: float
    """Real time for each simulation timestep."""

    pickle_location: Optional[str]
    """Location to save pickle file (AWS S3 bucket or local path)."""

    pickle_key: Optional[str]
    """Name of pickle file (AWS S3 bucket or local path)."""

    def __init__(
        self,
        h5_file_path: str,
        min_time_ix: int = 0,
        max_time_ix: int = -1,
        time_inc: int = 1,
        timestep: float = 100.0,
        pickle_location: Optional[str] = None,
        pickle_key: Optional[str] = None,
    ):
        self._readdy_trajectory = None
        self._trajectory = None
        self.h5_file_path = h5_file_path
        self.min_time_ix = min_time_ix
        self.max_time_ix = max_time_ix
        self.time_inc = time_inc
        self.timestep = timestep
        self.pickle_location = pickle_location
        self.pickle_key = pickle_key

    def readdy_trajectory(self) -> readdy.Trajectory:
        """
        Lazy load the ReaDDy trajectory object.

        Note that loading ReaDDy trajectories requires a path to a local file.
        Loading currently does not support S3 locations.

        Returns
        -------
        :
            The ReaDDy trajectory object.
        """
        if self._readdy_trajectory is None:
            self._readdy_trajectory = readdy.Trajectory(self.h5_file_path)
        return self._readdy_trajectory

    @staticmethod
    def _frame_edges(time_ix: int, topology_records: Any) -> list[list[int]]:
        """
        Get all edges at the given time index as [particle1 id, particle2 id].

        The ``topology_records`` object is output from
        ``readdy.Trajectory(h5_file_path).read_observable_topologies()``.
        """
        result = []
        for top in topology_records[time_ix]:
            for e1, e2 in top.edges:
                if e1 <= e2:
                    ix1 = top.particles[e1]
                    ix2 = top.particles[e2]
                    result.append([ix1, ix2])
        return result

    def _shape_trajectory_data(self) -> list[FrameData]:
        """Shape data from a ReaDDy trajectory for analysis."""
        (
            _,
            topology_records,
        ) = self.readdy_trajectory().read_observable_topologies()  # type: ignore
        (
            times,
            types,
            ids,
            positions,
        ) = self.readdy_trajectory().read_observable_particles()  # type: ignore
        result = []
        for time_ix in tqdm(range(len(times))):
            if (
                time_ix < self.min_time_ix
                or (self.max_time_ix >= 0 and time_ix > self.max_time_ix)
                or times[time_ix] % self.time_inc != 0
            ):
                continue
            frame = FrameData(time=self.timestep * time_ix)
            frame.edge_ids = ReaddyLoader._frame_edges(time_ix, topology_records)
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
                for edge in frame.edge_ids:
                    if p_id == edge[0]:
                        neighbor_ids.append(edge[1])
                    elif p_id == edge[1]:
                        neighbor_ids.append(edge[0])
                frame.particles[ids[time_ix][p]] = ParticleData(
                    uid=ids[time_ix][p],
                    type_name=self.readdy_trajectory().species_name(  # type: ignore
                        types[time_ix][p]
                    ),
                    position=np.array([position[0], position[1], position[2]]),
                    neighbor_ids=neighbor_ids,
                )
            result.append(frame)
        return result

    def trajectory(self) -> list[FrameData]:
        """
        Lazy load the shaped trajectory.

        Returns
        -------
        :
            The trajectory of data shaped for analysis.
        """

        if self._trajectory is not None:
            return self._trajectory

        if self.pickle_location is not None and self.pickle_key is not None:
            if check_key(self.pickle_location, self.pickle_key):
                print(f"Loading pickle file for ReaDDy data from {self.pickle_key}")
                self._trajectory = load_pickle(self.pickle_location, self.pickle_key)
            else:
                print(f"Loading ReaDDy data from h5 file {self.h5_file_path}")
                print(f"Saving pickle file for ReaDDy data to {self.h5_file_path}")
                self._trajectory = self._shape_trajectory_data()
                save_pickle(self.pickle_location, self.pickle_key, self._trajectory)
        else:
            print(f"Loading ReaDDy data from h5 file {self.h5_file_path}")
            self._trajectory = self._shape_trajectory_data()

        return self._trajectory
