#!/usr/bin/env python

from typing import List, Tuple

import numpy as np
import scipy.linalg as linalg

from .utils import get_unit_vector

X_AXIS: np.ndarray = np.array([1., 0., 0.])
Y_AXIS: np.ndarray = np.array([0., 1., 0.])

class SpatialAligner:
    @staticmethod
    def _last_normals(
        data: np.ndarray
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Get the normal to the farthest point from the x-axis 
        for each fiber at the last time point.
        """
        normals = []
        points_per_fiber = int(data.shape[2] / 3)
        perp_vectors = np.zeros((data.shape[0], data.shape[1], points_per_fiber, 3))
        for fiber_ix in range(data.shape[1]):
            for time_ix in range(data.shape[0]):
                polymer_trace = data[time_ix][fiber_ix].reshape((points_per_fiber, 3))
                position_vectors = polymer_trace - polymer_trace[0]
                projections = np.dot(position_vectors, X_AXIS)
                projection_positions = polymer_trace[0] + projections[:, None] * X_AXIS
                perp_vectors[time_ix][fiber_ix] = polymer_trace - projection_positions
            perp_distances = np.linalg.norm(perp_vectors[-1][fiber_ix], axis=0)
            max_ix = np.argmax(perp_distances)
            normals.append(get_unit_vector(perp_vectors[-1][fiber_ix][max_ix])[0])
        return normals, perp_vectors
    
    @staticmethod
    def _get_angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Get the angle between two vectors in radians.
        """
        return np.arccos(
            np.clip(
                np.dot(get_unit_vector(v1)[0], get_unit_vector(v2)[0]), -1.0, 1.0
            )
        )
    
    @staticmethod
    def _rotate(vector: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
        """
        Rotate a vector around axis by angle (radians).
        """
        rotation = linalg.expm(np.cross(np.eye(3), get_unit_vector(axis)[0] * angle))
        return np.dot(rotation, np.copy(vector))
    
    @staticmethod
    def align_fibers_y(
        data: np.ndarray
    ) -> np.ndarray:
        """
        Rotationally align the given fibers around the x-axis so that 
        the farthest position from the x-axis points up along the y-axis.

        Parameters
        ----------
        fiber_points: np.ndarray (shape = time x fiber x (3 * points_per_fiber))
            Array containing the flattened x,y,z positions of control points
            for each fiber at each time.
            
        Returns
        ----------
        aligned_data: np.ndarray
            The given data aligned.        
        """
        normals, perp_vectors = SpatialAligner._last_normals(data)
        aligned_data = np.zeros_like(data)
        points_per_fiber = int(data.shape[2] / 3)
        for fiber_ix in range(data.shape[1]):
            angle = SpatialAligner._get_angle_between_vectors(normals[fiber_ix], Y_AXIS)
            for time_ix in range(data.shape[0]):
                fiber_points = data[time_ix][fiber_ix].reshape((points_per_fiber, 3))
                for point_ix in range(points_per_fiber):
                    x_coord = fiber_points[point_ix][0]
                    new_normal = SpatialAligner._rotate(
                        vector=perp_vectors[time_ix][fiber_ix][point_ix], 
                        axis=X_AXIS,
                        angle=angle,
                    )
                    aligned_data[time_ix][fiber_ix][3 * point_ix:3 * point_ix + 3] = np.array([x_coord, 0., 0.]) + new_normal
        return aligned_data
