#!/usr/bin/env python
import numpy as np
import matplotlib as plt
# TODO: consider creating a fiber class?


def asymmetry(fibers_df):
    last_timepoint = fibers_df.loc[timepoints[-1]]
    last_timepoint_tension = last_timepoint['segment_energy']
    diff = np.zeros(len(last_timepoint_tension))
    for index, timepoint in enumerate(last_timepoint_tension):
        middle_index = np.round(fiber_at_time.shape[0] / 2).astype(int)
        diff[index] = np.abs(last_timepoint_tension[index] - last_timepoint_tension[-1-index])
    xs = np.linspace(0,1,len(last_timepoint_tension))
    plt.scatter(xs, diff)
    plt.xlabel('Position along filament')
    plt.ylabel('Level of Asymmetry')
        #print(diff[index])
    print(len(diff))
    #write 


def get_axis_distances_and_projections(fiber_points):
    """
    Returns the distances of fiber points from the axis

    Parameters
    ----------
    fiber_points: [n x 3] numpy array
        array containing the x,y,z positions of fiber points
        at a given time

    Returns
    ----------
    perp_distances: [n x 1] numpy array
        perpendicular distances of fiber points from the axis

    scaled_projections: [n x 1] numpy array
        length of fiber point projections along the axis, scaled
        by axis length.
        Can be negative.    
    """
    axis = fiber_points[-1] - fiber_points[0]
    axis_length = np.linalg.norm(axis)

    position_vectors = fiber_points - fiber_points[0]
    dot_products = np.dot(position_vectors, axis)

    projections = dot_products / axis_length
    projection_positions = fiber_points[0] + projections[:, None] * axis / axis_length

    perp_distances = np.linalg.norm(fiber_points - projection_positions, axis=1)
    scaled_projections = projections / axis_length

    return perp_distances, scaled_projections, projection_positions


def get_asymmetry_of_peak(fiber_points):
    """
    returns the scaled distance of the projection of the peak from the axis midpoint
    
    Parameters
    ----------
    fiber_points: [n x 3] numpy array
        array containing the x,y,z positions of fiber points
        at a given time
    
    Returns
    ----------
    peak_asym: float
        scaled distance of the projection of the peak from the axis midpoint
    """
    perp_distances, scaled_projections, _ = get_axis_distances_and_projections(fiber_points=fiber_points)
    projection_of_peak = scaled_projections[perp_distances == np.max(perp_distances)]
    peak_asym = np.max(projection_of_peak - 0.5)  # max kinda handles multiple peaks

    return peak_asym


def get_total_fiber_twist(fiber_points):
    """
    Returns the sum of angles between consecutive vectors from fiber points
    to the axis
    
    Parameters
    ----------
    fiber_points: [n x 3] numpy array
        array containing the x,y,z positions of fiber points
        at a given time
    
    Returns
    ----------
    total_twist: float
        sum of angles between vectors from fiber points to axis
    """
    _, _, projection_positions = get_axis_distances_and_projections(fiber_points=fiber_points)
    perp_vectors = fiber_points - projection_positions
    perp_vectors = perp_vectors / np.linalg.norm(perp_vectors, axis=1)[:, None]
    consecutive_angles = np.arccos(np.einsum('ij,ij->i', perp_vectors[1:], perp_vectors[:-1]))
    total_twist = np.nansum(consecutive_angles)

    return total_twist