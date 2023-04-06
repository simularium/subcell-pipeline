#!/usr/bin/env python
import numpy as np
import matplotlib as plt




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
