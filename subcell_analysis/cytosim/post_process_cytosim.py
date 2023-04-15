from typing import Any

import pandas as pd


# TODO add actual types
def read_fiber_forces(input_file: Any) -> Any:
    # input_file is'fiber_segment_curvature.txt'
    fiber_forces = open(input_file)
    return fiber_forces.readlines()


# TODO add actual types
def all_fibers(input_file: Any) -> Any:
    # input_file is'fiber_segment_curvature.txt'
    fibers = read_fiber_forces(input_file)
    timepoints_forces = []
    outputs = []
    fid = 0
    fiber_point = 0
    for line in fibers:
        line = line.strip()
        if line.startswith("%"):
            if line.startswith("% time"):
                time = float(line.split(" ")[-1])
                timepoints_forces.append(time)
                singles: Any = {}
            elif line.startswith("% end"):
                df = pd.DataFrame.from_dict(singles, orient="index")
                outputs.append(df)
                #                     fiber_point=0
                fid = 0
                # print 'finished parsing ' + rundir + ' timepoint ' + str(time)
        elif len(line.split()) > 0:
            [
                fiber_id,
                xpos,
                ypos,
                zpos,
                xforce,
                yforce,
                zforce,
                segment_curvature,
            ] = line.split()
            #                 figure out if you're on the first, second fiber point etc
            if int(fid) == int(fiber_id):
                fiber_point += 1
            #                     print(fiber_point)
            else:
                fiber_point = 0
                fid += 1
        #                     print('id: '+str(fid))
        singles[str(fiber_id) + "_" + str(fiber_point)] = {
            "fiber_id": int(fiber_id),
            "xpos": float(xpos),
            "ypos": float(ypos),
            "zpos": float(zpos),
            "xforce": float(xforce),
            "yforce": float(yforce),
            "zforce": float(zforce),
            "segment_curvature": float(segment_curvature),
        }

    all_outputs = pd.concat(outputs, keys=timepoints_forces, names=["time", "id"])

    return all_outputs
