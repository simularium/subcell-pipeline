#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import time

import numpy as np
import pandas
import psutil

from simularium_readdy_models.actin import (
    FiberData,
    ActinSimulation,
    ActinGenerator,
    ActinTestData,
)
from simularium_readdy_models import ReaddyUtil
from simulariumio import (
    DisplayData,
    DISPLAY_TYPE,
    MetaData,
    CameraData,
    UnitData,
)
from simulariumio.readdy import ReaddyConverter, ReaddyData
from simulariumio.filters import EveryNthTimestepFilter


def parse_args():
    parser = argparse.ArgumentParser(
        description="Runs and visualizes a ReaDDy branched actin simulation"
    )
    parser.add_argument(
        "params_path", help="the file path of an excel file with parameters"
    )
    parser.add_argument(
        "data_column", help="the column index for the parameter set to use"
    )
    parser.add_argument(
        "model_name", help="prefix for output file names", nargs="?", default=""
    )
    parser.add_argument(
        "replicate", help="which replicate?", nargs="?", default=""
    )
    return parser.parse_args()


def setup_parameters(args):
    parameters = pandas.read_excel(
        args.params_path,
        sheet_name="actin",
        usecols=[0, int(args.data_column)],
        dtype=object,
    )
    parameters.set_index("name", inplace=True)
    parameters.transpose()
    run_name = list(parameters)[0]
    parameters = parameters[run_name].to_dict()
    parameters["box_size"] = ReaddyUtil.get_box_size(parameters["box_size"])
    if not os.path.exists("outputs/"):
        os.mkdir("outputs/")
    parameters["name"] = (
        "outputs/" + 
        args.model_name + "_" + 
        str(run_name) + 
        ("_" + args.replicate if args.replicate else "")
    )
    return parameters


def config_init_conditions(actin_simulation):
    actin_simulation.add_obstacles()
    actin_simulation.add_membrane()
    actin_simulation.add_random_monomers()
    actin_simulation.add_random_linear_fibers(use_uuids=False)
    longitudinal_bonds = bool(actin_simulation.parameters.get("longitudinal_bonds", True))
    if bool(actin_simulation.parameters.get("orthogonal_seed", False)):
        print("Starting with orthogonal seed")
        monomers = ActinGenerator.get_monomers(
            fibers_data=[
                FiberData(
                    28,
                    [
                        np.array([-25, 0, 0]),
                        np.array([25, 0, 0]),
                    ],
                    "Actin-Polymer",
                )
            ], 
            use_uuids=False, 
            start_normal=np.array([0., 1., 0.]), 
            longitudinal_bonds=longitudinal_bonds,
        )
        monomers = ActinGenerator.setup_fixed_monomers(monomers, actin_simulation.parameters)
        actin_simulation.add_monomers_from_data(monomers)
    if bool(actin_simulation.parameters.get("branched_seed", False)):
        print("Starting with branched seed")
        actin_simulation.add_monomers_from_data(
            ActinGenerator.get_monomers(
                fibers_data=ActinTestData.simple_branched_actin_fiber(),
                use_uuids=False,
                longitudinal_bonds=longitudinal_bonds,
            )
        )


def report_hardware_usage():
    avg_load = [x / psutil.cpu_count() * 100 for x in psutil.getloadavg()]
    print(
        f"AVG load: {avg_load[0]} last min, {avg_load[1]} last 5 min, {avg_load[2]} last 15 min\n"
        f"RAM % used: {psutil.virtual_memory()[2]}\n"
        f"CPU % used: {psutil.cpu_percent()}\n"
        f"Disk % used: {psutil.disk_usage('/').percent}\n"
    )


def display_data(parameters) -> dict[str, DisplayData]:
    """
    Get DisplayData for ReaDDy actin simulations.
    """
    extra_radius = 1.5
    actin_radius = 2.0 + extra_radius
    n_polymer_numbers = 5
    result = {
        "obstacle": DisplayData(
            name="obstacle",
            display_type=DISPLAY_TYPE.SPHERE,
            radius=float(parameters["obstacle_radius"]),
            color="#8460bb",
        ),
        "membrane#outer": DisplayData(
            name="membrane#outer",
            display_type=DISPLAY_TYPE.SPHERE,
            radius=float(parameters["membrane_particle_radius"]),
            color="#8460bb",
        ),
        "membrane#inner": DisplayData(
            name="membrane#inner",
            display_type=DISPLAY_TYPE.SPHERE,
            radius=float(parameters["membrane_particle_radius"]),
            color="#8460bb",
        ),
        "membrane#outer_edge_4_1": DisplayData(
            name="membrane#outer_edge",
            display_type=DISPLAY_TYPE.SPHERE,
            radius=float(parameters["membrane_particle_radius"]),
            color="#8460bb",
        ),
        "membrane#outer_edge_2_3": DisplayData(
            name="membrane#outer_edge",
            display_type=DISPLAY_TYPE.SPHERE,
            radius=float(parameters["membrane_particle_radius"]),
            color="#8460bb",
        ),
        "membrane#inner_edge_4_1": DisplayData(
            name="membrane#inner_edge",
            display_type=DISPLAY_TYPE.SPHERE,
            radius=float(parameters["membrane_particle_radius"]),
            color="#8460bb",
        ),
        "membrane#inner_edge_2_3": DisplayData(
            name="membrane#inner_edge",
            display_type=DISPLAY_TYPE.SPHERE,
            radius=float(parameters["membrane_particle_radius"]),
            color="#8460bb",
        ),
        "actin#free": DisplayData(
            name="actin#free",
            display_type=DISPLAY_TYPE.SPHERE,
            radius=actin_radius,
            color="#bf9b30",
        ),
        "actin#free_ATP": DisplayData(
            name="actin#free",
            display_type=DISPLAY_TYPE.SPHERE,
            radius=actin_radius,
            color="#bf9b30",
        ),
        "actin#new": DisplayData(
            name="actin#new",
            display_type=DISPLAY_TYPE.SPHERE,
            radius=actin_radius,
            color="#bf9b30",
        ),
        "actin#new_ATP": DisplayData(
            name="actin#new",
            display_type=DISPLAY_TYPE.SPHERE,
            radius=actin_radius,
            color="#bf9b30",
        ),
        "actin#branch_1": DisplayData(
            name="actin#branch",
            display_type=DISPLAY_TYPE.SPHERE,
            radius=actin_radius,
            color="#a67c00",
        ),
        "actin#branch_ATP_1": DisplayData(
            name="actin#ATP_branch",
            display_type=DISPLAY_TYPE.SPHERE,
            radius=actin_radius,
            color="#a67c00",
        ),
        "actin#branch_barbed_1": DisplayData(
            name="actin#barbed",
            display_type=DISPLAY_TYPE.SPHERE,
            radius=actin_radius,
            color="#ffdc73",
        ),
        "actin#branch_barbed_ATP_1": DisplayData(
            name="actin#barbed_ATP",
            display_type=DISPLAY_TYPE.SPHERE,
            radius=actin_radius,
            color="#ffdc73",
        ),
        "arp2": DisplayData(
            name="arp#2",
            display_type=DISPLAY_TYPE.SPHERE,
            radius=actin_radius,
            color="#7230bf",
        ),
        "arp2#branched": DisplayData(
            name="arp#2",
            display_type=DISPLAY_TYPE.SPHERE,
            radius=actin_radius,
            color="#7230bf",
        ),
        "arp2#free": DisplayData(
            name="arp#2",
            display_type=DISPLAY_TYPE.SPHERE,
            radius=actin_radius,
            color="#7230bf",
        ),
        "arp3": DisplayData(
            name="arp#3",
            display_type=DISPLAY_TYPE.SPHERE,
            radius=actin_radius,
            color="#7230bf",
        ),
        "arp3#ATP": DisplayData(
            name="arp#3_ATP",
            display_type=DISPLAY_TYPE.SPHERE,
            radius=actin_radius,
            color="#7230bf",
        ),
        "arp3#new": DisplayData(
            name="arp#3",
            display_type=DISPLAY_TYPE.SPHERE,
            radius=actin_radius,
            color="#7230bf",
        ),
        "arp3#new_ATP": DisplayData(
            name="arp#3_ATP",
            display_type=DISPLAY_TYPE.SPHERE,
            radius=actin_radius,
            color="#7230bf",
        ),
    }
    for i in range(1, n_polymer_numbers + 1):
        result.update(
            {
                f"actin#{i}": DisplayData(
                    name="actin",
                    display_type=DISPLAY_TYPE.SPHERE,
                    radius=actin_radius,
                    color="#bf9b30",
                ),
                f"actin#mid_{i}": DisplayData(
                    name="actin#mid",
                    display_type=DISPLAY_TYPE.SPHERE,
                    radius=actin_radius,
                    color="#bf9b30",
                ),
                f"actin#fixed_{i}": DisplayData(
                    name="actin#fixed",
                    display_type=DISPLAY_TYPE.SPHERE,
                    radius=actin_radius,
                    color="#bf9b30",
                ),
                f"actin#mid_fixed_{i}": DisplayData(
                    name="actin#mid_fixed",
                    display_type=DISPLAY_TYPE.SPHERE,
                    radius=actin_radius,
                    color="#bf9b30",
                ),
                f"actin#ATP_{i}": DisplayData(
                    name="actin#ATP",
                    display_type=DISPLAY_TYPE.SPHERE,
                    radius=actin_radius,
                    color="#ffbf00",
                ),
                f"actin#mid_ATP_{i}": DisplayData(
                    name="actin#mid_ATP",
                    display_type=DISPLAY_TYPE.SPHERE,
                    radius=actin_radius,
                    color="#ffbf00",
                ),
                f"actin#fixed_ATP_{i}": DisplayData(
                    name="actin#fixed_ATP",
                    display_type=DISPLAY_TYPE.SPHERE,
                    radius=actin_radius,
                    color="#ffbf00",
                ),
                f"actin#mid_fixed_ATP_{i}": DisplayData(
                    name="actin#mid_fixed_ATP",
                    display_type=DISPLAY_TYPE.SPHERE,
                    radius=actin_radius,
                    color="#ffbf00",
                ),
                f"actin#barbed_{i}": DisplayData(
                    name="actin#barbed",
                    display_type=DISPLAY_TYPE.SPHERE,
                    radius=actin_radius,
                    color="#ffdc73",
                ),
                f"actin#barbed_ATP_{i}": DisplayData(
                    name="actin#barbed_ATP",
                    display_type=DISPLAY_TYPE.SPHERE,
                    radius=actin_radius,
                    color="#ffdc73",
                ),
                f"actin#fixed_barbed_{i}": DisplayData(
                    name="actin#fixed_barbed",
                    display_type=DISPLAY_TYPE.SPHERE,
                    radius=actin_radius,
                    color="#ffdc73",
                ),
                f"actin#fixed_barbed_ATP_{i}": DisplayData(
                    name="actin#fixed_barbed_ATP",
                    display_type=DISPLAY_TYPE.SPHERE,
                    radius=actin_radius,
                    color="#ffdc73",
                ),
                f"actin#pointed_{i}": DisplayData(
                    name="actin#pointed",
                    display_type=DISPLAY_TYPE.SPHERE,
                    radius=actin_radius,
                    color="#a67c00",
                ),
                f"actin#pointed_ATP_{i}": DisplayData(
                    name="actin#pointed_ATP",
                    display_type=DISPLAY_TYPE.SPHERE,
                    radius=actin_radius,
                    color="#a67c00",
                ),
                f"actin#pointed_fixed_{i}": DisplayData(
                    name="actin#pointed_fixed",
                    display_type=DISPLAY_TYPE.SPHERE,
                    radius=actin_radius,
                    color="#a67c00",
                ),
                f"actin#pointed_fixed_ATP_{i}": DisplayData(
                    name="actin#pointed_fixed_ATP",
                    display_type=DISPLAY_TYPE.SPHERE,
                    radius=actin_radius,
                    color="#a67c00",
                ),
            },
        )
    for n in range(1, 5):
        result.update({
            f"membrane#outer_edge_{n}": DisplayData(
                name=f"membrane#outer_edge",
                display_type=DISPLAY_TYPE.SPHERE,
                radius=float(parameters["membrane_particle_radius"]),
                color="#8460bb",
            ),
            f"membrane#inner_edge_{n}": DisplayData(
                name=f"membrane#inner_edge",
                display_type=DISPLAY_TYPE.SPHERE,
                radius=float(parameters["membrane_particle_radius"]),
                color="#8460bb",
            )
        })
    return result
    
    
def visualize_actin(parameters):
    path_to_readdy_h5 = parameters["name"] + ".h5"
    box_size = parameters["box_size"]
    total_steps = parameters["total_steps"]
    n_timepoints = 1000
    converter = ReaddyConverter(
        ReaddyData(
            timestep=1e-6 * (0.1 * total_steps / float(n_timepoints)),
            path_to_readdy_h5=path_to_readdy_h5,
            meta_data=MetaData(
                box_size=box_size,
                camera_defaults=CameraData(
                    position=np.array([0.0, 0.0, 250.0]),
                    look_at_position=np.array([0.0, 0.0, 0.0]),
                    fov_degrees=60.0,
                ),
                scale_factor=1.0,
            ),
            display_data=display_data(parameters),
            time_units=UnitData("ms"),
            spatial_units=UnitData("nm"),
        )
    )
    time_inc = int(converter._data.agent_data.times.shape[0] / n_timepoints)
    if time_inc >= 2:
        converter._data = converter.filter_data([EveryNthTimestepFilter(n=time_inc)])
    converter.save(output_path=path_to_readdy_h5, validate_ids=False)


def main():
    args = parse_args()
    parameters = setup_parameters(args)
    start_time = time.time()
    actin_simulation = ActinSimulation(
        parameters=parameters, 
        record=True, 
        save_checkpoints=False,
    )
    config_init_conditions(actin_simulation)
    actin_simulation.simulation.run(
        n_steps=int(actin_simulation.parameters["total_steps"]), 
        timestep=actin_simulation.parameters.get("internal_timestep", 0.1),
        show_summary=False,
    )
    print("Run time: %s seconds " % (time.time() - start_time))
    report_hardware_usage()
    visualize_actin(actin_simulation.parameters)


if __name__ == "__main__":
    main()
