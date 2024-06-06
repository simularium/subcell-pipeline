#!/usr/bin/env python

import argparse

import numpy as np
from subcell_analysis.readdy import ReaddyLoader, ReaddyPostProcessor


class Args(argparse.Namespace):
    def __init__(self) -> None:
        self.__parse()

    def __parse(self) -> None:
        p = argparse.ArgumentParser(
            prog="readdy-actin-fiber-points",
            description=(
                "Load a ReaDDy actin trajectory and "
                "calculate actin fiber control points."
            ),
        )
        p.add_argument(
            "h5_file_path",
            type=str,
            help="The path to the ReaDDy .h5 file",
        )
        p.parse_args(namespace=self)


def main() -> None:
    args = Args()
    post_processor = ReaddyPostProcessor(
        ReaddyLoader(args.h5_file_path).trajectory(),
        box_size=600.0 * np.ones(3),
    )
    fiber_chain_ids = post_processor.linear_fiber_chain_ids(
        start_particle_phrases=["pointed"],
        other_particle_types=[
            "actin#",
            "actin#ATP_",
            "actin#mid_",
            "actin#mid_ATP_",
            "actin#fixed_",
            "actin#fixed_ATP_",
            "actin#mid_fixed_",
            "actin#mid_fixed_ATP_",
            "actin#barbed_",
            "actin#barbed_ATP_",
            "actin#fixed_barbed_",
            "actin#fixed_barbed_ATP_",
        ],
        polymer_number_range=5,
    )
    axis_positions, _ = post_processor.linear_fiber_axis_positions(
        fiber_chain_ids=fiber_chain_ids,
        ideal_positions=np.array(
            [
                [24.738, 20.881, 26.671],
                [27.609, 24.061, 27.598],
                [30.382, 21.190, 25.725],
            ]
        ),
        ideal_vector_to_axis=np.array(
            [-0.01056751, -1.47785105, -0.65833209],
        ),
    )
    fiber_points = post_processor.linear_fiber_control_points(
        axis_positions=axis_positions,
        segment_length=10.0,
    )
    print(fiber_points)


if __name__ == "__main__":
    main()
