#!/usr/bin/env python

from typing import List

import numpy as np
import pytest

from subcell_analysis.readdy import ReaddyLoader, ReaddyPostProcessor


@pytest.mark.parametrize(
    "axis_positions, segment_length, expected_control_points",
    [
        (
            [
                np.array([-2.47203989e02, 5.70044691e-03, -2.56005623e-02]),
                np.array([-2.44400890e02, -1.13344916e-02, 2.36519172e-02]),
                np.array([-2.41597792e02, 1.63829372e-02, -2.04812904e-02]),
                np.array([-2.38794693e02, -2.05849546e-02, 1.62524931e-02]),
                np.array([-2.35991594e02, 2.37234456e-02, -1.11840070e-02]),
                np.array([-2.33188495e02, -2.56362595e-02, 5.53769673e-03]),
                np.array([-2.30385396e02, 2.62245702e-02, 3.94719939e-04]),
                np.array([-2.27582298e02, -2.54579825e-02, -6.30674332e-03]),
                np.array([-2.24779199e02, 2.33761024e-02, 1.18929274e-02]),
                np.array([-2.21976100e02, -2.00864907e-02, -1.68646606e-02]),
            ],
            10.0,
            np.array(
                [
                    [-2.47203989e02, 5.70044691e-03, -2.56005623e-02],
                    [-2.37205717e02, 4.53188350e-03, 6.99728204e-04],
                    [-2.27207445e02, -1.89274956e-02, -3.87293686e-03],
                ]
            ),
        ),
        (
            [
                np.array([-2.47203989e02, 5.70044691e-03, -2.56005623e-02]),
                np.array([-2.44400890e02, -1.13344916e-02, 2.36519172e-02]),
                np.array([-2.41597792e02, 1.63829372e-02, -2.04812904e-02]),
                np.array([-2.38794693e02, -2.05849546e-02, 1.62524931e-02]),
                np.array([-2.35991594e02, 2.37234456e-02, -1.11840070e-02]),
                np.array([-2.33188495e02, -2.56362595e-02, 5.53769673e-03]),
                np.array([-2.30385396e02, 2.62245702e-02, 3.94719939e-04]),
                np.array([-2.27582298e02, -2.54579825e-02, -6.30674332e-03]),
                np.array([-2.24779199e02, 2.33761024e-02, 1.18929274e-02]),
                np.array([-2.21976100e02, -2.00864907e-02, -1.68646606e-02]),
            ],
            1.0,
            np.array(
                [
                    [-2.47203989e02, 5.70044691e-03, -2.56005623e-02],
                    [-2.46204162e02, -3.75684012e-04, -8.03287259e-03],
                    [-2.45204335e02, -6.45181494e-03, 9.53481714e-03],
                    [-2.44204507e02, -9.39263048e-03, 2.05599797e-02],
                    [-2.43204680e02, 4.93799453e-04, 4.81826461e-03],
                    [-2.42204853e02, 1.03802294e-02, -1.09234504e-02],
                    [-2.41205026e02, 1.12030548e-02, -1.53342109e-02],
                    [-2.40205199e02, -1.98288954e-03, -2.23176981e-03],
                    [-2.39205371e02, -1.51688338e-02, 1.08706713e-02],
                    [-2.38205544e02, -1.12723204e-02, 1.04859547e-02],
                ]
            ),
        ),
    ],
)
def test_readdy_control_points(
    axis_positions: List[np.ndarray],
    segment_length: float,
    expected_control_points: np.ndarray,
) -> None:
    post_processor = ReaddyPostProcessor(
        trajectory=ReaddyLoader(
            h5_file_path="subcell_analysis/tests/data/readdy/actin_ortho_filament_10_steps.h5",
            timestep=0.1,
        ).trajectory(),
        box_size=np.array(3 * [600.0]),
        periodic_boundary=False,
    )
    test_control_points = post_processor.linear_fiber_control_points(
        axis_positions=[[axis_positions]],
        segment_length=segment_length,
    )[0][0][0:10]
    assert False not in np.isclose(test_control_points, expected_control_points)
