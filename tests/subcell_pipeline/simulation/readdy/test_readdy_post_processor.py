import unittest

import numpy as np

from subcell_pipeline.simulation.readdy.post_processor import ReaddyPostProcessor


class TestReaddyPostProcessor(unittest.TestCase):
    def test_readdy_post_processor_linear_fiber_control_points_n_points(self) -> None:
        with self.assertRaises(ValueError):
            ReaddyPostProcessor.linear_fiber_control_points(
                axis_positions=[[np.array([])]], n_points=1
            )

    def test_readdy_post_processor_linear_fiber_control_points(self) -> None:
        axis_positions = np.array(
            [
                [0.0, 0.0, 4.0],
                [1.0, -1.0, 3.0],
                [2.0, -2.0, 2.0],
                [3.0, -3.0, 1.0],
                [4.0, -4.0, 0.0],
            ]
        )

        parameters = [
            (
                2,
                np.array(
                    [
                        [0.0, 0.0, 4.0],
                        [4.0, -4.0, 0.0],
                    ]
                ),
            ),
            (
                5,
                np.array(
                    [
                        [0.0, 0.0, 4.0],
                        [1.0, -1.0, 3.0],
                        [2.0, -2.0, 2.0],
                        [3.0, -3.0, 1.0],
                        [4.0, -4.0, 0.0],
                    ]
                ),
            ),
            (
                9,
                np.array(
                    [
                        [0.0, 0.0, 4.0],
                        [0.5, -0.5, 3.5],
                        [1.0, -1.0, 3.0],
                        [1.5, -1.5, 2.5],
                        [2.0, -2.0, 2.0],
                        [2.5, -2.5, 1.5],
                        [3.0, -3.0, 1.0],
                        [3.5, -3.5, 0.5],
                        [4.0, -4.0, 0.0],
                    ]
                ),
            ),
        ]

        for n_points, expected_control_points in parameters:
            with self.subTest(n_points=n_points):
                test_control_points = ReaddyPostProcessor.linear_fiber_control_points(
                    axis_positions=[[axis_positions]], n_points=n_points
                )

                self.assertTrue(
                    np.isclose(test_control_points, expected_control_points).all()
                )


if __name__ == "__main__":
    unittest.main()
