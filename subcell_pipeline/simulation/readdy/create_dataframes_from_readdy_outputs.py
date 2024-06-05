import os
from typing import List

import numpy as np
import pandas as pd
from subcell_analysis.readdy import ReaddyLoader, ReaddyPostProcessor
from subcell_analysis.readdy.readdy_post_processor import array_to_dataframe

IDEAL_ACTIN_POSITIONS = np.array(
    [
        [24.738, 20.881, 26.671],
        [27.609, 24.061, 27.598],
        [30.382, 21.190, 25.725],
    ]
)
IDEAL_ACTIN_VECTOR_TO_AXIS = np.array([-0.01056751, -1.47785105, -0.65833209])


def _load_readdy_fiber_points(h5_file_path, box_size, n_points_per_fiber):
    readdy_loader = ReaddyLoader(str(h5_file_path))
    readdy_post_processor = ReaddyPostProcessor(
        readdy_loader.trajectory(),
        box_size=box_size,
    )
    fiber_chain_ids = readdy_post_processor.linear_fiber_chain_ids(
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
    axis_positions, _ = readdy_post_processor.linear_fiber_axis_positions(
        fiber_chain_ids=fiber_chain_ids,
        ideal_positions=IDEAL_ACTIN_POSITIONS,
        ideal_vector_to_axis=IDEAL_ACTIN_VECTOR_TO_AXIS,
    )
    fiber_points = readdy_post_processor.linear_fiber_control_points(
        axis_positions=axis_positions,
        n_points=n_points_per_fiber,
    )
    return np.array(fiber_points)


def generate_readdy_df(
    input_h5_file_dir: str = "data/aws_downloads/",
    output_dir: str = "data/dataframes/readdy/",
    n_points_per_fiber: int = 50,
    box_size: np.ndarray = np.array(3 * [600.0]),
    num_repeats: int = 5,
    compression_velocities: List[float] = [4.7, 15, 47, 150],
    reprocess: bool = True,
) -> pd.DataFrame:
    result = []
    os.makedirs(output_dir, exist_ok=True)
    for velocity in compression_velocities:
        for repeat in range(num_repeats):
            file_name = f"actin_compression_velocity={velocity}_{repeat}.h5"
            df_save_path = os.path.join(
                output_dir,
                f"readdy_actin_compression_velocity_{velocity}_repeat_{repeat}.csv",
            )
            if os.path.exists(df_save_path) and not reprocess:
                print(f"{file_name} already processed")
                df_points = pd.read_csv(df_save_path)
                result.append(df_points)
                continue
            h5_file_path = os.path.join(input_h5_file_dir, file_name)
            if not os.path.exists(h5_file_path):
                print(f"{file_name} not found")
                continue
            print(f"Processing {file_name}")
            fiber_points = _load_readdy_fiber_points(
                str(h5_file_path), box_size, n_points_per_fiber
            )
            df_points = array_to_dataframe(fiber_points)
            df_points.reset_index(inplace=True)
            df_points.rename(columns={0: "xpos", 1: "ypos", 2: "zpos"}, inplace=True)
            df_points["velocity"] = velocity
            df_points["repeat"] = repeat
            df_points["simulator"] = "readdy"
            df_points["normalized_time"] = (
                df_points["time"] - df_points["time"].min()
            ) / (df_points["time"].max() - df_points["time"].min())
            df_points.to_csv(
                df_save_path,
                index=False,
            )
            result.append(df_points)
    return pd.concat(result)


if __name__ == "__main__":
    output_dir = "data/dataframes/readdy/"
    df_readdy = generate_readdy_df(output_dir=output_dir)
    df_readdy.to_csv(
        output_dir / "readdy_actin_compression_all_velocities_and_repeats.csv"
    )
    df_readdy.to_parquet(
        output_dir / "readdy_actin_compression_all_velocities_and_repeats.parquet"
    )
