
import argparse
import numpy as np
import pandas as pd
from simulariumio import (
    TrajectoryData, 
    TrajectoryConverter, 
    MetaData, 
    AgentData, 
    UnitData,
    DimensionData,
    CameraData,
    DisplayData,
    DISPLAY_TYPE,
    HistogramPlotData,
)


SCALE = 0.1
MIN_COMPRESSION_BIN = 2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualizes tomography data"
    )
    parser.add_argument(
        "csv_path", help="file path for CSV tomography data"
    )
    return parser.parse_args()

def get_spatial_center_and_size(tomo_df):
    ixs = [
        list(tomo_df.columns).index("xpos"),
        list(tomo_df.columns).index("ypos"),
        list(tomo_df.columns).index("zpos"),
    ]
    unique_values = list(map(set, tomo_df.values.T))
    mins = []
    maxs = []
    for dim_ix in range(3):
        d_values = np.array(list(unique_values[ixs[dim_ix]]))
        mins.append(np.amin(d_values))
        maxs.append(np.amax(d_values))
    mins = np.array(mins)
    maxs = np.array(maxs)
    return mins + 0.5 * (maxs - mins), maxs - mins

def empty_plots():
    return {
        "CONTOUR_LENGTH" : HistogramPlotData(
            title="Contour Length",
            xaxis_title="filament contour length (nm)",
            traces={},
        ),
        "COMPRESSION_RATIO" : HistogramPlotData(
            title="Compression Percentage",
            xaxis_title="percent (%)",
            traces={},
        ),
        "AVERAGE_PERP_DISTANCE" : HistogramPlotData(
            title="Average Perpendicular Distance",
            xaxis_title="distance (nm)",
            traces={},
        ),
        "CALC_BENDING_ENERGY" : HistogramPlotData(
            title="Bending Energy",
            xaxis_title="energy",
            traces={},
        ),
        "NON_COPLANARITY" : HistogramPlotData(
            title="Non-coplanarity",
            xaxis_title="3rd component variance from PCA",
            traces={},
        ),
        "PEAK_ASYMMETRY" : HistogramPlotData(
            title="Peak Asymmetry",
            xaxis_title="normalized peak distance",
            traces={},
        ),
    }

def add_plots(tomo_df, converter):
    plots = empty_plots()
    for metric_name in plots:
        col_ix = list(tomo_df.columns).index(metric_name)
        plots[metric_name].traces["actin"] = np.array(list(list(map(set, tomo_df.values.T))[col_ix]))
        if metric_name == "COMPRESSION_RATIO":
            plots[metric_name].traces["actin"] *= 100.
        converter.add_plot(plots[metric_name], "histogram")

def main():
    args = parse_args()
    tomo_df = pd.read_csv(args.csv_path)
    tomo_df = tomo_df.sort_values(by=["COMPRESSION_RATIO", "monomer_ids"])
    tomo_df = tomo_df.reset_index(drop=True)
    names, ixs = np.unique(np.array(list(tomo_df["repeat"])), return_index=True)
    fiber_names = names[np.argsort(ixs)]
    center, box_size = get_spatial_center_and_size(tomo_df)
    max_points = 0
    subpoints = []
    compression_ratios = []
    for fiber_name in fiber_names:
        fiber_df = tomo_df.loc[tomo_df["repeat"] == fiber_name]
        points = np.array(fiber_df[["xpos", "ypos", "zpos"]]) - center
        subpoints.append(SCALE * points.flatten())
        compression_ratios.append(list(fiber_df["COMPRESSION_RATIO"])[0])
        if len(fiber_df) > max_points:
            max_points = len(fiber_df)
    n_agents = len(subpoints)
    compression_ratios = 100. * np.array(compression_ratios)
    min_compression_ratio = np.amin(compression_ratios)
    max_compression_ratio = np.amax(compression_ratios)
    bins = np.linspace(min_compression_ratio, max_compression_ratio, 100)
    digitized = np.digitize(compression_ratios, bins)
    type_names = []
    display_data = {}
    type_name_min = f"actin less than {MIN_COMPRESSION_BIN}.0 percent compressed"
    for agent_ix in range(n_agents):
        bin_percent = int(10 * bins[digitized[agent_ix] - 1]) / 10.
        if bin_percent < MIN_COMPRESSION_BIN:
            type_name = type_name_min
        else:
            type_name = f"actin {bin_percent} percent compressed"
        type_names.append(type_name)
        if type_name not in display_data:
            display_data[type_name] = DisplayData(
                name=type_name,
                display_type=DISPLAY_TYPE.FIBER,
            )
    display_data[type_name_min] = DisplayData(
        name=type_name_min,
        display_type=DISPLAY_TYPE.FIBER,
        color="#222222",
    )
    agent_data = AgentData.from_dimensions(DimensionData(
        total_steps=1,
        max_agents=n_agents,
        max_subpoints=3 * max_points,
    ))
    agent_data.n_agents[0] = n_agents
    agent_data.viz_types[0] = 1001.0 * np.ones(n_agents)
    agent_data.unique_ids[0] = np.arange(n_agents)
    agent_data.types[0] = type_names
    agent_data.radii *= 0.5
    for agent_ix in range(n_agents):
        n_subpoints = subpoints[agent_ix].shape[0]
        agent_data.n_subpoints[0][agent_ix] = n_subpoints
        agent_data.subpoints[0][agent_ix][:n_subpoints] = subpoints[agent_ix]
    agent_data.display_data = display_data
    traj_data = TrajectoryData(
        meta_data=MetaData(
            box_size=SCALE * box_size,
            camera_defaults=CameraData(position=np.array([0.0, 0.0, 70.0]))
        ),
        agent_data=agent_data,
        spatial_units=UnitData("um", 0.003),
    )
    converter = TrajectoryConverter(traj_data)
    add_plots(tomo_df, converter)
    converter.save(args.csv_path)


if __name__ == "__main__":
    main()