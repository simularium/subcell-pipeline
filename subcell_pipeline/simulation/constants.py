"""Constants for parsing simulations."""


COLUMN_NAMES = [
    "fiber_id",
    "xpos",
    "ypos",
    "zpos",
    "xforce",
    "yforce",
    "zforce",
    "segment_curvature",
    "time",
    "fiber_point",
]

COLUMN_DTYPES = {
    "fiber_id": int,
    "xpos": float,
    "ypos": float,
    "zpos": float,
    "xforce": float,
    "yforce": float,
    "zforce": float,
    "segment_curvature": float,
    "time": float,
    "fiber_point": int,
}