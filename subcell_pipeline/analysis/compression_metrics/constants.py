# Constants used in compression metric analysis.

ABSOLUTE_TOLERANCE: float = 1e-6
"""
The absolute tolerance for vector length.
Vectors smaller than this value are considered zero.
"""
SIMULATOR_COLOR_MAP: dict[str, str] = {
    "readdy": "#ca562c",
    "cytosim": "#008080",
}
"""Map of simulator name to color."""
DEFAULT_BENDING_CONSTANT: float = 1.0
"""The default bending constant in pN nm."""
