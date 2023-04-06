"""Top-level package for subcell_analysis."""

from importlib.metadata import PackageNotFoundError, version


try:
    __version__ = version("subcell-analysis")
except PackageNotFoundError:
    __version__ = "uninstalled"

__author__ = "Blair Lyons"
__email__ = "blair208@gmail.com"


