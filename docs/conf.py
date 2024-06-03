# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import importlib.metadata
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath("../subcell_anlysis"))

# -- Project information -----------------------------------------------------

project = "Subcell analysis"
copyright = f"{datetime.now().year}, Allen Institute for Cell Science"
release = importlib.metadata.version("subcell-analysis")

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "myst_parser",
    "sphinx_copybutton",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Decides whether module names are prepended to all object names.
add_module_names = False

# Default options for autodoc directives.
autodoc_default_options = {
    "member-order": "bysource",
    "undoc-members": True,
}

# List of modules to be mocked up. Useful when some external dependencies are
# not met at build time and break the building process.
autodoc_mock_imports = []

# Controls how to represent typehints.
autodoc_typehints = "signature"

# Selects how the signature will be displayed for the class.
autodoc_class_signature = "separated"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "furo"

# The name for this set of Sphinx documents.
html_title = f"<strong>{project}</strong> <br />{release}"
