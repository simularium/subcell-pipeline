# Subcellular model simulation and analysis pipeline

[![Build status](https://simularium.github.io/subcell-pipeline/_badges/build.svg)](https://github.com/simularium/subcell-pipeline/actions?query=workflow%3Abuild)
[![Lint status](https://simularium.github.io/subcell-pipeline/_badges/lint.svg)](https://github.com/simularium/subcell-pipeline/actions?query=workflow%3Alint)
[![Documentation](https://simularium.github.io/subcell-pipeline/_badges/documentation.svg)](https://simularium.github.io/subcell-pipeline/)
[![Coverage](https://simularium.github.io/subcell-pipeline/_badges/coverage.svg)](https://simularium.github.io/subcell-pipeline/_coverage/)
[![Code style](https://simularium.github.io/subcell-pipeline/_badges/style.svg)](https://github.com/psf/black)
[![License](https://simularium.github.io/subcell-pipeline/_badges/license.svg)](https://github.com/simularium/subcell-pipeline/blob/main/LICENSE)

Simulation, analysis, and visualization for subcellular models

---

## Installation

### Install with `conda` and `pip`

Use this installation method for a general user environment.
Note that with the installation, you cannot add/remove/update dependencies.

1. Create a virtual environment: `conda create -n subcell_pipeline python=3.10`
2. Activate the environment: `conda activate subcell_pipeline`
3. Install all dependencies: `make install`

Or,

1. Create a virtual environment: `conda create -n subcell_pipeline python=3.10`
2. Activate the environment: `conda activate subcell_pipeline`
3. Install conda-specific dependencies: `conda env update --file environment.yml --prune`
4. Install dependencies: `pip install -r requirements.txt`
5. Install the project in editable mode: `pip install -e .`

### Install with `conda` and `pdm`

Use this installation method for a complete development environment.

1. Create a virtual environment: `conda create -n subcell_pipeline python=3.10`
2. Activate the environment: `conda activate subcell_pipeline`
3. Install all dependencies: `make install DEV=1`

Or,

1. Create a virtual environment: `conda create -n subcell_pipeline python=3.10`
2. Activate the environment: `conda activate subcell_pipeline`
3. Install conda-specific dependencies: `conda env update --file environment.yml --prune`
4. Install dependencies: `pdm sync`

### Install with `pyenv` and `pdm`

Use this installation method for a (mostly) complete, non-Conda development environment.
Note that this installation method does not include the `readdy` package for ReaDDy-specific post-processsing; if needed, use a `conda` installation method.

1. Install Python 3.10 or higher with `pyenv`
2. Install dependencies: `pdm sync`
3. Activate the environment: `source .venv/bin/activate`

## Usage

The repository contains three major pipeline modules: `simulation`, `analysis`, and `visualization`.

### Simulations

The `simulation` module contains code for initializing, simulating, and post-processing simulations from different simulators.
The module is further organized by simulator.

- [simulation.cytosim](https://github.com/simularium/subcell-pipeline/blob/main/subcell_pipeline/simulation/cytosim) -- Simulations and processing for cytoskeleton simulation engine [Cytosim](https://gitlab.com/f-nedelec/cytosim)
- [simulation.readdy](https://github.com/simularium/subcell-pipeline/blob/main/subcell_pipeline/simulation/readdy) -- Simulations and processing for particle-based reaction-diffusion simulator [ReaDDy](https://readdy.github.io/)

### Analysis

The `analysis` module contains code for different analyses.
Each analysis type contains a README with additional information:

- [analysis.compression_metrics](https://github.com/simularium/subcell-pipeline/blob/main/subcell_pipeline/analysis/compression_metrics) -- Analysis and plotting for compression metrics on individual fiber shapes
- [analysis.dimensionality_reduction](https://github.com/simularium/subcell-pipeline/blob/main/subcell_pipeline/analysis/dimensionality_reduction) -- Analysis and plotting for dimensionality reduction on individual fiber shapes
- [analysis.tomography_data](https://github.com/simularium/subcell-pipeline/blob/main/subcell_pipeline/analysis/tomography_data) -- Processing and analysis of cryo-electron tomography data
- [analysis.wall_clock_time](https://github.com/simularium/subcell-pipeline/blob/main/subcell_pipeline/analysis/wall_clock_time) -- Analysis of simulation wall clock times

### Visualization

The `visualization` module contains code for visualizing simulation and analysis outputs.

## Glossary

Definitions of some terms used in this repo.

- *polymer trace*

    Refers to the line traced by a polymer backbone in three dimensions. For Cytosim, this corresponds to the positions of the control points. For ReaDDy, this corresponds to a derived metric that traces the polymer backbone

- *end-to-end axis*

    Refers to the line connecting the first and last points of a polymer trace
