# subcell-analysis

[![Build Status](https://github.com/Simularium/subcell-analysis/workflows/CI/badge.svg)](https://github.com/Simularium/subcell-analysis/actions)
[![Documentation](https://github.com/Simularium/subcell-analysis/workflows/Documentation/badge.svg)](https://Simularium.github.io/subcell-analysis)

Analysis functionality for subcellular models

---

## Installation

### Install with `conda` and `pip`

Use this installation method for a general user environment.
Note that with the installation, you cannot add/remove/update dependencies.

1. Create a virtual environment: `conda create -n subcell_analysis python=3.10`
2. Activate the environment: `conda activate subcell_analysis`
3. Install conda-specific dependencies: `conda env update --file environment.yml --prune`
4. Install dependencies: `pip install -r requirements.txt`
5. Install the project in editable mode: `pip install -e .`

### Install with `conda` and `pdm`

Use this installation method for a complete development environment.

1. Create a virtual environment: `conda create -n subcell_analysis python=3.10`
2. Activate the environment: `conda activate subcell_analysis`
3. Install conda-specific dependencies: `conda env update --file environment.yml --prune`
4. Install dependencies: `pdm sync`

### Install with `pyenv` and `pdm`

Use this installation method for a (mostly) complete, non-Conda development environment.
Note that this installation method does not include the `readdy` package for ReaDDy-specific post-processsing; if needed, use a `conda` installation method.

1. Install Python 3.10 or higher with `pyenv`
2. Install dependencies: `pdm sync`
3. Activate the environment: `source .venv/bin/activate`

## Usage

### 1. Run Simulations
The notebook aws_batch_job_submission.py can be used for configuring and executing simulations. Before this notebook can be succesfully executed, one must setup their AWSCLI credentials for submitting jobs and job definitions via the CLI. This notebook has five parts: 1a.) upload configuration files to s3, which involves generating the configuration files based on the template (vary_compress_rate.cym.tpl). Next, we have to create and register our job definition. In AWS Batch, a job definition specifies how many jobs are to be run, which docker image to use, how many CPUs to use, and the command a container should run when it is started.  We then submit our job to AWS using the submit_batch_job function. We can monitor these simulations as needed using check_batch_job. Finally, we can load in our results using the create_dataframes_for_repeats function which requires a bucket name, number of repeats, configs, and a save folder.

### 2. Post process to create dataframes from simulation data
Involves running scripts in `subcell_analysis/postprocessing`. This also creates subsampled dataframes for further processing and metric calculation.

### 3. Tomography data analysis
Selecting the tomogram filaments for analysis.

### 3. Calculate metrics

### 4. Visualize in simularium

### 5. Dimensionality reduction using PCA/PaCMAP
Runs through generate_figure_data.ipynb


## Glossary of terms
Definitions of some terms used in these analyses
* *polymer trace*:

    refers to the line traced by a polymer backbone in three dimensions. For cytosim, this corresponds to the positions of the control points. For ReaDDy, this corresponds to a derived metric that traces the polymer backbone

* *end-to-end axis*:

    refers to the line connecting the first and last points of a polymer trace

**Apache Software License 2.0**
